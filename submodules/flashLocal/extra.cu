#include "extra.h"
#include "types.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <new>
#include <queue>
#include <thrust/device_vector.h>
#include <vector>

namespace {

namespace kernel {

constexpr auto kGroupLevel    = 12;
constexpr auto kGroupRatioMin = 0.9f;
constexpr auto kGroupRatioMax = 1 / kGroupRatioMin;

static_assert(kGroupRatioMin <= 1.0f && kGroupRatioMax >= 1.0f);

struct irange {
public:
    struct iterator {
    public:
        iterator(int value) : value(value) {}
        iterator &operator++() {
            value++;
            return *this;
        }
        auto operator*() const -> int {
            return value;
        }
        auto operator!=(const iterator &other) const -> int {
            return value != other.value;
        }

    private:
        int value;
    };

    explicit irange(int finish) : start(0), finish(finish) {}
    explicit irange(int start, int finish) : start(start), finish(finish) {}

    auto begin() const -> iterator {
        return iterator(start);
    }
    auto end() const -> iterator {
        return iterator(finish);
    }

private:
    int start;
    int finish;
};

enum class ReorderType {
    NotReordered,
    BFSOrder,
    GroupOrder,
} order_type = ReorderType::NotReordered;

auto requires_type(ReorderType type) -> void {
    if (order_type != type) [[unlikely]]
        throw std::runtime_error("Error: wrong order type");
}

auto requires_not_type(ReorderType type) -> void {
    if (order_type == type) [[unlikely]]
        throw std::runtime_error("Error: wrong order type");
}

struct GroupInfo {
    int group_offset; // offset of group head
    int child_offset; // offset of first child
};

struct Holder {
private:
    int atomic_counter;        // need atomic data on device
    char _[256 - sizeof(int)]; // padding for alignment
public:
    __device__ auto reset() -> void {
        static_assert(sizeof(_[0]) == 1, "to avoid warning");
        atomic_counter = 0;
    }
    __device__ auto increment(int n = 1) -> int {
        return atomicAdd(&atomic_counter, n);
    }
    __host__ auto device_value() const & -> int {
        auto value = int{};
        cudaMemcpy(&value, this, sizeof(int), cudaMemcpyDeviceToHost);
        return value;
    }
    // to avoid copying on host, since the value is in device
    __host__ auto device_value() && = delete;

    // construct a holder from int pointer
    __host__ static auto from_ptr(int *ptr) -> std::pair<Holder *, int *> {
        constexpr auto off = 256 / sizeof(int);
        return {std::launder(reinterpret_cast<Holder *>(ptr)), ptr + off};
    }
};

struct Package {
    thrust::device_vector<char> temp_storage;
    thrust::device_vector<bool> render_counts;
    thrust::device_vector<int> render_offsets;
    thrust::device_vector<bool> is_smaller;
    thrust::device_vector<Holder> holder;
    thrust::device_vector<GroupInfo> groups; // group heads offsets
    thrust::device_vector<float> sizes[2];   // sizes of the nodes
    thrust::device_vector<int> slots;        // slots for group dispatch

    int group_start_index;
    int group_finish_index;

    auto resize(int N) -> void {
        const auto kHack = 78079; // this is a hacked size.
        temp_storage.resize(kHack);
        render_counts.resize(N);
        render_offsets.resize(N);
        is_smaller.resize(N);
        holder.resize(2);
        sizes[0].resize(N);
        sizes[1].resize(N);
    }
};

auto &package = *new Package{}; // this is intended leak to skip destructor

std::vector<int> reverse_map{};
std::vector<int> depth_count_global{};

static __global__ auto transform_node(SimpleNode *nodes, int N, const Node *from) -> void {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    nodes[idx] = SimpleNode(from[idx]);
}

static __global__ auto transform_node(Node *nodes, int N, const SimpleNode *from) -> void {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    nodes[idx] = Node(from[idx]);
}

auto to_simple(Node *node, int N) -> void {
    thrust::device_vector<Node> tmp_nodes(N);
    cudaMemcpy(tmp_nodes.data().get(), node, N * sizeof(Node), cudaMemcpyDeviceToDevice);
    auto *ptr = reinterpret_cast<SimpleNode *>(node);
    transform_node<<<(N + 255) / 256, 256>>>(ptr, N, tmp_nodes.data().get());
}

[[maybe_unused]]
auto to_normal(Node *node, int N) -> void {
    thrust::device_vector<SimpleNode> tmp_nodes(N);
    cudaMemcpy(tmp_nodes.data().get(), node, N * sizeof(SimpleNode), cudaMemcpyDeviceToDevice);
    transform_node<<<(N + 255) / 256, 256>>>(node, N, tmp_nodes.data().get());
}

auto do_reorder(
    int N, Node *gpu_nodes, const Node *cpu_nodes, Box *gpu_boxes, const Box *cpu_boxes,
    const std::vector<int> &new_index
) -> void {
    requires_not_type(ReorderType::NotReordered);

    auto new_nodes = std::vector<Node>(N);
    auto new_boxes = std::vector<Box>(N);

    // reoreder the nodes
    for (const auto i : irange(N)) {
        new_nodes[new_index[i]] = cpu_nodes[i];
        new_boxes[new_index[i]] = cpu_boxes[i];
    }

    for (const auto i : irange(N))
        if (new_nodes[i].parent != -1)
            new_nodes[i].parent = new_index[new_nodes[i].parent];

    std::cerr << "Reorder nodes: " << N << std::endl;
    reverse_map.resize(N);
    for (const auto i : irange(N))
        reverse_map[new_index[i]] = i;

    auto depths     = std::vector<int>(N, 0);
    auto depths_map = std::map<int, int>{};
    for (const auto i : irange(N)) {
        depths[i] = new_nodes[i].parent == -1 ? 0 : depths[new_nodes[i].parent] + 1;
        depths_map[depths[i]]++;
    }

    // set the depth count global
    depth_count_global.clear();
    for (const auto &[_, count] : depths_map)
        depth_count_global.push_back(count);

    cudaMemcpy(gpu_nodes, new_nodes.data(), N * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_boxes, new_boxes.data(), N * sizeof(Box), cudaMemcpyHostToDevice);
}

[[gnu::noinline]] // Don't inline this big function
auto reorder_nodes(int N, Node *nodes, Box *boxes, bool grouped = false) -> void {
    requires_type(ReorderType::NotReordered);

    struct DepthNode {
        int parent;
        int depth; // note that this depth is from root, not leaf.
        int start_children;
        int count_children;
    };

    const auto cpu_nodes = [nodes, N] {
        auto vec = std::vector<Node>(N);
        cudaMemcpy(vec.data(), nodes, N * sizeof(Node), cudaMemcpyDeviceToHost);
        return vec;
    }();

    const auto dep_nodes = [&cpu_nodes, N] {
        auto vec = std::vector<DepthNode>(N);
        for (const auto i : irange(N)) {
            vec[i].parent         = cpu_nodes[i].parent;
            vec[i].start_children = cpu_nodes[i].start_children;
            vec[i].count_children = cpu_nodes[i].count_children;
            if (vec[i].parent == -1) {
                vec[i].depth = 0;
            } else {
                if (vec[i].parent >= i)
                    throw std::runtime_error("Error: parent is greater than itself");
                vec[i].depth = vec[vec[i].parent].depth + 1;
            }
        }
        return vec;
    }();

    auto new_index = [N, nodes = dep_nodes.data(), grouped] {
        auto vec   = std::vector<int>(N, -1);
        auto queue = std::queue<int>();
        auto index = 0;
        queue.push(0);

        // allocate one index for each node
        auto _allocate = [&vec, &index](int idx) {
            if (vec[idx] != -1)
                throw std::runtime_error("Error: double visit");
            vec[idx] = index++;
        };
        // add sons of certain node to the queue
        auto _add_sons = [&queue](const DepthNode &node) {
            for (const auto i : irange(node.count_children))
                queue.push(node.start_children + i);
        };

        while (!queue.empty()) {
            const auto idx = queue.front();
            queue.pop();
            // if grouped, we need to stop at certain depth
            if (grouped && nodes[idx].depth == kGroupLevel)
                break;
            _allocate(idx);
            _add_sons(nodes[idx]);
        }

        if (grouped) {
            // now, all depth < kGroupLevel are reordered
            // for those depth >= kGroupLevel, we need to reorder them

            // Step 1:
            //  find all the head nodes, traverse them first
            //  these nodes depth = kGroupLevel
            const auto group_heads = [&]() {
                auto vec = std::vector<int>{};
                while (!queue.empty()) {
                    vec.push_back(queue.front());
                    queue.pop();
                }
                return vec;
            }();

            package.group_start_index = index;
            for (const auto &idx : group_heads) {
                _allocate(idx);
                if (nodes[idx].depth != kGroupLevel)
                    throw std::runtime_error("Sir, your bfs is wrong");
            }
            package.group_finish_index = index;

            auto group_child_cnt = std::vector<int>{};
            group_child_cnt.reserve(group_heads.size());
            for (const auto &idx : group_heads) {
                auto child_cnt = std::size_t{};
                _add_sons(nodes[idx]);
                while (!queue.empty()) {
                    const auto idx = queue.front();
                    queue.pop();
                    _allocate(idx);
                    _add_sons(nodes[idx]);
                    child_cnt++;
                }
                group_child_cnt.push_back(child_cnt);
            }

            // assure that certain groups of children will not exceed the limit
            // access the maximum SM units on the GPU
            const auto kMaxSum = [] {
                auto device_prop = cudaDeviceProp{};
                cudaGetDeviceProperties(&device_prop, 0);
                return device_prop.multiProcessorCount;
            }();

            auto current_sum = 0;
            auto prefix_sum  = 0;

            // first group is always at {0, 0}
            auto groups = std::vector<GroupInfo>{};
            groups.push_back({.group_offset = 0, .child_offset = 0});
            for (const auto i : irange(group_heads.size())) {
                prefix_sum += group_child_cnt[i];
                current_sum += group_child_cnt[i];
                // if current_sum > kMaxSum, then we need to end this group
                if (current_sum > kMaxSum) {
                    groups.push_back({.group_offset = i + 1, .child_offset = prefix_sum});
                    current_sum = 0;
                }
            }

            if (current_sum != 0) { // mark the end of the last group
                const auto last_offset = int(group_heads.size());
                groups.push_back({.group_offset = last_offset, .child_offset = prefix_sum});
            }

            // copy this group to gpu
            package.groups.resize(groups.size());
            package.slots.resize(groups.size() - 1);
            cudaMemcpy(
                package.groups.data().get(), groups.data(), groups.size() * sizeof(GroupInfo),
                cudaMemcpyHostToDevice
            );
        }

        if (index != N)
            throw std::runtime_error(
                "Error: " + std::to_string(index) + " != " + std::to_string(N)
            );
        for (const auto &i : vec)
            if (i == -1)
                throw std::runtime_error("Error: not all nodes are visited");
        return vec;
    }();

    // reorder cpu_node and cpu_box, then copy back to device
    const auto cpu_boxes = [boxes, N] {
        auto vec = std::vector<Box>(N);
        cudaMemcpy(vec.data(), boxes, N * sizeof(Box), cudaMemcpyDeviceToHost);
        return vec;
    }();

    // set the order type
    order_type = grouped ? ReorderType::GroupOrder : ReorderType::BFSOrder;
    return do_reorder(N, nodes, cpu_nodes.data(), boxes, cpu_boxes.data(), new_index);
}

auto reorder_back_nodes(int N, int *nodes_for_render_indices) -> void {
    requires_not_type(ReorderType::NotReordered);

    auto vec = std::vector<int>(N);
    cudaMemcpy(vec.data(), nodes_for_render_indices, N * sizeof(int), cudaMemcpyDeviceToHost);

    auto bak = vec;
    for (const auto i : irange(N))
        vec[i] = reverse_map[bak[i]];

    cudaMemcpy(nodes_for_render_indices, vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
}

__device__ auto inboxCUDA(const Box &box, const Point &viewpoint) -> bool {
    bool inside = true;
    for (int i = 0; i < 3; i++) {
        inside &= viewpoint.xyz[i] >= box.minn.xyz[i] && viewpoint.xyz[i] <= box.maxx.xyz[i];
    }
    return inside;
}

__device__ auto pointboxdistCUDA(const Box &box, const Point &viewpoint) -> float {
    Point closest = {
        max(box.minn.xyz[0], min(box.maxx.xyz[0], viewpoint.xyz[0])),
        max(box.minn.xyz[1], min(box.maxx.xyz[1], viewpoint.xyz[1])),
        max(box.minn.xyz[2], min(box.maxx.xyz[2], viewpoint.xyz[2]))
    };

    Point diff = {
        viewpoint.xyz[0] - closest.xyz[0], viewpoint.xyz[1] - closest.xyz[1],
        viewpoint.xyz[2] - closest.xyz[2]
    };

    return sqrt(diff.xyz[0] * diff.xyz[0] + diff.xyz[1] * diff.xyz[1] + diff.xyz[2] * diff.xyz[2]);
}

static constexpr auto FMAX = std::numeric_limits<float>::max();

[[gnu::pure]]
__device__ auto computeSizeGPU(const Box &box, const Point &viewpoint) -> float {
    if (inboxCUDA(box, viewpoint))
        return FMAX;
    float min_dist = pointboxdistCUDA(box, viewpoint);
    return box.minn.xyz[3] / min_dist;
}
__forceinline__ __device__ float3 tPoint4x3(const float3& p, const float* matrix)
{
	float3 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
	};
	return transformed;
}
__forceinline__ __device__ float4 tPoint4x4(const float3& p, const float* matrix)
{
	float4 transformed = {
		matrix[0] * p.x + matrix[4] * p.y + matrix[8] * p.z + matrix[12],
		matrix[1] * p.x + matrix[5] * p.y + matrix[9] * p.z + matrix[13],
		matrix[2] * p.x + matrix[6] * p.y + matrix[10] * p.z + matrix[14],
		matrix[3] * p.x + matrix[7] * p.y + matrix[11] * p.z + matrix[15]
	};
	return transformed;
}
__forceinline__ __device__ bool in_frustum(
	const float3 p_orig,
	const float* __restrict__ viewmatrix,
	const float* __restrict__ projmatrix)
{
	float3 p_view = tPoint4x3(p_orig, viewmatrix);
	float4 p_hom = tPoint4x4(p_view, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	const float thre = 1.3;
	if (p_view.z <= 0.2f || p_proj.x < -thre || p_proj.x > thre || p_proj.y < -thre || p_proj.y > thre)
		return false;
	return true;
} 
__device__ __forceinline__ auto isValid(
    const Box *boxes, const SimpleNode &node, const float target_size, Point viewpoint, Point,
    int node_id
) -> bool {
    if (node.is_leaf())
        return target_size <= computeSizeGPU(boxes[node.parent], viewpoint);
    if (node.is_root() || !node.count_merged)
        return false;
    return target_size <= computeSizeGPU(boxes[node.parent], viewpoint) &&
           target_size > computeSizeGPU(boxes[node_id], viewpoint);
}

__global__ auto markNodesForSize(
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, int N, 
    const Point *viewpoint, const float* view_transform, const float* projection_matrix, Point zdir,
    float target_size, bool *render_counts
) -> void {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;
    auto &node = nodes[idx]; 
    if (node.is_root())
        return void(render_counts[idx] = false);
    const float parent_size = computeSizeGPU(boxes[node.parent], *viewpoint);
    const float child_size = computeSizeGPU(boxes[idx], *viewpoint);
    if (!in_frustum(means3D[node.start], view_transform, projection_matrix))
        return void(render_counts[idx] = false);
    render_counts[idx] = target_size <= parent_size && target_size > child_size;
} 

__global__ auto markNodesByDepth(
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, int N, 
    const Point *viewpoint, const float* view_transform, const float* projection_matrix, 
    Point, 
    float target_size, bool *render_counts, bool *is_smaller, int offset
) -> void {
    const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (base_idx >= N)
        return;

    const int idx = base_idx + offset;
    auto &node    = nodes[idx];

    if (node.is_root())
        return void(is_smaller[idx] = false);
    if (is_smaller[node.parent] == true)
        return void(is_smaller[idx] = true);
    if(!in_frustum(means3D[node.start], view_transform, projection_matrix))
        return void(is_smaller[idx] = false);

    render_counts[idx] = is_smaller[idx] = target_size > computeSizeGPU(boxes[idx], *viewpoint);
} 

__global__ auto putRenderIndices(
    const SimpleNode *nodes, int N, const bool *node_counts, const int *node_offsets,
    int *render_indices, int *parent_indices, int *nodes_for_render_indices
) -> void {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    auto count = node_counts[idx];

    // count is 0 or 1.
    if (count == 0)
        return;

    auto &node                       = nodes[idx];
    auto offset                      = idx == 0 ? 0 : node_offsets[idx - 1];
    render_indices[offset]           = node.start;
    auto parentgaussian              = node.is_root() ? -1 : nodes[node.parent].start;
    if (parent_indices != nullptr)
        parent_indices[offset]      = parentgaussian;
    if (nodes_for_render_indices!=nullptr)
        nodes_for_render_indices[offset] = idx;
}

__global__ auto markNodesByDepthFused(
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, 
    int N, const Point *viewpoint, const float* view_transform, 
    const float* projection_matrix, Point,
    float target_size, bool *is_smaller, int offset, Holder *holder,        // inputs & auxiliaries
    int *render_indices, int *parent_indices, int *nodes_for_render_indices // output pointers
) -> void {
    const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (base_idx >= N)
        return;

    const int idx = base_idx + offset;
    auto &node    = nodes[idx];

    if (node.is_root()) {
        holder[0].reset(); // reset the counter
        holder[1].reset(); // reset the counter
        return void(is_smaller[idx] = false);
    }

    if (is_smaller[node.parent] == true)
        return void(is_smaller[idx] = true);

    int start = node.start;

    if(!in_frustum(means3D[start], view_transform, projection_matrix))
        return void(is_smaller[idx] = false);
    const auto smaller = target_size > computeSizeGPU(boxes[idx], *viewpoint);
    is_smaller[idx] = smaller;
    if (!smaller) // render only when smaller is false
        return;

    // now need to render, get a new_id and store the data
    auto new_id                      = holder[0].increment();
    render_indices[new_id]           = node.start;
    if (parent_indices != nullptr)
        parent_indices[new_id]           = nodes[node.parent].start;
    if (nodes_for_render_indices!=nullptr)
        nodes_for_render_indices[new_id] = idx;
}


__global__ auto groupCalcReuseKernel(
    // inputs & auxiliaries
    const Box *boxes, int group_cnt, const Point *viewpoint, const bool *is_smaller, Holder *holder,
    const float *last, const GroupInfo *groups, int group_start_index,
    // output pointers
    float *next, int *slots
) -> void {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= group_cnt)
        return;

    const auto start  = groups[idx + 0].group_offset;
    const auto finish = groups[idx + 1].group_offset;

    auto no_render = true;
    for (int i = start; i < finish; ++i) {
        const auto idx = group_start_index + i; // the head node's index
        const auto siz = computeSizeGPU(boxes[idx], *viewpoint);
        next[i]        = siz;
        // if may need to render, then check the ratio.
        // if the ratio is outof the range, then render it.
        if (no_render && !is_smaller[idx]) {
            if constexpr (kGroupRatioMin != 1) {
                auto ratio = last[i] / siz;
                no_render  = kGroupRatioMin < ratio && ratio < kGroupRatioMax;
            } else {
                no_render = false;
            }
        }
    }

    // need to render this group if no_render is false
    if (no_render)
        return;

    // now need to render, get a new_id and store the data
    slots[holder[1].increment()] = idx;
}

auto fusion_kernel_impl(
    // input parameters
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, int N, const Point *viewpoint,
    const float* view_transform, const float* projection_matrix, float target_size,
    // output parameters
    int *render_indices, int *parent_indices, int *nodes_for_render_indices,
    // some other helpers
    const std::vector<int> &depth_count
) -> int {
    requires_type(ReorderType::BFSOrder);

    auto offset     = int{};
    auto holder     = package.holder.data().get();
    auto is_smaller = package.is_smaller.data().get();

    for (const auto &count : depth_count) {
        const auto num_blocks = (count + 255) / 256;
        markNodesByDepthFused<<<num_blocks, 256>>>(
            nodes, boxes, means3D, count, viewpoint, view_transform, projection_matrix, 
            {}, target_size, is_smaller, offset, holder,
            render_indices, parent_indices, nodes_for_render_indices
        );
        offset += count;
    }

    if (offset != N)
        throw std::runtime_error("Error: overall count != N");

    return package.holder.data().get()->device_value();
}

auto default_kernel_impl(
    // input parameters
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, int N, const Point *viewpoint,
    float* view_transform, float* projection_matrix, float target_size,
    // output parameters
    int *render_indices, int *parent_indices, int *nodes_for_render_indices
) -> int {
    auto render_counts  = package.render_counts.data().get();
    auto render_offsets = package.render_offsets.data().get();
    auto &temp_storage  = package.temp_storage;
    auto temp_bytes     = std::size_t{};

    requires_type(ReorderType::BFSOrder);
    markNodesForSize<<<(N + 255) / 256, 256>>>(
        nodes, boxes, means3D, N, viewpoint, view_transform, projection_matrix, 
        {}, target_size, render_counts
    );

    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, render_counts, render_offsets, N);

    temp_storage.resize(temp_bytes);
    auto gpu_ptr = temp_storage.data().get();

    cub::DeviceScan::InclusiveSum(gpu_ptr, temp_bytes, render_counts, render_offsets, N);

    putRenderIndices<<<(N + 255) / 256, 256>>>(
        nodes, N, render_counts, render_offsets, render_indices, parent_indices,
        nodes_for_render_indices
    );

    int count = 0;
    cudaMemcpy(&count, render_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

auto layerwise_kernel_impl(
    // input parameters
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, int N, const Point *viewpoint,
    const float* view_transform, const float* projection_matrix, float target_size,
    // output parameters
    int *render_indices, int *parent_indices, int *nodes_for_render_indices,
    // some other helpers
    const std::vector<int> &depth_count
) -> int {
    auto offset         = int{};
    auto is_smaller     = package.is_smaller.data().get();
    auto render_counts  = package.render_counts.data().get();
    auto render_offsets = package.render_offsets.data().get();
    auto &temp_storage  = package.temp_storage;
    auto temp_bytes     = std::size_t{};

    for (const auto &count : depth_count) {
        const auto num_blocks = (count + 255) / 256;
        markNodesByDepth<<<num_blocks, 256>>>(
            nodes, boxes, means3D, count, viewpoint, view_transform, projection_matrix, 
            {}, target_size, //
            render_counts, is_smaller, offset                //
        );
        offset += count;
    }

    if (offset != N)
        throw std::runtime_error("Error: overall count != N");

    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, render_counts, render_offsets, N);

    temp_storage.resize(temp_bytes);
    auto gpu_ptr = temp_storage.data().get();

    cub::DeviceScan::InclusiveSum(gpu_ptr, temp_bytes, render_counts, render_offsets, N);

    putRenderIndices<<<(N + 255) / 256, 256>>>(
        nodes, N, render_counts, render_offsets, render_indices, parent_indices,
        nodes_for_render_indices
    );

    int count = 0;
    cudaMemcpy(&count, render_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}

auto dispatcher_kernel(
    // input parameters
    const SimpleNode *nodes, const Box *boxes, const float3* means3D, int N, 
    const Point *viewpoint_p, float* view_transform, float* projection_matrix, Point,
    float target_size,
    // output parameters
    int *, int *render_indices, int *parent_indices, int *nodes_for_render_indices,
    // whether to use custom kernel
    const std::vector<int> &depth_count, int mode 
) -> int {
    package.resize(N);

    enum {
        Fused     = 0,
        Default   = 1,
        LayerWise = 2,
        Reuse     = 3,
    };

    switch (mode) {
        case Default:
            return default_kernel_impl(
                nodes, boxes, means3D, N, viewpoint_p, 
                view_transform, projection_matrix, target_size, render_indices, parent_indices,
                nodes_for_render_indices
            );
        case LayerWise:
            return layerwise_kernel_impl(
                nodes, boxes, means3D, N, viewpoint_p, 
                view_transform, projection_matrix, target_size, render_indices, parent_indices,
                nodes_for_render_indices, depth_count
            );
        case Fused:
            return fusion_kernel_impl(
                nodes, boxes, means3D, N, viewpoint_p, 
                view_transform, projection_matrix, target_size, render_indices, parent_indices,
                nodes_for_render_indices, depth_count
            );
        default: // unknown mode
            throw std::runtime_error("Error: mode not supported");
    }
}

__global__ auto computeTsIndexed(
    SimpleNode *nodes, Box *boxes, int N, int *indices, Point viewpoint, Point, float target_size,
    float *ts, int *kids
) -> void {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N)
        return;

    int node_id = indices[idx];
    auto &node  = nodes[node_id];

    float t;
    if (node.is_root())
        t = 1.0f;
    else {
        float parentsize = computeSizeGPU(boxes[node.parent], viewpoint);
        if (parentsize > 2.0f * target_size) {
            t = 1.0f;
        } else {
            float size  = computeSizeGPU(boxes[node_id], viewpoint);
            float start = max(0.5f * parentsize, size);
            float diff  = parentsize - start;

            if (diff <= 0)
                t = 1.0f;
            else {
                float tdiff = max(0.0f, target_size - start);
                t           = max(1.0f - (tdiff / diff), 0.0f);
            }
        }
    }

    ts[idx]   = t;
    kids[idx] = (node.is_root()) ? 1 : nodes[node.parent].count_children;
}

} // namespace kernel
} // namespace

namespace dark {

auto reorder_nodes(int N, int *_node, float *_box, bool grouped = false) -> void {
    auto *node = reinterpret_cast<Node *>(_node);
    auto *box  = reinterpret_cast<Box *>(_box);
    kernel::reorder_nodes(N, node, box, grouped);
    kernel::to_simple(node, N);
    kernel::package.resize(N);
}

auto reorder_result(int N, int *arr) -> void {
    kernel::reorder_back_nodes(N, arr);
}

} // namespace dark

#ifndef _LOCAL_DEBUG

#include "runtime_switching.h"

float Switching::getTsIndexed(
    int N, int *indices, float target_size, int *nodes, float *boxes, float vx, float vy, float vz,
    float x, float y, float z, float *ts, int *kids, void *stream
){
    requires_not_type(kernel::ReorderType::NotReordered);
    float elapse = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    Point zdir     = {x, y, z};
    Point cam      = {vx, vy, vz};
    int num_blocks = (N + 255) / 256;
    kernel::computeTsIndexed<<<num_blocks, 256, 0, (cudaStream_t)stream>>>(
        (SimpleNode *)nodes, (Box *)boxes, N, indices, cam, zdir, target_size, ts, kids
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse, start, end);
    return elapse;
}

std::tuple<int, float> Switching::expandToSize(
    int N, float target_size, 
    int *nodes, float *boxes, float* means3D, 
    float *viewpoint,
    float* view_transform,
    float* projection_matrix, 
    float x, float y, float z,
    int *render_indices, int *node_markers, int *parent_indices, int *nodes_for_render_indices,
    int mode // default mode = 0
) {
    kernel::requires_not_type(kernel::ReorderType::NotReordered);
    float elapse = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    
    int ret = kernel::dispatcher_kernel(
        (SimpleNode *)nodes, (Box *)boxes, (float3*)means3D, N, (Point *)viewpoint,
        view_transform, projection_matrix, {x, y, z}, target_size,
        node_markers, render_indices, parent_indices, nodes_for_render_indices,
        kernel::depth_count_global, mode 
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse, start, end);
    return std::make_tuple(ret, elapse);
}

#endif
