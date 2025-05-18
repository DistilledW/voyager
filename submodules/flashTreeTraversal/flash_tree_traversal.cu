/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * modified 
 */
#include "flash_tree_traversal.h"
#include "types.h"
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>
#include <map>
#include <new>
#include <queue>
#include <thrust/device_vector.h>
#include <vector>
#include <thrust/count.h>
#include <float.h> 
#include <cuda_fp16.h>

// #include <device_launch_parameters.h>
// #include <cub/cub.cuh>
// #include <cub/device/device_radix_sort.cuh>
// #include <thrust/sequence.h>
// #include <thrust/device_vector.h>
// #include <nvtx3/nvToolsExt.h>
// #include <thrust/host_vector.h>
// #include <cooperative_groups.h>
// #include <cooperative_groups/reduce.h>
// #include <thrust/fill.h>
// #include <thrust/reduce.h>
namespace {
namespace kernel {
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
} order_type = ReorderType::NotReordered;

auto requires_type(ReorderType type) -> void {
    if (order_type != type) [[unlikely]]
        throw std::runtime_error("Error: wrong order type");
}

auto requires_not_type(ReorderType type) -> void {
    if (order_type == type) [[unlikely]]
        throw std::runtime_error("Error: wrong order type");
}

struct  Holder { 
    private:
        int atomic_counter;        // need atomic data on device
        char _[256 - sizeof(int)]; // padding for alignment
    public:
        __device__ auto reset(int n=0) -> void {
            static_assert(sizeof(_[0]) == 1, "to avoid warning");
            atomic_counter = n;
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
}; 
__global__ void setHolder(Holder *holder, int n=1, int number = 0){
    for (auto i = 0; i < n; ++i){
        holder[i].reset(number); 
    }
}
struct HolderPair{ 
    private:
        unsigned long long atomic_counter;  // need atomic data on device 
        char _[256 - sizeof(unsigned long long)];  // padding for alignment
        __device__ static inline unsigned long long pack(std::pair<int, int> p) {
            return (static_cast<unsigned long long>(p.second) << 32) | static_cast<unsigned long long>(p.first);
        }
    public:
        __device__ auto reset(int n = 0) -> void { 
            static_assert(sizeof(_[0]) == 1, "to avoid warning");
            atomic_counter = n;
        }
        __device__ auto increment(std::pair<int, int> p) -> std::pair<int, int> {
            unsigned long long old = atomicAdd(static_cast<unsigned long long*>(&atomic_counter), this->pack(p));
            return {static_cast<int>(old&0xFFFFFFFF), static_cast<int>(old>>32)};
        }
        __host__ auto device_value() const& -> int {
            unsigned long long value;
            cudaMemcpy(&value, &atomic_counter, sizeof(value), cudaMemcpyDeviceToHost);
            return static_cast<int>((value>>32)&0xFFFFFFFF);
        }
        // to avoid copying on host, since the value is in device 
        __host__ auto device_value() && = delete;
};
__global__ void setHolderPair(HolderPair *holder, int n=1, int number = 0){
    for (auto i = 0; i < n; ++i){
        holder[i].reset(number); 
    }
}

struct Package {
    thrust::device_vector<char> temp_storage;
    thrust::device_vector<bool> render_counts;
    thrust::device_vector<int> render_offsets;
    thrust::device_vector<bool> is_smaller;
    thrust::device_vector<Holder> holder;
    auto resize(int N) -> void {
        const auto kHack = 78079; // this is a hacked size.
        temp_storage.resize(kHack);
        render_counts.resize(N);
        render_offsets.resize(N);
        is_smaller.resize(N);
        holder.resize(2);
    }
};
auto &package = *new Package{}; // this is intended leak to skip destructor

static constexpr auto FMAX = std::numeric_limits<float>::max();
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

// ===================================================================
__forceinline__ __device__ auto inboxCUDA(const Box &box, const Point &viewpoint) -> bool {
    bool inside = true;
    for (int i = 0; i < 3; i++) {
        inside &= viewpoint.xyz[i] >= box.minn.xyz[i] && viewpoint.xyz[i] <= box.maxx.xyz[i];
    }
    return inside;
}
__forceinline__ __device__ auto pointboxdistCUDA(const Box &box, const Point &viewpoint) -> float {
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
__forceinline__ __device__ auto computeSizeGPU(const Box &box, const Point &viewpoint) -> float {
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
__forceinline__ __device__ void writeIntToBytes(int value, uint8_t* ptr) {
    union {
        int i;
        uint8_t bytes[4];
    } converter;
    converter.i = value;
    memcpy(ptr, converter.bytes, 4);
}
__device__ __forceinline__ int readIntFromBytes(const uint8_t* ptr) {
    union {
        int i;
        uint8_t bytes[4];
    } converter;
    memcpy(converter.bytes, ptr, 4);
    return converter.i;
} 
__forceinline__ __device__ void writeFloatToHalfBytes(float value, uint8_t* &ptr){
    union {
        uint32_t u;
        float f;
    } src;
    src.f = value;
    uint32_t f = src.u;
    uint32_t sign = (f >> 31) & 0x1;
    uint32_t exp = (f >> 23) & 0xFF;
    uint32_t frac = f & 0x7FFFFF;
    uint16_t result;
    if (exp == 255) {
        // Inf or NaN
        result = (sign << 15) | (0x1F << 10) | (frac ? (frac >> 13) : 0);
    } else if (exp > 142) {
        // Too large, becomes Inf
        result = (sign << 15) | (0x1F << 10);
    } else if (exp < 113) {
        // Subnormal range in float16
        int shift = 113 - exp;
        frac |= 0x800000;  // Add implicit leading 1
        uint16_t sub = frac >> (shift + 13);
        if (((frac >> (shift + 12)) & 1) && ((sub & 1) || (frac & ((1 << (shift + 12)) - 1)))) {
            sub += 1; // Round-to-nearest-even
        }
        result = (sign << 15) | sub;
    } else {
        // Normalized
        int new_exp = exp - 127 + 15;
        uint16_t new_frac = frac >> 13;
        if ((frac >> 12) & 1 && ((new_frac & 1) || (frac & 0xFFF))) {
            new_frac += 1;
            if (new_frac == 0x400) { // carry overflow
                new_frac = 0;
                new_exp += 1;
            }
        }
        result = (sign << 15) | (new_exp << 10) | new_frac;
    }

    ptr[0] = static_cast<uint8_t>(result & 0xFF);
    ptr[1] = static_cast<uint8_t>((result >> 8) & 0xFF);
    ptr += 2;
}
__forceinline__ __device__ float readFloatFromHalfBytes(uint8_t* &ptr) {
    uint16_t raw = static_cast<uint16_t>(ptr[0]) | static_cast<uint16_t>(ptr[1] << 8);
    ptr += 2;

    uint16_t sign = (raw >> 15) & 0x1;
    uint16_t exp  = (raw >> 10) & 0x1F;
    uint16_t frac = raw & 0x3FF;

    uint32_t result;

    if (exp == 0) {
        if (frac == 0) {
            result = (sign << 31);
        } else {
            exp = 1;
            while ((frac & 0x400) == 0) { 
                frac <<= 1;
                exp--;
            }
            frac &= 0x3FF; 
            exp = 127 - 15 - (10 - exp);  
            uint32_t f_frac = frac << 13;
            result = (sign << 31) | (exp << 23) | f_frac;
        }
    } else if (exp == 0x1F) {
        // Inf or NaN
        if (frac == 0) {
            result = (sign << 31) | (0xFF << 23);  // ±Inf
        } else {
            result = (sign << 31) | (0xFF << 23) | (frac << 13);  // NaN
        }
    } else {
        // Normalized float16 number
        uint32_t f_exp = exp - 15 + 127;
        uint32_t f_frac = frac << 13;
        result = (sign << 31) | (f_exp << 23) | f_frac;
    }
    union {
        uint32_t u;
        float f;
    } out;
    out.u = result;
    return out.f;
}

// ==================================================================================
auto do_reorder(
    int N, Node *gpu_nodes, const Node *cpu_nodes, Box *gpu_boxes, const Box *cpu_boxes,
    const std::vector<int> &new_index, int* _depth_count
) -> int {
    // requires_not_type(ReorderType::NotReordered);

    auto new_nodes = std::vector<Node>(N);
    auto new_boxes = std::vector<Box>(N);

    // reorder the nodes
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
    int tree_height = 0;
    for (const auto &[_, count] : depths_map){
        depth_count_global.push_back(count);
        _depth_count[tree_height++] = count;
    }

    cudaMemcpy(gpu_nodes, new_nodes.data(), N * sizeof(Node), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_boxes, new_boxes.data(), N * sizeof(Box), cudaMemcpyHostToDevice);
    return tree_height;
}
[[gnu::noinline]] // Don't inline this big function 
auto reorder_nodes(int N, Node *nodes, Box *boxes, int* _depth_count) -> int { 
    // requires_type(ReorderType::NotReordered);
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

    auto new_index = [N, nodes = dep_nodes.data()] {
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
            _allocate(idx);
            _add_sons(nodes[idx]);
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

    return do_reorder(N, nodes, cpu_nodes.data(), boxes, cpu_boxes.data(), new_index, _depth_count);
} 
auto to_simple(Node *node, int N) -> void {
    thrust::device_vector<Node> tmp_nodes(N);
    cudaMemcpy(tmp_nodes.data().get(), node, N * sizeof(Node), cudaMemcpyDeviceToDevice);
    auto *ptr = reinterpret_cast<SimpleNode *>(node);
    transform_node<<<(N + 255) / 256, 256>>>(ptr, N, tmp_nodes.data().get());
}
auto reorder_back_nodes(int N, int *nodes_for_render_indices) -> void {
    // requires_not_type(ReorderType::NotReordered);

    auto vec = std::vector<int>(N);
    cudaMemcpy(vec.data(), nodes_for_render_indices, N * sizeof(int), cudaMemcpyDeviceToHost);

    auto bak = vec;
    for (const auto i : irange(N))
        vec[i] = reverse_map[bak[i]];

    cudaMemcpy(nodes_for_render_indices, vec.data(), N * sizeof(int), cudaMemcpyHostToDevice);
}

// ===============================================================================
__global__ void putParents(Node* node, int* parents, int N){
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) 
        return;
    parents[idx] = node[idx].parent;
}
__global__ auto putRenderIndices(
    const SimpleNode*   __restrict__ nodes, 
    const Box*          __restrict__ boxes, 
    const int N, 
    int*                __restrict__ render_indices, 
    int*                __restrict__ node_indices, 
    const bool*         __restrict__ node_counts, 
    const int*          __restrict__ node_offsets 
) -> void {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    auto count = node_counts[idx];
    if (idx >= N || !count)
        return;

    auto &node                  = nodes[idx];
    auto offset                 = idx == 0 ? 0 : node_offsets[idx - 1];
    render_indices[offset]      = node.start;
    node_indices[offset]        = idx; 
}
__global__ auto markNodesForSize( 
    const SimpleNode*   __restrict__ nodes, 
    const Box*          __restrict__ boxes, 
    const float3*       __restrict__ means3D, 
    const int N, 
    const Point *viewpoint, 
    float target_size, 
    const int window_size, 
    float*              __restrict__ view_transform, 
    float*              __restrict__ projection_matrix, 
    const bool frustum_culling, 
    // output parameters 
    int*                __restrict__ least_recently, 
    bool*               __restrict__ render_counts 
) -> void {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) 
        return;

    auto &node = nodes[idx]; 
    least_recently[idx]   = min(least_recently[idx] + 1, 100);
    
    if (node.is_root())
        return void(render_counts[idx] = false);
    const float parent_size = computeSizeGPU(boxes[node.parent], *viewpoint);
    const float child_size = computeSizeGPU(boxes[idx], *viewpoint);

    int start = node.start;
    if (frustum_culling && !in_frustum(means3D[start], view_transform, projection_matrix))
        return void(render_counts[idx] = false);

    render_counts[idx] = target_size <= parent_size && target_size > child_size; 
    if (render_counts[idx]){ 
        if (least_recently[idx] < window_size) 
            render_counts[idx] = false;
        least_recently[idx] = 0;
    } 
} 

__global__ auto markNodesByDepth(
    const SimpleNode*   __restrict__ nodes, 
    const Box*          __restrict__ boxes, 
    const float3*       __restrict__ means3D, 
    const int N, 
    const Point *viewpoint, 
    const float target_size, 
    const int window_size, 
    const float*        __restrict__ view_transform, 
    const float*        __restrict__ projection_matrix, 
    const bool frustum_culling, 
    int*                __restrict__ least_recently,
    bool *              __restrict__ render_counts, 
    bool*               __restrict__ is_smaller, 
    const int offset 
) -> void {
    const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (base_idx >= N) 
        return;

    const int idx       = base_idx + offset;
    auto &node          = nodes[idx];
    least_recently[idx] = min(least_recently[idx] + 1, 100);

    if (node.is_root())
        return void(is_smaller[idx] = false);

    if (is_smaller[node.parent] == true)
        return void(is_smaller[idx] = true);

    int start = node.start;
    if (frustum_culling && !in_frustum(means3D[start], view_transform, projection_matrix))
        return void(is_smaller[idx] = false);
    
    const auto smaller = target_size > computeSizeGPU(boxes[idx], *viewpoint);
    is_smaller[idx] = smaller;
    if (!smaller)
        return ;
    
    if (static_cast<int>(least_recently[idx]) < window_size)
        render_counts[idx] = true;
    least_recently[idx] = static_cast<int>(0);
}

__global__ auto markNodesByDepthFused( 
    const SimpleNode*   __restrict__ nodes, 
    const Box*          __restrict__ boxes, 
    const float3*       __restrict__ means3D, 
    const int N, 
    const Point* viewpoint, 
    float target_size, 
    const int window_size, 
    float*              __restrict__ view_transform, 
    float*              __restrict__ projection_matrix, 
    const bool frustum_culling, 
    bool*               __restrict__ is_smaller, 
    int offset, 
    Holder* holder, 
    // output parameters 
    int*                __restrict__ least_recently, 
    int*                __restrict__ render_indices, 
    int*                __restrict__ node_indices 
) -> void { 
    const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (base_idx >= N) return;
    const int idx = base_idx + offset;
    auto &node    = nodes[idx];
    least_recently[idx]   = min(least_recently[idx] + 1, 100);
    
    if (node.is_root())
        return void(is_smaller[idx] = false);

    if (is_smaller[node.parent]) 
        return void(is_smaller[idx] = true);
    
    int start = node.start;
    if (!in_frustum(means3D[start], view_transform, projection_matrix))
        return void(is_smaller[idx] = false);

    // is_smaller parent is false, which means that parent size >= target_size
    const auto smaller = target_size > computeSizeGPU(boxes[idx], *viewpoint);
    is_smaller[idx] = smaller;
    if (!smaller) // render only when smaller is false 
        return ;
    if (least_recently[idx] > window_size) {
        // now need to render, get a new_id and store the data 
        auto putIdx                 = holder[0].increment(); 
        render_indices[putIdx]      = start; 
        node_indices[putIdx]        = idx;
    } 
    least_recently[idx] = 0;
}


__global__ void SGTInitGPU(
    const int N, 
    int*    indices, 
    float*  features_cur,
    float*  shs_cur, 
    int*    starts, 
    float*  means3D, 
    float*  opacities, 
    float*  rotations, 
    float*  scales, 
    float*  shs, 
    float*  boxes, 
    int*    back_pointer 
){ 
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N) return;
    auto start      = idx; 
    int node_idx    = indices[idx]; 
    starts[node_idx] = start; 
    back_pointer[start]     = node_idx;

    float* ptr = features_cur + 18 * idx;
    float* menas3D_ptr = means3D + start * 3;
    for(auto j = 0; j < 3; ++j) menas3D_ptr[j] = ptr[j];
    opacities[start] = ptr[3];
    ptr+=4;
    float* rot_ptr = rotations + start * 4;
    for(auto j = 0; j < 4; ++j) rot_ptr[j] = ptr[j];
    ptr += 4;
    float* sca_ptr = scales + start * 3;
    for(auto j = 0; j < 3; ++j) sca_ptr[j] = ptr[j];
    ptr += 3;
    float* box_ptr = boxes + start * 8;
    for(auto j = 0; j < 7; ++j) box_ptr[j] = ptr[j];
    ptr += 7;
    float* shs_ptr = shs + 48 * start;
    for (auto j = 0; j < 48; ++j) shs_ptr[j] = shs_cur[j + 48 * idx];
} 

__global__ auto SGTExpandGPU( 
    const int N, 
    const int idx_offset, 
    const int*  __restrict__ starts, const int* parents, 
    const float3*   __restrict__ means3D, 
    const Box*      __restrict__ boxes, 
    const float target_size, 
    const Point* viewpoint, 
    bool frustum_culling, 
    const float*    __restrict__ view_transform, 
    const float*    __restrict__ projection_matrix, 
    bool*           __restrict__ is_smaller, 
    Holder* holder,
    int*            __restrict__ least_recently, 
    int*            __restrict__ render_indices 
) -> void { 
    const int base_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (base_idx >= N)
        return;

    const int  idx      = base_idx + idx_offset;
    const auto p_idx    = parents[idx];
    const auto start    = starts[idx];
    if (start != -1) 
        least_recently[start]   = min(least_recently[start] + 1, 100);
    if (p_idx == -1)
        return void(is_smaller[idx] = false);
    else if (is_smaller[p_idx])
        return void(is_smaller[idx] = true);
    else if (start == -1)
        return void(is_smaller[idx] = false);

    float3 p_orig = means3D[start]; 
    if (frustum_culling && !in_frustum(p_orig, view_transform, projection_matrix))
        return void(is_smaller[idx] = false);
    
    // is_smaller parent is false, which means that parent size >= target_size
    const auto smaller = target_size > computeSizeGPU(boxes[start], *viewpoint);
    is_smaller[idx] = smaller;
    if (!smaller) // render only when smaller is false
        return;
    // now need to render, get a new_id and store the data 
    auto offset             = holder[0].increment();
    render_indices[offset]  = start;
    least_recently[start]   = 0;
}
__global__ void putUpdateIndices(const int N, const int* least_recently, const int window_size, Holder* holder, int* update_indices){
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= N || least_recently[idx] <= window_size) return ;
    int offset = holder[0].increment(); 
    update_indices[offset] = idx; 
} 
__global__ void updateCUDA(
    const int   to_update, 
    const int   to_expand, 
    int*        update_indices, 
    int*        __restrict__ indices_cur,
    float*      __restrict__ features_cur,
    float*      __restrict__ shs_cur, 
    int*        __restrict__ starts, 
    float*      __restrict__ means3D, 
    float*      __restrict__ opacities, 
    float*      __restrict__ rotations, 
    float*      __restrict__ scales, 
    float*      __restrict__ shs, 
    float*      __restrict__ boxes, 
    int*        __restrict__ back_pointer, 
    int*        __restrict__ least_recently, 
    const int   featureMaxx 
){  
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= (to_update + to_expand))
        return ;
    int start = -1, b_idx=0;
    if (idx < to_update){
        start = update_indices[idx];
        b_idx = back_pointer[start];
        if (starts[b_idx] != -1) 
            starts[b_idx] = -1;
    } 
    else 
        start = idx - to_update + featureMaxx;
    int node_idx        = indices_cur[idx]; 
    if (starts[node_idx] != -1){ 
        least_recently[starts[node_idx]]   = 100;
        back_pointer[starts[node_idx]]     = -1;
    } 
    starts[node_idx]        = start; 
    back_pointer[start]     = node_idx;
    least_recently[start]   = 0;

    float* ptr = features_cur + 18 * idx;
    float* menas3D_ptr = means3D + start * 3;
    for(auto j = 0; j < 3; ++j) menas3D_ptr[j] = ptr[j];
    opacities[start] = ptr[3];
    ptr+=4;
    float* rot_ptr = rotations + start * 4;
    for(auto j = 0; j < 4; ++j) rot_ptr[j] = ptr[j];
    ptr += 4;
    float* sca_ptr = scales + start * 3;
    for(auto j = 0; j < 3; ++j) sca_ptr[j] = ptr[j];
    ptr += 3;
    float* box_ptr = boxes + start * 8;
    for(auto j = 0; j < 7; ++j) box_ptr[j] = ptr[j];
    ptr += 7;
    float* shs_ptr = shs + 48 * start;
    for (auto j = 0; j < 48; ++j) shs_ptr[j] = shs_cur[j + 48 * idx];
} 
// ===============================================================================
auto default_kernel_impl( 
    // input parameters 
    const SimpleNode *nodes, 
    const Box *boxes, 
    float3* means3D, 
    int N, 
    const Point *viewpoint, 
    float target_size, 
    int window_size, 
    float* view_transform, 
    float* projection_matrix, 
    bool frustum_culling, 
    // output parameters 
    int* least_recently, 
    int* render_indices, 
    int* node_indices 
) -> int {
    auto render_counts  = package.render_counts.data().get();
    auto render_offsets = package.render_offsets.data().get();
    auto &temp_storage  = package.temp_storage;
    auto temp_bytes     = std::size_t{};

    // requires_type(ReorderType::BFSOrder);
    markNodesForSize<<<(N + 255) / 256, 256>>>( 
        // input parameters 
        nodes, boxes, means3D, N, viewpoint, target_size, window_size, 
        view_transform, projection_matrix, frustum_culling, 
        // output parameters 
        least_recently, 
        render_counts 
    );

    cub::DeviceScan::InclusiveSum(nullptr, temp_bytes, render_counts, render_offsets, N);
    temp_storage.resize(temp_bytes);
    auto gpu_ptr = temp_storage.data().get();
    cub::DeviceScan::InclusiveSum(gpu_ptr, temp_bytes, render_counts, render_offsets, N);

    putRenderIndices<<<(N + 255) / 256, 256>>>( 
        nodes, boxes, N, render_indices, node_indices, render_counts, render_offsets
    );

    int count = 0;
    cudaMemcpy(&count, render_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
} 

auto layerwise_kernel_impl( 
    // input parameters 
    const SimpleNode *nodes, 
    const Box *boxes, 
    float3* means3D, 
    int N, 
    const Point *viewpoint, 
    float target_size, 
    int window_size, 
    float* view_transform, 
    float* projection_matrix, 
    bool frustum_culling, 
    // output parameters 
    int* least_recently, 
    int* render_indices, 
    int* node_indices, 
    // some other helpers 
    const std::vector<int> &depth_count 
) -> int { 
    auto is_smaller     = package.is_smaller.data().get();
    auto render_counts  = package.render_counts.data().get();
    auto render_offsets = package.render_offsets.data().get();
    auto &temp_storage  = package.temp_storage;
    auto temp_bytes     = std::size_t{};
    auto offset         = int{};

    for (const auto &count : depth_count) {
        const auto num_blocks = (count + 255) / 256;
        markNodesByDepth<<<num_blocks, 256>>>(
            nodes, boxes, means3D, count, viewpoint, target_size, window_size, 
            view_transform, projection_matrix, frustum_culling, 
            least_recently, render_counts, is_smaller, offset 
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
        nodes, boxes, N, render_indices, node_indices, render_counts, render_offsets
    );
    int count = 0;
    cudaMemcpy(&count, render_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    return count;
}
auto fusion_kernel_impl( 
    // input parameters 
    const SimpleNode *nodes, 
    const Box *boxes, 
    float3* means3D, 
    int N, 
    const Point *viewpoint, 
    float target_size, 
    int window_size, 
    float* view_transform, float* projection_matrix, bool frustum_culling, 
    // output parameters 
    int* least_recently, 
    int* render_indices, 
    int* node_indices, 
    // some other helpers 
    const std::vector<int> &depth_count 
) -> int { 
    // requires_type(ReorderType::BFSOrder);
    auto holder     = package.holder.data().get();
    auto is_smaller = package.is_smaller.data().get();
    auto offset     = int{};

    setHolder<<<1, 1>>>(package.holder.data().get()); 
    for (const auto &count : depth_count) {
        const auto num_blocks = (count + 255) / 256;
        markNodesByDepthFused<<<num_blocks, 256>>>(
            nodes, boxes, means3D, count, viewpoint, target_size, window_size, 
            view_transform, projection_matrix, frustum_culling, 
            is_smaller, offset, holder,
            least_recently, render_indices, node_indices 
        );
        offset += count;
    } 
    if (offset != N) 
        throw std::runtime_error("Error: overall count != N");
    return package.holder.data().get()->device_value();
}
std::tuple<int, float> dispatcher_kernel(
    // input parameters
    const SimpleNode* nodes, 
    const Box* boxes, 
    float3* means3D, 
    int N, 
    const Point* viewpoint_p, 
    float target_size, 
    int window_size, 
    float* view_transform, 
    float* projection_matrix, 
    bool frustum_culling, 
    // output parameters 
    int* least_recently, 
    int* render_indices,
    int* node_indices,
    // whether to use custom kernel 
    const std::vector<int> &depth_count, 
    int mode 
) {
    package.resize(N);

    enum {
        Fused     = 0,
        Default   = 1,
        LayerWise = 2,
    };
    int to_render = 0;
    float elapse = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    switch (mode) {
        case Default:
            cudaEventRecord(start);
            to_render = default_kernel_impl(
                nodes, 
                boxes, 
                means3D, 
                N, 
                viewpoint_p, 
                target_size, 
                window_size, 
                view_transform, 
                projection_matrix, 
                frustum_culling, 
                least_recently, 
                render_indices,
                node_indices 
            );
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapse, start, end);
            break;
        case LayerWise:
            cudaEventRecord(start);
            to_render = layerwise_kernel_impl( 
                nodes, 
                boxes, 
                means3D, 
                N, 
                viewpoint_p, 
                target_size, 
                window_size, 
                view_transform, 
                projection_matrix, 
                frustum_culling, 
                least_recently, 
                render_indices,
                node_indices,
                depth_count 
            );
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapse, start, end);
            break;
        case Fused:
            cudaEventRecord(start);
            to_render = fusion_kernel_impl(
                nodes, 
                boxes, 
                means3D, 
                N, 
                viewpoint_p, 
                target_size, 
                window_size, 
                view_transform, 
                projection_matrix, 
                frustum_culling, 
                least_recently, 
                render_indices,
                node_indices,
                depth_count 
            );
            cudaEventRecord(end);
            cudaEventSynchronize(end);
            cudaEventElapsedTime(&elapse, start, end);
            break;
        default: // unknown mode 
            throw std::runtime_error("Error: mode not supported");
    }
    return std::make_tuple(to_render, elapse);
} 
auto subGraphTreeInit( 
    int N, 
    int* indices_cur,
    float* features_cur,
    float* shs_cur, 
    int* starts,
    float* means3D,
    float* opacities,
    float* rotations,
    float* scales,
    float* shs,
    float* boxes,
    int*  back_pointer
) -> int{
    int num_blocks = (N + 255) / 256;
    SGTInitGPU<<<num_blocks, 256>>>(
        N, indices_cur, features_cur, shs_cur, 
        starts, means3D, opacities, rotations, scales, shs, boxes, back_pointer
    );
    return N; 
} 
auto subGraphTreeExpand( 
    int N, 
    const int*      starts, 
    const int*      parents, 
    const int       tree_height,
    const int*      depth_count, 
    const float3*   means3D,
    const Box *     boxes, 
    const float     target_size,
    const Point*    viewpoint, 
    const bool      frustum_culling, 
    const float*    view_transform, 
    const float* projection_matrix, 
    int* least_recently, 
    int* render_indices 
) -> int { 
    thrust::device_vector<Holder> holder(1);
    thrust::device_vector<bool> is_smaller(N);
    auto offset     = int{};

    setHolder<<<1, 1>>>(holder.data().get()); 
    for(auto i = 0; i < tree_height; ++i){
        const auto count = depth_count[i];
        const auto num_blocks = (count + 255) / 256;
        SGTExpandGPU<<<num_blocks, 256>>>( 
            count, offset, starts, parents, means3D, boxes, target_size, viewpoint, 
            frustum_culling, view_transform, projection_matrix, 
            is_smaller.data().get(), holder.data().get(),
            least_recently, render_indices 
        );
        offset += count;
    } 
    if (offset != N)
        throw std::runtime_error("Error: overall count != N");

    return holder.data().get()->device_value();
} 
auto subGraphTreeUpdate(
    const int N, 
    int*    indices_cur,
    float*  features_cur,
    float*  shs_cur, 
    int*    starts, 
    float*  means3D,
    float*  opacities,
    float*  rotations,
    float*  scales,
    float*  shs,
    float*  boxes,
    int*    back_pointer, 
    int*    least_recently,
    int     window_size, 
    const int featureMaxx 
)->int{ 
    thrust::device_vector<Holder> holder(1);
    thrust::device_vector<int> update_indices(featureMaxx);
    // thrust::device_vector<char> temp_storage;
    setHolder<<<1, 1>>>(holder.data().get());
    int num_blocks = (N + 255) / 256;
    putUpdateIndices<<<(featureMaxx + 255) / 256, 256>>>(featureMaxx, least_recently, window_size, holder.data().get(), update_indices.data().get());
    int to_update = holder.data().get()->device_value();
    int to_expand = 0;
    if(N >= to_update) 
        to_expand = N - to_update;
    else 
        to_update = N;
    updateCUDA<<<(to_update + to_expand + 255) / 256, 256>>>( 
        to_update, to_expand, update_indices.data().get(), 
        indices_cur, features_cur, shs_cur, 
        starts, means3D, opacities, rotations, scales, shs, boxes, back_pointer, 
        least_recently, featureMaxx 
    ); 
    to_expand += featureMaxx;
    return to_expand; 
} 
}; // namespace kernel
}; // namespace

// ===================================================================================
auto reorder_nodes(int N, int *_node, float *_box, int*_depth_count, int* parents) -> int {
    auto *node = reinterpret_cast<Node *>(_node);
    auto *box  = reinterpret_cast<Box *>(_box);
    auto tree_height = kernel::reorder_nodes(N, node, box, _depth_count);
    int num_blocks = (N + 255) / 256;
    kernel::putParents<<<num_blocks, 256>>>(node, parents, N);
    kernel::to_simple(node, N);
    kernel::package.resize(N);
    return tree_height;
} 
auto reorder_result(int N, int *arr) -> void {
    kernel::reorder_back_nodes(N, arr);
}

std::tuple<int, float> flashTreeTraversal(
    int N,
    int* nodes,
    float* boxes,
    float* means3D, 
    float target_size,
    float* viewpoint,
    float* view_transform, 
    float* projection_matrix, 
    bool frustum_culling,
    int window_size, 
    int* least_recently, 
    int* render_indices,
    int* node_indices,
    int mode 
    ) { 
    // kernel::requires_not_type(kernel::ReorderType::NotReordered);
    return kernel::dispatcher_kernel( 
        (SimpleNode *) nodes, 
        (Box *) boxes, 
        (float3*) means3D, 
        N, 
        (Point *) viewpoint, 
        target_size, 
        window_size, 
        view_transform, 
        projection_matrix, 
        frustum_culling, 
        least_recently, 
        render_indices, 
        node_indices, 
        kernel::depth_count_global, 
        mode
    );
} 

auto subGraphTreeInit(
    int     N, 
    int*    indices_cur,
    float*  features_cur,
    float*  shs_cur, 
    int*    starts,
    float*  means3D,
    float*  opacities,
    float*  rotations,
    float*  scales,
    float*  shs,
    float*  boxes,
    int*    back_pointer
) -> int{ 
    return kernel::subGraphTreeInit(
        N, indices_cur, features_cur, shs_cur, 
        starts, means3D, opacities, rotations, scales, shs, boxes, back_pointer 
    );
} 
std::tuple<int, float> subGraphTreeExpand(
    int N, 
    int* starts, 
    int* parents, 
    int tree_height, 
    int* depth_count,
    float* means3D,
    float* boxes,
    float target_size,
    float* viewpoint,
    bool frustum_culling, 
    float* view_transform,
    float* projection_matrix, 
    int* least_recently, 
    int* render_indices 
){ 
    float elapse = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    int ret = kernel::subGraphTreeExpand( 
        N, 
        starts, 
        parents, 
        tree_height, 
        depth_count, 
        (float3*) means3D,
        (Box *) boxes, 
        target_size,
        (Point*) viewpoint, 
        frustum_culling, 
        view_transform, 
        projection_matrix, 
        least_recently, 
        render_indices 
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse, start, end);
    return std::make_tuple(ret, elapse); 
} 
std::tuple<int, float> subGraphTreeUpdate( 
    int     N, 
    int*    indices_cur,
    float*  features_cur,
    float*  shs_cur, 
    int*    starts, 
    float*  means3D,
    float*  opacities,
    float*  rotations,
    float*  scales,
    float*  shs,
    float*  boxes,
    int*    back_pointer, 
    int*    least_recently,
    int     window_size, 
    const int   featureMaxx 
){ 
    float elapse = 0.0;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);

    int ret = kernel::subGraphTreeUpdate(
        N, indices_cur, features_cur, shs_cur, 
        starts, means3D, opacities, rotations, scales, shs, boxes, back_pointer, 
        least_recently, window_size, featureMaxx
    );
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&elapse, start, end);
    return std::make_tuple(ret, elapse); 
}
