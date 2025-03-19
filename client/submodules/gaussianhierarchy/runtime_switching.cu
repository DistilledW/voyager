/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <float.h>
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <iostream>
#include <algorithm>
#include <numeric>
#include <thrust/sequence.h>
#include <thrust/device_vector.h>
#include <cuda.h>
#include <fstream>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <nvtx3/nvToolsExt.h>
#include <thrust/host_vector.h>
#include <tuple>
#include "types.h"
#include "runtime_switching.h"

__global__ void markTargetNodes(Node* nodes, int N, int target, int* node_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int count = 0;
	Node node = nodes[idx];
	if (node.depth > target)
		count = node.count_leafs;
	else if (node.parent != -1)
	{
		Node parentnode = nodes[node.parent];
		if (parentnode.depth > target)
		{
			count = node.count_leafs;
			if (node.depth != 0)
				count += node.count_merged;
		}
	}
	node_counts[idx] = count;
}

__global__ void putRenderIndices(Node* nodes, int N, int* node_counts, int* node_offsets, int* render_indices, int* parent_indices = nullptr, int* nodes_for_render_indices = nullptr)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	Node node = nodes[idx];
	int count = node_counts[idx];
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];
	int start = node.start;
	
	int parentgaussian = -1;
	if (node.parent != -1)
	{
		parentgaussian = nodes[node.parent].start;
	}

	for (int i = 0; i < count; i++)
	{
		render_indices[offset + i] = node.start + i;
		if (parent_indices)
			parent_indices[offset + i] = parentgaussian; 
		if (nodes_for_render_indices)
			nodes_for_render_indices[offset + i] = idx;
	}
}

int Switching::expandToTarget(
	int N,
	int target,
	int* nodes,
	int* render_indices
)
{
	thrust::device_vector<int> render_counts(N);
	thrust::device_vector<int> render_offsets(N);

	int num_blocks = (N + 255) / 256;
	markTargetNodes << <num_blocks, 256 >> > ((Node*)nodes, N, target, render_counts.data().get());

	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);

	putRenderIndices << <num_blocks, 256 >> > ((Node*)nodes, N, render_counts.data().get(), render_offsets.data().get(), render_indices);

	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

__device__ bool inboxCUDA(Box& box, Point viewpoint)
{
	bool inside = true;
	for (int i = 0; i < 3; i++)
	{
		inside &= viewpoint.xyz[i] >= box.minn.xyz[i] && viewpoint.xyz[i] <= box.maxx.xyz[i];
	}
	return inside;
}

__device__ float pointboxdistCUDA(Box& box, Point viewpoint)
{
	Point closest = {
		max(box.minn.xyz[0], min(box.maxx.xyz[0], viewpoint.xyz[0])),
		max(box.minn.xyz[1], min(box.maxx.xyz[1], viewpoint.xyz[1])),
		max(box.minn.xyz[2], min(box.maxx.xyz[2], viewpoint.xyz[2]))
	};

	Point diff = {
		viewpoint.xyz[0] - closest.xyz[0],
		viewpoint.xyz[1] - closest.xyz[1],
		viewpoint.xyz[2] - closest.xyz[2]
	};

	return sqrt(diff.xyz[0] * diff.xyz[0] + diff.xyz[1] * diff.xyz[1] + diff.xyz[2] * diff.xyz[2]);
}

__device__ float computeSizeGPU(Box& box, Point viewpoint, Point zdir)
{
	if (inboxCUDA(box, viewpoint))
		return FLT_MAX;

	float min_dist = pointboxdistCUDA(box, viewpoint);

	return box.minn.xyz[3] / min_dist;
}

__global__ void changeNodesOnce(
	Node* nodes,
	int N,
	int* indices,
	Box* boxes,
	Point* viewpoint,
	Point zdir,
	float target_size,
	int* split,
	int* node_counts,
	int* node_ids,
	char* needs_children
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	float size = computeSizeGPU(boxes[node_id], *viewpoint, zdir);

	int count = 1; // repeat yourself
	char need_child = 0;
	if (size >= target_size)
	{
		if (node.depth > 0 && split[node_id] == 0) // split
		{
			if (node.start_children == -1)
			{
				node_ids[idx] = node_id;
				need_child = 1;
			}
			else
			{
				count += node.count_children;
				split[node_id] = 1;
			}
		}
	}
	else
	{
		int parent_node_id = node.parent;
		if (parent_node_id != -1)
		{
			Node parent_node = nodes[parent_node_id];
			float parent_size = computeSizeGPU(boxes[parent_node_id], *viewpoint, zdir);
			if (parent_size < target_size) // collapse
			{
				split[parent_node_id] = 0;
				count = 0; // forget yourself
			}
		}
	}
	needs_children[idx] = need_child;
	node_counts[idx] = count;
}

__global__ void putNodes(
	Node* nodes,
	int N,
	int* indices,
	int* node_counts,
	int* node_offsets,
	int* next_nodes)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int count = node_counts[idx];
	if (count == 0)
		return;

	int node_id = indices[idx];
	Node node = nodes[node_id];
	int offset = idx == 0 ? 0 : node_offsets[idx - 1];

	next_nodes[offset] = node_id;
	for (int i = 1; i < count; i++)
		next_nodes[offset + i] = node.start_children + i - 1;
}

__global__ void countRenderIndicesIndexed(Node* nodes, int* split, int N, int* node_indices, int* render_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_idx = node_indices[idx];

	Node node = nodes[node_idx];
	int count = node.count_leafs;
	if (node.depth > 0 && split[node_idx] == 0)
		count += node.count_merged;

	render_counts[idx] = count;
}

__global__ void putRenderIndicesIndexed(Node* nodes, int N, int* node_indices, int* render_counts, int* render_offsets, int* render_indices, int* parent_indices, int* nodes_of_render_indices, Box* boxes, float3* debug)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_idx = node_indices[idx];

	Node node = nodes[node_idx];
	int count = render_counts[idx];
	int offset = idx == 0 ? 0 : render_offsets[idx - 1];
	// int start = node.start;

	int parentgaussian = -1;
	if (node.parent != -1)
	{
		parentgaussian = nodes[node.parent].start;
	}

	for (int i = 0; i < count; i++)
	{
		render_indices[offset + i] = node.start + i;
		parent_indices[offset + i] = parentgaussian;
		nodes_of_render_indices[offset + i] = node_idx;
	}

	if (debug != nullptr)
	{
		Box box = boxes[node_idx];
		for (int i = 0; i < count; i++)
		{
			float red = min(1.0f, node.depth / 10.0f);
			debug[offset + i] = { red, 1.0f - red, 0 };
			if (node.depth == 0)
				debug[offset + i] = { 0, 0, 1.0f };
		}
	}
}

void Switching::changeToSizeStep(
	float target_size,
	int N,
	int* node_indices,
	int* new_node_indices,
	int* nodes,
	float* boxes,
	float* viewpoint,
	float x, float y, float z,
	int* split,
	int* render_indices,
	int* parent_indices,
	int* nodes_of_render_indices,
	int* nodes_to_expand,
	float* debug,
	char*& scratchspace,
	size_t& scratchspacesize,
	int* NsrcI,
	int* NdstI,
	char* NdstC,
	int* numI,
	int maxN,
	int& add_success,
	int* new_N,
	int* new_R,
	int* need_expansion,
	void* maintenanceStream)
{
	cudaStream_t stream = (cudaStream_t)maintenanceStream;

	int num_node_blocks = (N + 255) / 256;

	Point zdir = { x, y, z };

	int* num_to_expand = numI;
	int* node_counts = NsrcI, * node_offsets = NdstI, * node_ids = NdstI;
	char* need_children = NdstC;
	if (scratchspacesize == 0)
	{
		size_t testsize;

		cub::DeviceScan::InclusiveSum(nullptr, testsize, node_counts, node_offsets, maxN, stream);
		scratchspacesize = testsize;
		cub::DeviceSelect::Flagged(nullptr, testsize, node_ids, need_children, nodes_to_expand, num_to_expand, maxN, stream);
		scratchspacesize = std::max(testsize, scratchspacesize);

		if (scratchspace)
			cudaFree(scratchspace);
		scratchspacesize = testsize;
		cudaMalloc(&scratchspace, scratchspacesize);
	}

	changeNodesOnce << <num_node_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		N, 
		node_indices, 
		(Box*)boxes, 
		(Point*)viewpoint, 
		zdir, 
		target_size, 
		split, 
		node_counts, 
		node_ids, 
		need_children
		);

	cub::DeviceSelect::Flagged(scratchspace, scratchspacesize, node_ids, need_children, nodes_to_expand, num_to_expand, N, stream);
	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, node_counts, node_offsets, N, stream);

	cudaMemcpyAsync(need_expansion, num_to_expand, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaMemcpyAsync(new_N, node_offsets + N - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);

	if (*new_N > maxN)
	{
		add_success = 0;
		return;
	}

	putNodes << <num_node_blocks, 256, 0, stream>> > (
		(Node*)nodes,
		N, 
		node_indices, 
		node_counts, 
		node_offsets, 
		new_node_indices
		);

	int num_render_blocks = (*new_N + 255) / 256;
	int* render_counts = NsrcI, * render_offsets = NdstI;

	countRenderIndicesIndexed << <num_render_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		split, 
		*new_N, 
		new_node_indices, 
		render_counts
		);

	cub::DeviceScan::InclusiveSum(scratchspace, scratchspacesize, render_counts, render_offsets, *new_N, stream);

	putRenderIndicesIndexed << <num_render_blocks, 256, 0, stream >> > (
		(Node*)nodes, 
		*new_N, 
		new_node_indices, 
		render_counts, 
		render_offsets, 
		render_indices, 
		parent_indices, 
		nodes_of_render_indices, 
		(Box*)boxes,
		(float3*)debug
		);

	cudaMemcpyAsync(new_R, render_offsets + *new_N - 1, sizeof(int), cudaMemcpyDeviceToHost, stream);

	add_success = 1;
}

__global__ void markNodesForSize(Node* nodes, Box* boxes, int N, Point* viewpoint, Point zdir, float target_size, int* render_counts, int* node_markers)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = idx;
	Node node = nodes[node_id];
	float size = computeSizeGPU(boxes[node_id], *viewpoint, zdir);

	int count = 0;
	if (size >= target_size)
		count = node.count_leafs;
	else if (node.parent != -1)
	{
		float parent_size = computeSizeGPU(boxes[node.parent], *viewpoint, zdir);
		if (parent_size >= target_size)
		{
			count = node.count_leafs;
			if (node.depth != 0)
				count += node.count_merged;
		}
	}

	if (count != 0 && node_markers != nullptr)
		node_markers[node_id] = 1;

	if (render_counts != nullptr)
		render_counts[node_id] = count;
}

__global__ void computeTsIndexed(
	Box* child_boxes,
	Box* parent_boxes,
	int N,
	int* indices,
	Point viewpoint,
	Point zdir,
	float target_size,
	float* ts //,
	// int* kids
)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N)
		return;

	int node_id = indices[idx];

	float t;
	if (parent_boxes[node_id].minn.xyz[0] == parent_boxes[node_id].maxx.xyz[0] && parent_boxes[node_id].minn.xyz[0] == 0) 
		t = 1.0f; // parent boxes 不存在
	else
	{
		float parentsize = computeSizeGPU(parent_boxes[node_id], viewpoint, zdir);

		if (parentsize > 2.0f * target_size)
			t = 1.0f;
		else
		{
			float size = computeSizeGPU(child_boxes[node_id], viewpoint, zdir);
			float start = max(0.5f * parentsize, size);
			float diff = parentsize - start;

			if (diff <= 0)
				t = 1.0f;
			else
			{
				float tdiff = max(0.0f, target_size - start);
				t = max(1.0f - (tdiff / diff), 0.0f);
			}
		}
	}

	ts[idx] = t;
	// kids[idx] = (node.parent == -1) ? 1 : nodes[node.parent].count_children;
}

void Switching::getTsIndexed(
	int N,
	int* indices,
	float target_size,
	float* child_boxes,
	float* parent_boxes,
	float vx, float vy, float vz,
	float x, float y, float z,
	float* ts,
	// int* kids,
	void* stream
)
{
	Point zdir = { x, y, z };
	Point cam = { vx, vy, vz };
	int num_blocks = (N + 255) / 256;
	computeTsIndexed<<<num_blocks, 256, 0, (cudaStream_t)stream >>>(
		(Box*)child_boxes, 
		(Box*)parent_boxes, 
		N, 
		indices, 
		cam,
		zdir, 
		target_size, 
		ts); // , kids
		
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
	float3 p_orig,
	const float* viewmatrix,
	const float* projmatrix)
{
	// float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };

	// Bring points to screen space
	float4 p_hom = tPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	float3 p_view = tPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f)// || ((p_proj.x < -1.3 || p_proj.x > 1.3 || p_proj.y < -1.3 || p_proj.y > 1.3)))
	{
		return false;
	}
	return true;
}

__global__ void expand_markNodesForSize(int N, Box* c_boxes, Box*p_boxes, float* means3d, int threshold, Point* viewpoint, Point zdir, bool if_culling, float* view_transform, float*  projection_matrix, bool* leafs_tag, int* last_frame, int* render_counts)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N) 
		return;
	last_frame[idx]++;
	int curr_idx = idx;
	// 视锥检查，过滤掉所有在视锥外的点
	Box box = c_boxes[curr_idx];
	if (if_culling){
		float3 p_orig = { means3d[3 * idx], means3d[3 * idx + 1], means3d[3 * idx + 2] };
		if (!in_frustum(p_orig, view_transform, projection_matrix)){
			render_counts[idx] = 0;
			return ;
		}
	}
	float size = computeSizeGPU(c_boxes[curr_idx], *viewpoint, zdir);	// child size
	if (size >= threshold){
		// if (leafs_tag[curr_idx]){
			render_counts[curr_idx] = 1;
			last_frame[curr_idx] = 0;
		// }
	}else {
		float p_size = computeSizeGPU(p_boxes[curr_idx], *viewpoint, zdir);	// parent size
		if(p_size >= threshold){
			render_counts[curr_idx] = 1;
			last_frame[curr_idx] = 0;
		}
	} 
}
__global__ void expand_putRenderIndices(int N, Box* c_boxes, Box* p_boxes, float threshold, Point* viewpoint, Point zdir, int*render_indices, float* interpolation_weights, int*render_counts, int*render_offsets)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= N || render_counts[idx] <= 0) 
		return;
	int curr_idx = idx;
	int offset = render_offsets[curr_idx];
	
	float c_size = computeSizeGPU(c_boxes[curr_idx], *viewpoint, zdir);	// child size 
	float p_size = computeSizeGPU(p_boxes[curr_idx], *viewpoint, zdir);	// parent size 
	float t = 1.0f, start, diff;

	if (p_size >= 2.0f * threshold){
		t = 1.0f;
	}else{
		start = max(0.5f * p_size, c_size);
		diff = p_size - start;
		if (diff <= 1.0f)
			t = 1.0f;
		else{
			float tdiff = max(0.0f, threshold - start);
			t = max(1.0f - (tdiff / diff), 0.0f);
		}
	}
	render_indices[offset] = curr_idx;
	interpolation_weights[offset] = t;
}

int Switching::forceSearch(
	int N, 						// point number
	float* c_boxes, 			// children's boxes
	float* p_boxes, 			// parents' boxes
	float* means3d, 
	float threshold, 			// target size
	float* viewpoint,
	float x, float y, float z,
	bool if_culling,
	float* view_transform,
	float* projection_matrix,	
	bool* leafs_tag, 
	// Output 
	int* last_frame,			// 上一次用到该 gs point有多久。
	int* render_indices, 
	float* interpolation_weights) 
{
	size_t temp_storage_bytes;
	thrust::device_vector<char> temp_storage;
	thrust::device_vector<int> render_counts(N);
	thrust::device_vector<int> render_offsets(N);
	Point zdir = { x, y, z };
	int num_blocks = (N + 255) / 256;
	expand_markNodesForSize << <num_blocks, 256 >> > (N, (Box*)c_boxes, (Box*)p_boxes, means3d, threshold, (Point*)viewpoint, zdir, if_culling, \
		view_transform, projection_matrix, (bool*)leafs_tag, last_frame, render_counts.data().get());
	cub::DeviceScan::InclusiveSum(nullptr, temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	temp_storage.resize(temp_storage_bytes);
	cub::DeviceScan::InclusiveSum(temp_storage.data().get(), temp_storage_bytes, render_counts.data().get(), render_offsets.data().get(), N);
	expand_putRenderIndices << <num_blocks, 256 >> > (N, (Box*)c_boxes, (Box*)p_boxes, threshold, (Point*)viewpoint, zdir, render_indices, 
		interpolation_weights, render_counts.data().get(), render_offsets.data().get());
	int count = 0;
	cudaMemcpy(&count, render_offsets.data().get() + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
	return count;
}

void Switching::markVisibleForAllViewpoints(
	float target_size,
	int* nodes,
	int num_nodes,
	float* boxes,
	float* viewpoints,
	int num_viewpoints,
	int* seen,
	float zx,
	float zy,
	float zz
)
{
	thrust::device_vector<int> seen_cuda(num_nodes);
	thrust::device_vector<Point> viewpoint_cuda(1);
	thrust::device_vector<Node> nodes_cuda(num_nodes);
	thrust::device_vector<Box> boxes_cuda(num_nodes);

	cudaMemcpy(nodes_cuda.data().get(), nodes, sizeof(Node) * num_nodes, cudaMemcpyHostToDevice);
	cudaMemcpy(boxes_cuda.data().get(), boxes, sizeof(Box) * num_nodes, cudaMemcpyHostToDevice);

	Point zdir = { zx, zy, zz };

	Point* points = (Point*)viewpoints;
	int num_blocks = (num_nodes + 255) / 256;
	for (int i = 0; i < num_viewpoints; i++)
	{
		Point viewpoint = points[i];
		cudaMemcpy(viewpoint_cuda.data().get(), &viewpoint, sizeof(Point), cudaMemcpyHostToDevice);

		markNodesForSize << <num_blocks, 256 >> > (
			nodes_cuda.data().get(),
			boxes_cuda.data().get(),
			num_nodes,
			viewpoint_cuda.data().get(),
			zdir,
			target_size,
			nullptr,
			seen_cuda.data().get());
	}
	cudaMemcpy(seen, seen_cuda.data().get(), sizeof(int) * num_nodes, cudaMemcpyDeviceToHost);

	if (cudaDeviceSynchronize())
		std::cout << "Errors: " << cudaDeviceSynchronize() << std::endl;
}