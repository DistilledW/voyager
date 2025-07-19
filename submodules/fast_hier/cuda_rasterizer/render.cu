/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_texture_types.h>
namespace cg = cooperative_groups;

extern cudaTextureObject_t texObj;

// // Main rasterization method. Only compute the RGB framebuffer.
// template <uint32_t CHANNELS>
// __global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
// renderCUDA_Pure( 
// 	const uint2* __restrict__ ranges,
// 	const uint32_t* __restrict__ point_list,
// 	int W, int H,
// 	const float* ts,
// 	const int* kids,
// 	const float2* __restrict__ points_xy_image,
// 	const float* __restrict__ features,
// 	const float4* __restrict__ conic_opacity,
// 	const float* __restrict__ bg_color,
// 	float* __restrict__ out_color,
// 	int P, int skyboxnum) 
// {
// 	// Identify current tile and associated min/max pixel range.
// 	auto block = cg::this_thread_block();
// 	const uint32_t lane = block.thread_rank() % WARP_SIZE;
// 	const uint32_t warp_idx = block.thread_rank() / WARP_SIZE;

// 	// the placement of threads in a warp with 2*2 window size
// 	// 00 01 04 05 16 17 20 21
// 	// 02 03 06 07 18 19 22 23
// 	// 08 09 12 13 24 25 28 29
// 	// 10 11 14 15 26 27 30 31
// 	constexpr uint X_size = 8 / GROUP_X;

// 	const uint midrank = lane / GROUP_SIZE;
// 	const uint midx = midrank%X_size*GROUP_X;
// 	const uint midy = midrank/X_size*GROUP_Y;
// 	const uint headrank = lane % GROUP_SIZE;
// 	const uint headx = headrank%GROUP_X;
// 	const uint heady = headrank/GROUP_X;
// 	const uint2 pix_min = { 
// 		block.group_index().x*BLOCK_X + (warp_idx%2)*8 + midx,
// 		block.group_index().y*BLOCK_Y + (warp_idx/2)*4 + midy};
// 	const uint2 pix = { 
// 		pix_min.x + headx, 
// 		pix_min.y + heady};
// 	const uint32_t pix_id = W * pix.y + pix.x;
// 	const float2 pixf = { (float)pix.x, (float)pix.y };
// 	const float2 pix_cent = { 
// 		(float)(pix_min.x) + (float)(GROUP_X-1.0f)/2.0f, 
// 		(float)(pix_min.y) + (float)(GROUP_Y-1.0f)/2.0f };

// 	// Check if this thread is associated with a valid pixel or outside.
// 	bool inside = pix.x < W && pix.y < H;
// 	// Done threads can help with fetching, but don't rasterize
// 	bool done = !inside;

// 	// Load start/end range of IDs to process in bit sorted list.
// 	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
// 	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
// 	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
// 	int toDo = range.y - range.x;

// 	// Allocate storage for batches of collectively fetched data.
// 	__shared__ int collected_id[BLOCK_SIZE];
// 	__shared__ float2 collected_xy[BLOCK_SIZE];
// 	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
// 	__shared__ float2 collected_interp[BLOCK_SIZE];
// 	__shared__ bool precomp_alpha[BLOCK_SIZE];

// 	// Initialize helper variables
// 	float T = 1.0f;
// 	float C[CHANNELS] = { 0 };

// 	// Iterate over batches until all done or range is complete
// 	const int check = (P - skyboxnum);
// 	const bool do_interp = (ts != nullptr && kids != nullptr);

// 	float2 xy, d;
// 	float4 con_o;
// 	float power, alpha;
// 	int coll_id;
// 	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
// 	{
// 		// End if entire block votes that it is done rasterizing
// 		int num_done = __syncthreads_count(done);
// 		if (num_done == BLOCK_SIZE)
// 			break;

// 		// Collectively fetch per-Gaussian data from global to shared
// 		int progress = i * BLOCK_SIZE + block.thread_rank();
// 		if (range.x + progress < range.y)
// 		{
// 			coll_id = point_list[range.x + progress];
// 			collected_id[block.thread_rank()] = coll_id;
// 			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
// 			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
// 			if(coll_id < check && ts != nullptr && kids != nullptr)
// 			  collected_interp[block.thread_rank()] = { ts[coll_id], 1.0f / (float)kids[coll_id] };
// 		}
// 		block.sync();

// 		// Iterate over current batch
// 		int to_render = min(toDo, BLOCK_SIZE);
// 		for (int render_progress = 0; render_progress < to_render; render_progress+=GROUP_SIZE)
// 		{
// 			int index = render_progress+(block.thread_rank()%GROUP_SIZE);
// 			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
// 			prepare_data();
			
// 			precomp_alpha[block.thread_rank()] = (power > -0.01f || power < -6.0f);
// 			__syncwarp();


// 			if (!done)
// 			{
// 				for (int j = 0; j < min(to_render - render_progress, GROUP_SIZE); j++)
// 				{
// 					if (precomp_alpha[(block.thread_rank()&(~(GROUP_SIZE-1))) + j]) continue;
// 					index = render_progress+j;

// 					prepare_data();
// 					if (power > -0.01f || power < -6.0f) continue;

// 					alpha = compute_alpha(power, collected_interp[index], do_interp && coll_id < check);
// 					float test_T = T * (1 - alpha);
// 					if (test_T < 0.0001f)
// 					{
// 						done = true;
// 						continue;
// 					}

// 					// Eq. (3) from 3D Gaussian splatting paper.
// 					for (int ch = 0; ch < CHANNELS; ch++)
// 						C[ch] += features[coll_id * CHANNELS + ch] * alpha * T;

// 					T = test_T;
// 				}
// 			}
// 		}
// 	}

// 	// All threads that treat valid pixel write out their final
// 	// rendering data to the frame and auxiliary buffers.
// 	if (inside)
// 	{
// 		for (int ch = 0; ch < CHANNELS; ch++)
// 			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
// 	}
// }


__forceinline__ __device__ float compute_alpha(const float power, const float2 interp, const bool do_interp)
{
	// float alpha = min(0.99f, __expf(power + ln_opacity));
	// float tmp = ln_opacity + power;
	if (power > -0.01f || power < -6.0f) return 0.0f;

	float alpha = __expf(power);										
	if (do_interp)
	{
		float kidsqrt_alpha = 1.0f - __powf(1.0f - alpha, interp.y);
		alpha = interp.x * alpha + (1.0f - interp.x) * kidsqrt_alpha;
	}

	return alpha;
}

#define prepare_data(__pix, __index) do {\
	coll_id = collected_id[__index];														\
	xy = collected_xy[__index];															\
	d = { xy.x - __pix.x, xy.y - __pix.y };												\
	con_o = collected_conic_opacity[__index];												\
	power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y + con_o.w;	\
} while(0)

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
renderCUDA( 
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* ts,
	const int* kids,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int P, int skyboxnum,
	const float* __restrict__ depths,
	float* __restrict__ invdepth) 
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float2 collected_interp[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };

	float expected_invdepth = 0.0f;

	// Iterate over batches until all done or range is complete
	int check = (P - skyboxnum);
	bool do_interp = (ts != nullptr && kids != nullptr);

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
			if(coll_id < (P - skyboxnum) && ts != nullptr && kids != nullptr)
			  collected_interp[block.thread_rank()] = { ts[coll_id], 1.0f / (float)kids[coll_id] };
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;

			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y + con_o.w;
			// if (power > -0.01f || power < -5.55f)
			if (power > -0.01f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			// float tmp_idx_f = -power * inv_exp_lut_step;
			// int tmp_idx = (int)tmp_idx_f;
			// float my_alpha = exp_lut[tmp_idx];
			// float my_alpha = exp_lut[tmp_idx] * (tmp_idx+1.0f-tmp_idx) + 
			// 	(tmp_idx_f - tmp_idx) * exp_lut[tmp_idx+1];
			float my_alpha = exp(power);
			// float my_alpha = min(0.99f, con_o.w * exp(power));
			float alpha;
			int coll_id = collected_id[j];
			if (do_interp && coll_id < check) 
			{
				float2 interp = collected_interp[j];
				float kidsqrt_alpha = 1.0f - __powf(1.0f - my_alpha, interp.y);
				alpha = interp.x * my_alpha + (1.0f - interp.x) * kidsqrt_alpha;
			}
			else
			{
				alpha = my_alpha;
			}

			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[coll_id * CHANNELS + ch] * alpha * T;

			if(invdepth)
				expected_invdepth += (1 / depths[collected_id[j]]) * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];

		if (invdepth)
		invdepth[pix_id] = expected_invdepth;// 1. / (expected_depth + T * 1e3);
	}
}

// Main rasterization method. Only compute the RGB framebuffer.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
renderCUDA_Pure( 
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* ts,
	const int* kids,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	int P, int skyboxnum) 
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	const uint32_t lane = block.thread_rank() % WARP_SIZE;
	const uint32_t warp_idx = block.thread_rank() / WARP_SIZE;

	// the placement of threads in a warp with 2*2 window size
	// 00 01 02 03 04 05 06 07
	// 08 09 10 11 12 13 14 15
	// 16 17 18 19 20 21 22 23
	// 24 25 26 27 28 29 30 31
	const uint2 pix_min = { 
		block.group_index().x*BLOCK_X + (warp_idx%2)*8,
		block.group_index().y*BLOCK_Y + (warp_idx/2)*4};
	const uint2 pix = { 
		pix_min.x + lane%8, 
		pix_min.y + lane/8};
	const uint32_t pix_id = W * pix.y + pix.x;
	const float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W && pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float2 collected_interp[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	float C[CHANNELS] = { 0 };

	// Iterate over batches until all done or range is complete
	const int check = (P - skyboxnum);
	bool do_interp = (ts != nullptr && kids != nullptr);
	// do_interp = false;

	float2 xy, d;
	float4 con_o;
	float power, alpha;
	int coll_id;
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];

			if(coll_id < check && ts != nullptr && kids != nullptr)
			  collected_interp[block.thread_rank()] = { ts[coll_id], 1.0f / (float)kids[coll_id] };
		}
		block.sync();

		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{	
			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
			prepare_data(pixf, j);
			// if (power > -0.01f || power < -5.55f) continue;
			if(power < -5.55f) 
				continue;
			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).

			// float tmp_idx_f = power * inv_exp_lut_step;
			float tmp_idx_f = min(-0.001, power) * inv_exp_lut_step;
			int tmp_idx = floorf(tmp_idx_f);
			float my_alpha = lut[tmp_idx] * (tmp_idx+1.0f-tmp_idx_f) + 
				(tmp_idx_f - tmp_idx) * lut[tmp_idx+1];

			if (do_interp && coll_id < check) 
			{
				float2 interp = collected_interp[j];
				float kidsqrt_alpha = 1.0f - __powf(1.0f - my_alpha, interp.y);
				alpha = interp.x * my_alpha + (1.0f - interp.x) * kidsqrt_alpha;
			} 
			else
			{
				alpha = my_alpha;
			}

			if (alpha < 1.0f / 255.0f)
				continue;
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[coll_id * CHANNELS + ch] * alpha * T;

			T = test_T;
		}
	}

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}


void FORWARD::render( 
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* ts,
	const int* kids,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	int P,
	int skyboxnum,
	cudaStream_t stream,
	float* depths,
	float* depth) 
{
#if PURE_FORWARD
	renderCUDA_Pure<NUM_CHANNELS> << <grid, block, 0, stream >> > (
		ranges,
		point_list,
		W, H,
		ts,
		kids,
		means2D,
		colors,
		conic_opacity,
		bg_color,
		out_color,
		P,
		skyboxnum);
#else
	renderCUDA<NUM_CHANNELS> << <grid, block, 0, stream >> > (
		ranges,
		point_list,
		W, H,
		ts,
		kids,
		means2D,
		colors,
		conic_opacity,
		final_T,
		n_contrib,
		bg_color,
		out_color,
		P,
		skyboxnum,
		depths, 
		depth);
#endif
}
