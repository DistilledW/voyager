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
namespace cg = cooperative_groups;

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y) 
div_probeCUDA( 
	int2 pix_min, int batch_i,
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float* ts,
	const int* kids,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	const float* __restrict__ bg_color,
	float* __restrict__ div_buffer,
	int P, int skyboxnum
) 
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	int2 block_idx = {pix_min.x / BLOCK_X, pix_min.y / BLOCK_Y};
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	float2 pixf = { (float)pix.x, (float)pix.y };
	int pix_id = block.thread_rank();

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block_idx.y * horizontal_blocks + block_idx.x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	__shared__ float2 collected_interp[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;

	// Iterate over batches until all done or range is complete
	int check = (P - skyboxnum);
	bool do_interp = (ts != nullptr && kids != nullptr);

	for (int i = 0; i < rounds && i <= batch_i; i++, toDo -= BLOCK_SIZE)
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

			// Resample using conic matrix (cf. "Surface Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix).
			float my_alpha = min(0.99f, con_o.w * exp(power));
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
			{
				if (batch_i == i)
				{
					div_buffer[pix_id * 256 + j] = power;
				}
				continue;
			}
			else
			{
				if (batch_i == i)
				{
					div_buffer[pix_id * 256 + j] = -power;
					// div_buffer[pix_id * 256 + j] = 1;
				}
			}


			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}

			T = test_T;
		}
	}

}

void FORWARD::divergence( 
	const int2 pix_min, const int batch_i,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float* ts,
	const int* kids,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	const float* bg_color,
	float* div_buffer,
	int P,
	int skyboxnum,
	cudaStream_t stream
	) 
{
	dim3 grid(1, 1, 1);
	dim3 block(BLOCK_X, BLOCK_Y, 1);
	div_probeCUDA << <grid, block, 0, stream >> > (
		pix_min, batch_i,
		ranges,
		point_list,
		W, H,
		ts,
		kids,
		means2D,
		colors,
		conic_opacity,
		bg_color,
		div_buffer,
		P,
		skyboxnum);
}
