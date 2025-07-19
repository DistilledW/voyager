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

struct MatMat
{
	float vals[16];
};

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int tidx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx]; // means3d
	glm::vec3 dir = pos - campos; dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs; // shape = (16, 3), in which max coefficients equals 3. 
	glm::vec3 result = SH_C0 * sh[0]; 

	if (deg > 0) // Always equals 3 
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result + 
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * tidx + 0] = (result.x < 0);
	clamped[3 * tidx + 1] = (result.y < 0);
	clamped[3 * tidx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

__forceinline__ __device__ glm::vec3 interp(glm::vec3* a, glm::vec3* b, int i, float t)
{
	glm::vec3 arr = a[i];
	glm::vec3 brr;
	brr = b[i];
	return t * arr + (1.0f - t) * brr;
}

__device__ glm::vec3 computeColorFromSHInterp(int idx, int p_idx, int tidx, float t, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3* sh_ = ((glm::vec3*)shs) + p_idx * max_coeffs;
	glm::vec3 result = SH_C0 * interp(sh, sh_, 0, t);

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * interp(sh, sh_, 1, t) + SH_C1 * z * interp(sh, sh_, 2, t) - SH_C1 * x * interp(sh, sh_, 3, t);

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * interp(sh, sh_, 4, t) +
				SH_C2[1] * yz * interp(sh, sh_, 5, t) +
				SH_C2[2] * (2.0f * zz - xx - yy) * interp(sh, sh_, 6, t) +
				SH_C2[3] * xz * interp(sh, sh_, 7, t) +
				SH_C2[4] * (xx - yy) * interp(sh, sh_, 8, t);

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * interp(sh, sh_, 9, t) +
					SH_C3[1] * xy * z * interp(sh, sh_, 10, t) +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * interp(sh, sh_, 11, t) +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * interp(sh, sh_, 12, t) +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * interp(sh, sh_, 13, t) +
					SH_C3[5] * z * (xx - yy) * interp(sh, sh_, 14, t) +
					SH_C3[6] * x * (xx - 3.0f * yy) * interp(sh, sh_, 15, t);
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * tidx + 0] = (result.x < 0);
	clamped[3 * tidx + 1] = (result.y < 0);
	clamped[3 * tidx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	// mod equals 1 
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const int* indices, const int* parent_indices,

	const float* ts,
	const float* orig_points, // mean3d
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,

	bool* clamped,
	bool* clamped_p,
	const float* cov3D_precomp, const float* colors_precomp, // nullptr
	const float* viewmatrixs, const float* projmatrixs,
	const glm::vec3* cam_posp,
	const int W, int H,
	const float tan_fovx, float tan_fovy, 
	const float focal_x, float focal_y, 
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	int skyboxnum, // 0
	float biglimit,
	MatMat viewmats,
	MatMat projmats, 
	glm::vec3 cam_pos)
{
	const float* viewmatrix = (viewmatrixs == nullptr) ? viewmats.vals : viewmatrixs;
	const float* projmatrix = (projmatrixs == nullptr) ? projmats.vals : projmatrixs;

	if (cam_posp != nullptr)
		cam_pos = *cam_posp;

	auto t_idx = cg::this_grid().thread_rank(); // target index 
	if (t_idx >= P)
		return;

	bool sky = t_idx >= (P - skyboxnum);
	int r_idx; // reverse idx
	if (sky)
	{
		r_idx = -(t_idx - (P - skyboxnum)) - 1;
		parent_indices = nullptr; // Sky has no parents
	}
	else
	{
		r_idx = indices == nullptr ? t_idx : indices[t_idx];
	}

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[t_idx] = 0;
	tiles_touched[t_idx] = 0;

	int p_idx = 0;
	float t = 0;

	// Perform near culling, quit if outside.
	float3 p_orig = { orig_points[3 * r_idx], orig_points[3 * r_idx + 1], orig_points[3 * r_idx + 2] };

	if (parent_indices != nullptr) // nullptr
	{
		p_idx = parent_indices[t_idx];
		if (p_idx == -1)
			parent_indices = nullptr; // be safe
		else
			t = ts[t_idx];
	}
	if (parent_indices != nullptr)
	{
		float3 pa_orig = { orig_points[3 * p_idx], orig_points[3 * p_idx + 1], orig_points[3 * p_idx + 2] };

		p_orig = {
			t * p_orig.x + (1.0f - t) * pa_orig.x,
			t * p_orig.y + (1.0f - t) * pa_orig.y,
			t * p_orig.z + (1.0f - t) * pa_orig.z
		};
	}

	// Bring points to screen space 
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
	float3 p_view = transformPoint4x3(p_orig, viewmatrix);

	if (p_view.z <= 0.2f) // 筛点 
		return;

	// if (p_orig.x < boxmin.x || p_orig.y < boxmin.y || p_orig.z < boxmin.z ||
	// 	p_orig.x > boxmax.x || p_orig.y > boxmax.y || p_orig.z > boxmax.z)
	// 	return;
	if (p_proj.x < boxmin.x || p_proj.y < boxmin.y || p_proj.z < boxmin.z ||
		p_proj.x > boxmax.x || p_proj.y > boxmax.y || p_proj.z > boxmax.z)
		return;

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 

	float* cov3D;
	
	if (cov3D_precomp == nullptr)
	{
		glm::vec3 scale = scales[r_idx];
		glm::vec4 rot = rotations[r_idx];
		if (parent_indices != nullptr)
		{
			scale = t * scale + (1.0f - t) * scales[p_idx];
			glm::vec4 otherrot = rotations[p_idx];

			float dot_product = glm::dot(rot, otherrot);
			if (dot_product < 0.0)
			{
				otherrot = -otherrot;
			}
			rot = t * rot + (1.0f - t) * otherrot;
		}

		if (!sky && max(scale.z, max(scale.x, scale.y)) > biglimit)
			return;

		computeCov3D(scale, scale_modifier, rot, cov3Ds + t_idx * 6);
		cov3D = cov3Ds + t_idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	constexpr float h_var = 0.3f;
	const float det_cov = cov.x * cov.z - cov.y * cov.y;
	cov.x += h_var;
	cov.z += h_var;
	const float det_cov_plus_h_cov = cov.x * cov.z - cov.y * cov.y;

#ifdef DGR_FIX_AA // 抗锯齿修正 
	const float h_convolution_scaling = sqrt(max(0.000025f, det_cov / det_cov_plus_h_cov)); // max for numerical stability
#endif 

	// Invert covariance (EWA algorithm)
	const float det = det_cov_plus_h_cov;

	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };

	// Compute extent in screen space (by finding eigenvalues of
	// 2D covariance matrix). Use extent to compute a bounding rectangle
	// of screen-space tiles that this Gaussian overlaps with. Quit if
	// rectangle covers 0 tiles. 

	float mid = 0.5f * (cov.x + cov.z);
	float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
	float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
	float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	uint2 rect_min, rect_max;
	// Inverse 2D covariance and opacity neatly pack into one float4
	float opacity = opacities[r_idx];
	if (parent_indices != nullptr)
		opacity = t * opacity + (1.0f - t) * opacities[p_idx];

	if (rects == nullptr) 	// More conservative
	{
		getRect(point_image, my_radius, rect_min, rect_max, grid);
	}
	else // Slightly more aggressive, might need a math cleanup
	{
		// Choose this, get a rectangle to fix the gaussian ball 
#if OPACITY_AABB
		const float mahala_dist = max(0.f, min(2.0f * (lowest_alpha_coeff + logf(opacity)), 9.0f));
#else
		const float mahala_dist = 9.0f;
#endif

		const int2 my_rect = { (int)ceil(sqrt(cov.x * mahala_dist)), (int)ceil(sqrt(cov.z * mahala_dist)) };
		rects[t_idx] = my_rect;
		getRect(point_image, my_rect, rect_min, rect_max, grid);
	}

	if ((rect_max.x - rect_min.x) * (rect_max.y - rect_min.y) == 0)
		return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr){
		glm::vec3 result;
		if (parent_indices == nullptr)
		{
			result = computeColorFromSH(r_idx, t_idx, D, M, (glm::vec3*)orig_points, cam_pos, shs, clamped);
		}
		else
		{
			result = computeColorFromSHInterp(r_idx, p_idx, t_idx, t, D, M, (glm::vec3*)orig_points, cam_pos, shs, clamped_p);
		}

		rgb[t_idx * C + 0] = result.x;
		rgb[t_idx * C + 1] = result.y;
		rgb[t_idx * C + 2] = result.z;
	} 

	// Store some useful helper data for the next steps.
	depths[t_idx] = p_view.z;
	radii[t_idx] = my_radius;
	points_xy_image[t_idx] = point_image;

#ifdef DGR_FIX_AA
	conic_opacity[t_idx] = { conic.x, conic.y, conic.z, logf(opacity * h_convolution_scaling) };
#else
	conic_opacity[t_idx] = { conic.x, conic.y, conic.z, logf(opacity) };
#endif

	tiles_touched[t_idx] = (rect_max.y - rect_min.y) * (rect_max.x - rect_min.x);
}


void FORWARD::preprocess(int P, int D, int M,
	const int* indices, const int* parent_indices, // nullptr
	const float* ts,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	bool* p_clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	int2* rects,
	float3 boxmin,
	float3 boxmax,
	int skyboxnum,
	cudaStream_t stream,
	float biglimit,
	bool on_cpu)
{
	MatMat viewview, projproj;
	if (on_cpu){
		for (int i = 0; i < 16; i++)
		{
			viewview.vals[i] = viewmatrix[i];
			projproj.vals[i] = projmatrix[i];
		}
	}

	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256, 0, stream >> > (
		P, D, M,
		indices, parent_indices,
		ts,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		p_clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix,
		projmatrix,
		on_cpu ? nullptr : cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		rects,
		boxmin,
		boxmax,
		skyboxnum,
		biglimit,
		viewview,
		projproj,
		on_cpu ? *cam_pos : glm::vec3()
		);
}