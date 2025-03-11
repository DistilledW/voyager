#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
LoadHierarchy(std::string filename);

void WriteHierarchy(
					std::string filename,
					torch::Tensor& pos,
					torch::Tensor& shs,
					torch::Tensor& opacities,
					torch::Tensor& log_scales,
					torch::Tensor& rotations,
					torch::Tensor& nodes,
					torch::Tensor& boxes);

torch::Tensor
ExpandToTarget(torch::Tensor& nodes, int target);

int ForceSearch(
	torch::Tensor& child_boxes, 
	torch::Tensor& parent_boxes, 
	torch::Tensor& child_means3D, 
	float threshold, 
	torch::Tensor& viewpoint, 
	bool if_culling,
	torch::Tensor& view_transform,
	torch::Tensor& projection_matrix,
	torch::Tensor& viewdir, 
	torch::Tensor& leafs_tag,
	// output:
	torch::Tensor& last_frame,
	torch::Tensor& render_indices,
	torch::Tensor& interpolation_weights);
	
void GetTsIndexed(
	torch::Tensor& indices,
	float size,
	torch::Tensor& child_boxes,
	torch::Tensor& parent_boxes,
	torch::Tensor& viewpoint, 
	torch::Tensor& viewdir, 
	torch::Tensor& ts );//,
//	torch::Tensor& num_kids)
