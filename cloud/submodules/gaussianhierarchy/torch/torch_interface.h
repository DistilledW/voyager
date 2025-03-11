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

int ExpandToSize(
	torch::Tensor& nodes, 
	torch::Tensor& boxes, 
	torch::Tensor& means3D, 
	float threshold, 
	torch::Tensor& viewpoint, 
	torch::Tensor& viewdir, 
	int frame_index, int window_size,
	torch::Tensor& world_view_transform,
	torch::Tensor& projection_matrix, 
	// torch::Tensor& frustum_plans, 
	// list for clients
	torch::Tensor& last_frame, 
	torch::Tensor& child_indices,
	torch::Tensor& parent_indices,
	torch::Tensor& child_box_indices, 
	torch::Tensor& parent_box_indices, 
	torch::Tensor& leafs_tag,
	torch::Tensor& num_siblings);
void GetTsIndexed(
	torch::Tensor& indices,
	float size,
	torch::Tensor& nodes,
	torch::Tensor& boxes,
	torch::Tensor& viewpoint, 
	torch::Tensor& viewdir, 
	torch::Tensor& ts,
	torch::Tensor& num_kids);