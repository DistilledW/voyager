#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LoadHierarchy(std::string filename);
void WriteHierarchy(std::string filename, torch::Tensor& pos, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& log_scales, torch::Tensor& rotations, torch::Tensor& nodes, torch::Tensor& boxes);

// cloud functions 
int ReorderNodes(torch::Tensor& nodes, torch::Tensor& boxes, torch::Tensor& depth_count, torch::Tensor& parents);
std::tuple<int, float> FlashTreeTraversal( 
	torch::Tensor& nodes, 
	torch::Tensor& boxes, 
	torch::Tensor& means3D,
	float target_size, 
	torch::Tensor& viewpoint, 
	torch::Tensor& view_transform,
	torch::Tensor& projection_matrix, 
	bool frustum_culling, 
	int window_size, 
	torch::Tensor& least_recently, 
	torch::Tensor& render_indices,
	torch::Tensor& node_indices,
	int mode 
);

// client functions 
int SubGraphTreeInit(
	int 	N,
	torch::Tensor& indices_cur,
	torch::Tensor& features_cur,
	torch::Tensor& shs_cur,
	torch::Tensor& starts,
	torch::Tensor& means3D,
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	torch::Tensor& back_pointer
);

std::tuple<int, float> SubGraphTreeExpand(
	torch::Tensor& starts, 
	torch::Tensor& parents,
	torch::Tensor& depth_count, 
	torch::Tensor& means3D, 
	torch::Tensor& boxes, 
	float threshold, 
	torch::Tensor& viewpoint, 
	bool frustum_culling, 
	torch::Tensor& view_transform,
	torch::Tensor& projection_matrix,
	torch::Tensor& least_recently,
	torch::Tensor& render_indices 
);

std::tuple<int, float> SubGraphTreeUpdate( 
	int N, 
	torch::Tensor& indices_cur,
	torch::Tensor& features_cur,
	torch::Tensor& shs_cur, 
	torch::Tensor& starts,
	torch::Tensor& means3D,
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	torch::Tensor& back_pointer,
	torch::Tensor& least_recently, 
	int window_size, 
	const int featureMaxx 
);
