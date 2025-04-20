#pragma once
#include <torch/extension.h>
#include <cstdio>
#include <tuple>
#include <string>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> LoadHierarchy(std::string filename);
void WriteHierarchy(std::string filename, torch::Tensor& pos, torch::Tensor& shs, torch::Tensor& opacities, torch::Tensor& log_scales, torch::Tensor& rotations, torch::Tensor& nodes, torch::Tensor& boxes);

// cloud functions 
int ReorderNodes(torch::Tensor& nodes, torch::Tensor& boxes, torch::Tensor& depth_count);
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
	torch::Tensor& num_siblings,
	int mode 
);

std::tuple<int, float> TransimissionCompress( 
	// input parameters 
	int N, 
	torch::Tensor& render_indices, 
	torch::Tensor& node_indices, 
	torch::Tensor& num_siblings, 
	// initial data 
	torch::Tensor& means3D, 
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	// output 
	torch::Tensor& data_to_pass,
	torch::Tensor& sizes,
	float opacity_min, float inv_range
);

// client functions 
int SubGraphTreeInit(
	torch::Tensor& compressed_data,
	torch::Tensor& sizes,
	torch::Tensor& nodes,
	torch::Tensor& means3D,
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	torch::Tensor& num_siblings,
	float opacity_min, float range_255 
);

std::tuple<int, float> SubGraphTreeExpand(
	torch::Tensor& nodes,
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
	torch::Tensor& compressed_data,
	torch::Tensor& sizes,
	torch::Tensor& nodes,
	torch::Tensor& means3D,
	torch::Tensor& opacities,
	torch::Tensor& rotations,
	torch::Tensor& scales,
	torch::Tensor& shs,
	torch::Tensor& boxes,
	torch::Tensor& num_siblings,
	torch::Tensor& least_recently, int window_size, 
	float opacity_min, float range_255, int featureMaxx 
);
