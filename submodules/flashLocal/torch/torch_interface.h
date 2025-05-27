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

torch::Tensor ExpandToTarget(torch::Tensor& nodes, int target);

std::tuple<int, float> ExpandToSize(
torch::Tensor& nodes, 
torch::Tensor& boxes, 
torch::Tensor& means3D,
float size, 
torch::Tensor& viewpoint, 
torch::Tensor& view_transform,
torch::Tensor& projection_matrix, 
torch::Tensor& viewdir, 
torch::Tensor& render_indices, 
int mode);

float GetTsIndexed(
torch::Tensor& indices,
float size,
torch::Tensor& nodes,
torch::Tensor& boxes,
torch::Tensor& viewpoint, 
torch::Tensor& viewdir, 
torch::Tensor& ts,
torch::Tensor& num_kids);

void ReorderNodes(torch::Tensor& nodes, torch::Tensor& boxes, bool grouped=false);
void ReorderResult(torch::Tensor& nodes, torch::Tensor& nodes_for_render_indices);
