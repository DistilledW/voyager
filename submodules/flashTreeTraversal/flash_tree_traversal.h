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
#pragma once

#include "types.h"
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <string>

// cloud functions 
auto reorder_nodes(int N, int *, float *, int* ) -> int;
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
    int* num_siblings, 
    int mode 
);

std::tuple<int, float> transimissionCompress(
    // input parameters
    int N, 
    int* render_indices,
    int* node_indices,
    int* num_siblings, 
    // initial data
    float* means3D, 
    float* opacities,
    float* rotations,
    float* scales,
    float* shs,
    float* boxes,
    // output 
    uint8_t* data_to_pass,
    uint8_t* sizes,
    float opacity_min, float inv_range 
);
// client functions 
auto subGraphTreeInit( 
    int N, 
    uint8_t* compressed_data, 
    uint8_t* sizes,
    int* nodes, 
    float* means3D,
    float* opacities,
    float* rotations,
    float* scales,
    float* shs,
    float* boxes,
    int* num_siblings,
    float opacity_min, float range_255 
) -> int;

std::tuple<int, float> subGraphTreeExpand(
    int N, 
    int* nodes, 
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
);
std::tuple<int, float> subGraphTreeUpdate( 
    int N,
    uint8_t* compressed_data,
    uint8_t* sizes,
    int* nodes, 
    float* means3D,
    float* opacities,
    float* rotations,
    float* scales,
    float* shs,
    float* boxes,
    int* num_siblings, 
    int* least_recently, int window_size,
    float opacity_min, float range_255, int featureMaxx
);