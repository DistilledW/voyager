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
auto reorder_nodes(int N, int *, float *, int* , int*) -> int;
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
    int mode 
);

// client functions 
auto subGraphTreeInit( 
    int     N, 
    int*    indices_cur,
    float*  features_cur,
    float*  shs_cur, 
    int*    starts,
    float*  means3D,
    float*  opacities,
    float*  rotations,
    float*  scales,
    float*  shs,
    float*  boxes,
    int*    back_pointer 
) -> int;

std::tuple<int, float> subGraphTreeExpand(
    int N, 
    int* starts, 
    int* parents, 
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
    int* indices_cur,
    float* features_cur,
    float* shs_cur, 
    int* starts, 
    float* means3D,
    float* opacities,
    float* rotations,
    float* scales,
    float* shs,
    float* boxes,
    int* back_pointer, 
    int* least_recently,
    int window_size, 
    const int featureMaxx 
);