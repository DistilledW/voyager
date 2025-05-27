/*
 * Copyright (C) 2024, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#pragma once

#ifndef __CUDACC__

#ifndef __device__
#define __device__
#endif
#ifndef __host__
#define __host__
#endif
#ifndef __global__
#define __global__
#endif

#include <Eigen/Core>

struct Box {
    Box(Eigen::Vector3f minn, Eigen::Vector3f maxx) :
        minn(minn.x(), minn.y(), minn.z(), 0), maxx(maxx.x(), maxx.y(), maxx.z(), 0) {}

    Box() {};

    Eigen::Vector4f minn;
    Eigen::Vector4f maxx;
};

typedef Eigen::Matrix<float, 48, 1> SHs;

#else

struct Point {
    float xyz[3];
};

struct Point4 {
    float xyz[4];
};

struct Box {
    Point4 minn;
    Point4 maxx;
};

#endif

struct SimpleNode;

struct Node {
    int depth  = -1;
    int parent = -1;
    int start;
    int count_leafs;
    int count_merged;
    int start_children;
    int count_children;

    __device__ __host__ operator SimpleNode() const;
};

struct HalfNode {
    int parent = -1;
    int start;
    int start_children;
    short dccc[4];
};
