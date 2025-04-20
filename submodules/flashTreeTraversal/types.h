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

#include <cstdint>
#include <cstdio>
#include <tuple>
#include <string>

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

struct Box{
	Box(Eigen::Vector3f minn, Eigen::Vector3f maxx) : minn(minn.x(), minn.y(), minn.z(), 0), maxx(maxx.x(), maxx.y(), maxx.z(), 0){}
	Box() {};
	Eigen::Vector4f minn;
	Eigen::Vector4f maxx;
};
typedef Eigen::Matrix<float, 48, 1> SHs;
#else

struct Point{
	float xyz[3];
};

struct Point4{
	float xyz[4];
};

struct Box{
	Point4 minn;
	Point4 maxx;
};
#endif
struct SimpleNode; 
struct Node
{
	int depth = -1;
	int parent = -1;
	int start;
	int count_leafs;
	int count_merged;
	int start_children;
	int count_children;
	__device__ __host__ operator SimpleNode() const;
};
struct HalfNode
{
	int parent = -1;
	int start;
	int start_children;
	short dccc[4];
};
struct SimpleNode {
    std::uint8_t depth = static_cast<std::uint8_t>(-1);
    std::uint8_t count_merged;
    std::uint16_t count_children;

    int parent = -1;
    int start;
    int start_children;

    __device__ __host__ auto is_leaf() const -> bool {
        return depth == 0;
    }

    __device__ __host__ auto is_root() const -> bool {
        return parent == -1;
    }

    __device__ __host__ operator Node() const {
        return {// clang-format off
            static_cast<int>(depth),
            static_cast<int>(parent),
            start,
            static_cast<int>(is_leaf()),
            static_cast<int>(count_merged),
            start_children,
            static_cast<int>(count_children)
        };
        // clang-format on
    }
};
inline __device__ __host__ Node::operator SimpleNode() const {
    return {
        static_cast<std::uint8_t>(depth),
        static_cast<std::uint8_t>(count_merged),
        static_cast<std::uint16_t>(count_children),
        parent,
        start,
        start_children
    };
}
struct SGTNode{
	int parent 	= -1;
	int start 	= -1;
};
