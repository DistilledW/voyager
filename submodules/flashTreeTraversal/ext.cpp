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

#include <torch/extension.h>
#include "torch/torch_interface.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("load_hierarchy", &LoadHierarchy);
    m.def("write_hierarchy", &WriteHierarchy);
    m.def("reorder_nodes", &ReorderNodes);
    m.def("flash_tree_traversal", &FlashTreeTraversal);
    m.def("transimission_compress", &TransimissionCompress);
    m.def("subgraph_tree_init", &SubGraphTreeInit);
    m.def("subgraph_expand", &SubGraphTreeExpand); 
    m.def("subgraph_update", &SubGraphTreeUpdate);
}
