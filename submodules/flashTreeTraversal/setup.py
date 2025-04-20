#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

setup(
    name="flash_tree_traversal",
    ext_modules = [
        CUDAExtension(
            name="flash_tree_traversal._C",
            sources=[ 
                "hierarchy_loader.cpp",
                "hierarchy_writer.cpp",
                "flash_tree_traversal.cu",
                "torch/torch_interface.cpp",
                "ext.cpp"
            ],
            extra_compile_args={
                "cxx": [
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "dependencies/eigen/"),
                    "-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "half.hpp") 
                ]
            }
        )
    ],
    extra_compile_args=['-O3'],
    cmdclass={
        'build_ext': BuildExtension
    }
)