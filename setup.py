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

from setuptools import setup, find_packages
import os

setup(
    name="simple_knn_t",
    packages=find_packages(),
    package_data={'simple_knn': ['csrc/*']},
    include_package_data=True
)
