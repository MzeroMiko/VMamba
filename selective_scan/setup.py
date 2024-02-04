# Modified by Mzero #20240123
# Copyright (c) 2023, Albert Gu, Tri Dao.
import sys
import warnings
import os
import re
import ast
from pathlib import Path
from packaging.version import parse, Version
import platform
import shutil

from setuptools import setup, find_packages
import subprocess
from wheel.bdist_wheel import bdist_wheel as _bdist_wheel

import torch
from torch.utils.cpp_extension import (
    BuildExtension,
    CppExtension,
    CUDAExtension,
    CUDA_HOME,
)

# ninja build does not work unless include_dirs are abs path
this_dir = os.path.dirname(os.path.abspath(__file__))
# For CI, we want the option to build with C++11 ABI since the nvcr images use C++11 ABI
FORCE_CXX11_ABI = os.getenv("FORCE_CXX11_ABI", "FALSE") == "TRUE"

def get_cuda_bare_metal_version(cuda_dir):
    raw_output = subprocess.check_output(
        [cuda_dir + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = raw_output.split()
    release_idx = output.index("release") + 1
    bare_metal_version = parse(output[release_idx].split(",")[0])

    return raw_output, bare_metal_version


def get_ext():
    ext_modules = []
    cc_flag = []

    print("\n\ntorch.__version__  = {}\n\n".format(torch.__version__))

    # Check, if CUDA11 is installed for compute capability 8.0
    if CUDA_HOME is not None:
        _, bare_metal_version = get_cuda_bare_metal_version(CUDA_HOME)
        if bare_metal_version < Version("11.6"):
            raise RuntimeError(
                f"package is only supported on CUDA 11.6 and above.  "
                "Note: make sure nvcc has a supported version by running nvcc -V."
            )

    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_70,code=sm_70")
    cc_flag.append("-gencode")
    cc_flag.append("arch=compute_80,code=sm_80")
    if bare_metal_version >= Version("11.8"):
        cc_flag.append("-gencode")
        cc_flag.append("arch=compute_90,code=sm_90")

    # HACK: The compiler flag -D_GLIBCXX_USE_CXX11_ABI is set to be the same as
    # torch._C._GLIBCXX_USE_CXX11_ABI
    # https://github.com/pytorch/pytorch/blob/8472c24e3b5b60150096486616d98b7bea01500b/torch/utils/cpp_extension.py#L920
    if FORCE_CXX11_ABI:
        torch._C._GLIBCXX_USE_CXX11_ABI = True

    ext_modules.append(
        CUDAExtension(
            name="selective_scan_cuda_core",
            sources=[
                "csrc/selective_scan/selective_scan.cpp",
                "csrc/selective_scan/selective_scan_core.cu",
                "csrc/selective_scan/selective_scan_core_fwd2.cu",
                "csrc/selective_scan/selective_scan_core_fwd3.cu",
                "csrc/selective_scan/selective_scan_core_fwd4.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                            "-O3",
                            "-std=c++17",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT16_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-U__CUDA_NO_BFLOAT162_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT162_CONVERSIONS__",
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                            "--ptxas-options=-v",
                            "-lineinfo",
                        ]
                        + cc_flag
                        + ["--threads", "4"],
            },
            include_dirs=[Path(this_dir) / "csrc" / "selective_scan"],
        )
    )

    return ext_modules


ext_modules = get_ext()
setup(
    name="selective_scan",
    version="0.0.1",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "include",
            "tests",
            "dist",
            "docs",
            "benchmarks",
        )
    ),
    author="Tri Dao, Albert Gu, Mzreo",
    author_email="tri@tridao.me, agu@cs.cmu.edu, liuyue171@mails.ucas.ac.cn",
    description="selective scan",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/state-spaces/mamba",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: Unix",
    ],
    ext_modules=ext_modules,
    cmdclass={"bdist_wheel": _bdist_wheel, "build_ext": BuildExtension} if ext_modules else {"bdist_wheel": _bdist_wheel,},
    python_requires=">=3.7",
    install_requires=[
        "torch",
        "packaging",
        "ninja",
        "einops",
    ],
)
