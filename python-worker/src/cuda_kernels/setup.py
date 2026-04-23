"""
Build script for the RMSNorm CUDA extension.

Usage (from this directory):
    pip install -e .                              # editable install
    python setup.py build_ext --inplace           # build .so in-place (dev)

For development without installing, use torch.utils.cpp_extension.load()
directly in benchmark_rmsnorm.py — no setup.py needed.
"""

import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

HERE = os.path.dirname(os.path.abspath(__file__))

if not os.environ.get("CUDA_HOME"):
    print(
        "WARNING: CUDA_HOME is not set. "
        "If nvcc is in your PATH (e.g. via conda install cuda-nvcc), "
        "torch will find it automatically. Otherwise set CUDA_HOME explicitly."
    )

ext = CUDAExtension(
    name="rmsnorm_cuda",
    sources=[
        os.path.join(HERE, "rmsnorm_kernel.cu"),
        os.path.join(HERE, "rmsnorm_binding.cpp"),
    ],
    extra_cuda_cflags=[
        "-arch=sm_75",       # RTX 2080 Ti (Turing)
        "-O3",
        "--use_fast_math",   # enables fused multiply-add (FMAD) and fast rsqrtf
        "-lineinfo",         # embeds source line info for ncu source correlation
    ],
    extra_cflags=["-O3", "-std=c++17"],
)

setup(
    name="rmsnorm_cuda",
    version="0.1.0",
    description="Hand-rolled fused RMSNorm CUDA kernel (sm_75)",
    ext_modules=[ext],
    cmdclass={"build_ext": BuildExtension},
)
