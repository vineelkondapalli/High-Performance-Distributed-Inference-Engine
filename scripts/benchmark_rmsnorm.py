"""
RMSNorm CUDA kernel benchmark.

Compares hand-rolled kernel against torch.nn.functional.rms_norm across
multiple shapes and dtypes, using CUDA events for accurate GPU timing.

Usage (from repo root):
    cd python-worker/src/cuda_kernels
    python setup.py build_ext --inplace   # build .so once
    cd ../../..
    CUDA_VISIBLE_DEVICES=8 python scripts/benchmark_rmsnorm.py

Exit code 1 if any correctness check fails.
"""

import os
import sys
import glob
import importlib.util
import torch

KERNEL_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "python-worker", "src", "cuda_kernels"
)

# Find the pre-built .so (built by setup.py build_ext --inplace in kernel dir)
so_pattern = os.path.join(KERNEL_DIR, "rmsnorm_cuda*.so")
so_files = glob.glob(so_pattern)
if not so_files:
    print(
        "ERROR: rmsnorm_cuda*.so not found in", KERNEL_DIR,
        "\nBuild it first:\n"
        "  cd python-worker/src/cuda_kernels\n"
        "  python setup.py build_ext --inplace"
    )
    sys.exit(1)

so_path = so_files[0]
print(f"Loading pre-built extension: {os.path.basename(so_path)}")
spec = importlib.util.spec_from_file_location("rmsnorm_cuda", so_path)
rmsnorm_cuda = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rmsnorm_cuda)
print("Extension loaded.\n")

WARMUP   = 10
MEASURED = 100
EPS      = 1e-6

# (batch, seq_len, hidden_dim)
SHAPES = [
    (1, 2048, 4096),
    (4, 2048, 4096),
    (1, 4096, 8192),
]
DTYPES = [torch.float32, torch.float16]

# Correctness tolerances (max absolute error).
# fp16 tolerance is 2e-2: at hidden_dim=8192, accumulated rounding in fp16
# sum-of-squares can produce ~1 ULP error in the final normalized value, which
# reaches ~0.015 at that scale. torch.rms_norm also accumulates in fp16 here.
ABS_TOL = {torch.float32: 1e-4, torch.float16: 2e-2}

all_passed = True
rows = []

for shape in SHAPES:
    for dtype in DTYPES:
        B, S, H = shape
        x = torch.randn(B, S, H, dtype=dtype, device="cuda")
        w = torch.randn(H,       dtype=dtype, device="cuda")

        # ── Correctness ──────────────────────────────────────────────────────
        ref = torch.nn.functional.rms_norm(x, (H,), w, eps=EPS)
        out = rmsnorm_cuda.rmsnorm_forward(x, w, EPS)
        max_err = (ref.float() - out.float()).abs().max().item()
        tol     = ABS_TOL[dtype]
        passed  = max_err < tol
        if not passed:
            all_passed = False
            print(f"FAIL  shape={shape} dtype={dtype} max_err={max_err:.6f} > tol={tol}")

        # ── Timing: custom kernel ────────────────────────────────────────────
        for _ in range(WARMUP):
            rmsnorm_cuda.rmsnorm_forward(x, w, EPS)
        torch.cuda.synchronize()

        t_start = torch.cuda.Event(enable_timing=True)
        t_end   = torch.cuda.Event(enable_timing=True)
        t_start.record()
        for _ in range(MEASURED):
            rmsnorm_cuda.rmsnorm_forward(x, w, EPS)
        t_end.record()
        torch.cuda.synchronize()
        kernel_ms = t_start.elapsed_time(t_end) / MEASURED

        # ── Timing: torch reference ──────────────────────────────────────────
        for _ in range(WARMUP):
            torch.nn.functional.rms_norm(x, (H,), w, eps=EPS)
        torch.cuda.synchronize()

        t_start2 = torch.cuda.Event(enable_timing=True)
        t_end2   = torch.cuda.Event(enable_timing=True)
        t_start2.record()
        for _ in range(MEASURED):
            torch.nn.functional.rms_norm(x, (H,), w, eps=EPS)
        t_end2.record()
        torch.cuda.synchronize()
        torch_ms = t_start2.elapsed_time(t_end2) / MEASURED

        speedup = torch_ms / kernel_ms
        dtype_str = "fp32" if dtype == torch.float32 else "fp16"
        status = "PASS" if passed else "FAIL"
        rows.append((shape, dtype_str, kernel_ms, torch_ms, speedup, max_err, status))

# ── Print table ──────────────────────────────────────────────────────────────
header = f"{'Shape':<22} {'dtype':<6} {'Kernel ms':>10} {'Torch ms':>10} {'Speedup':>8} {'MaxErr':>10} {'Status'}"
print(header)
print("-" * len(header))
for shape, dtype_str, kms, tms, spd, err, status in rows:
    shape_str = f"({shape[0]},{shape[1]},{shape[2]})"
    print(f"{shape_str:<22} {dtype_str:<6} {kms:>10.4f} {tms:>10.4f} {spd:>8.2f}x {err:>10.2e}  {status}")

print()
if not all_passed:
    print("SOME CORRECTNESS CHECKS FAILED.")
    sys.exit(1)
else:
    print("All correctness checks passed.")
