# Hand-Rolled RMSNorm CUDA Kernel

A fused RMSNorm kernel written from scratch in CUDA C++, targeting the RTX 2080 Ti (sm_75 / Turing).

## Formula

```
output[i] = input[i] * weight[i] / sqrt(mean(input²) + eps)
```

Input shape: `(batch, seq_len, hidden_dim)`. One CUDA block per row.

## Design Decisions

### 256 threads per block
RTX 2080 Ti has 1024 max threads per SM. At 256 threads/block, 4 blocks are resident simultaneously → 100% thread occupancy. At 512, only 2 blocks fit. 256 also divides evenly into hidden_dim=4096 (16 elements/thread) and 8192 (32 elements/thread).

### Two DRAM reads instead of caching in shared memory
Caching one full row (4096 fp32 elements = 16 KB) in shared memory would limit the SM to 3 concurrent blocks (48 KB shared/SM ÷ 16 KB), which is worse for throughput than a second DRAM read. The kernel reads the row twice: once to accumulate sum-of-squares, once to write normalized output.

### Warp-level reduction with `__shfl_down_sync`
Within each warp (32 threads), 5 butterfly steps reduce 32 partial sums to lane 0 using register shuffles — no shared memory, no synchronization barrier, ~1 cycle per step vs ~100 cycles for a shared-memory round-trip. Only 8 floats (one per warp) touch shared memory for the inter-warp reduction phase.

### `float4` vectorized loads
When `hidden_dim % 4 == 0` (always true for 4096/8192), each thread loads 4 elements at once via a 128-bit transaction. This halves the number of memory instructions and improves DRAM bandwidth utilization.

### All accumulation in `float`
fp16/bf16 sum-of-squares overflows easily: for hidden_dim=4096 with unit-variance inputs, sum-of-squares ≈ 4096, but fp16 max ≈ 65504 — barely fits. For larger inputs or non-unit variance, overflow is certain. Accumulating in `float` throughout avoids this at negligible cost (the accumulation registers are already 32-bit).

### Why memory-bound
For shape (1, 2048, 4096) fp32:
- DRAM traffic: 2 reads + 1 write = 3 × (2048 × 4096 × 4 B) ≈ 96 MB
- Arithmetic: ~3 FLOP/element = ~25 MFLOP total
- Arithmetic intensity: 25M / 96M ≈ **0.26 FLOP/byte**
- RTX 2080 Ti roofline ridge point: 13.4 TFLOP ÷ 616 GB/s ≈ 21.8 FLOP/byte
- 0.26 ≪ 21.8 → **deeply memory-bound**; peak throughput equals peak DRAM bandwidth

## Benchmark Results

Hardware: RTX 2080 Ti (sm_75), 11 GB VRAM, ~616 GB/s peak DRAM BW  
PyTorch: 2.5.1+cu124 — reference is `torch.nn.functional.rms_norm`

**Timing** (CUDA events, 10 warmup + 100 measured iterations):

| Shape (B, S, H)    | dtype | Kernel ms | Torch ms | Speedup | Max Abs Err |
|--------------------|-------|-----------|----------|---------|-------------|
| (1, 2048, 4096)    | fp32  | 0.155     | 0.434    | **2.80×** | 1.91e-06  |
| (1, 2048, 4096)    | fp16  | 0.068     | 0.275    | **4.05×** | 7.81e-03  |
| (4, 2048, 4096)    | fp32  | 0.633     | 1.702    | **2.69×** | 2.86e-06  |
| (4, 2048, 4096)    | fp16  | 0.253     | 0.902    | **3.56×** | 1.56e-02  |
| (1, 4096, 8192)    | fp32  | 0.733     | 1.701    | **2.32×** | 1.91e-06  |
| (1, 4096, 8192)    | fp16  | 0.345     | 0.909    | **2.63×** | 1.56e-02  |

The kernel is **2.3–4.1× faster** than PyTorch's `rms_norm` across all tested shapes and dtypes.

**Bandwidth utilization** (measured via `torch.profiler`, CUDA device time; bytes = 2×input reads + 1×output write, weight assumed in L2):

| Shape (B, S, H)    | dtype | Kernel us | BW GB/s | % of 616 GB/s peak | Torch BW GB/s | Torch % peak |
|--------------------|-------|-----------|---------|---------------------|---------------|--------------|
| (1, 2048, 4096)    | fp32  | 153.5     | 437     | **71%**             | 156           | 25%          |
| (1, 2048, 4096)    | fp16  | 67.0      | 501     | **81%**             | 121           | 20%          |
| (4, 2048, 4096)    | fp32  | 630.3     | 426     | **69%**             | 159           | 26%          |
| (1, 4096, 8192)    | fp32  | 732.2     | 367     | **60%**             | 158           | 26%          |
| (1, 4096, 8192)    | fp16  | 343.1     | 391     | **64%**             | 148           | 24%          |

The custom kernel achieves **60–81% of peak DRAM bandwidth**. PyTorch's ATen path reaches only ~25% — it dispatches through multiple smaller kernels (separate reduce + elementwise), each with their own launch overhead and less coalesced access patterns. Our kernel fuses both passes and uses `float4` vectorized loads.

Note: Nsight Compute hardware counters (SM occupancy, L1/L2 cache hit rates, warp stall breakdown) require `RmProfilingAdminOnly=0`, which is not set on this shared server. The bandwidth figures above are derived from `torch.profiler` device timing + known byte counts. The roofline arithmetic intensity analysis below uses hardware spec math.

## Files

```
rmsnorm_kernel.cu       — CUDA kernel (template over float/half/bfloat16)
rmsnorm_binding.cpp     — PyTorch C++ extension binding (TORCH_CHECK, dispatch)
setup.py                — pip-installable build (sm_75, -O3, --use_fast_math)
```

## Building

```bash
conda activate /data/vineel/conda-envs/inference-engine
cd python-worker/src/cuda_kernels
python setup.py build_ext --inplace
```

## Running

```bash
# From repo root:
CUDA_VISIBLE_DEVICES=8 python scripts/benchmark_rmsnorm.py
```

## Possible Next Steps

1. **Persistent kernel**: launch exactly 68 blocks (one per SM on RTX 2080 Ti), loop over rows with a grid-stride loop — eliminates per-row kernel launch overhead.
2. **CUDA Graph capture**: amortize Python-side launch cost across many calls.
3. **Single-pass with register caching**: for small hidden_dim (≤512) all elements fit in registers; normalize in one pass, eliminating the second DRAM read.
4. **Fused kernel**: combine RMSNorm with the following linear (matmul) layer to eliminate writing normalized values to DRAM entirely.
5. **fp8 support**: E4M3/E5M2 for inference quantization pipelines.
