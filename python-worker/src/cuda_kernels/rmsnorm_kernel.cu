/*
 * Fused RMSNorm CUDA Kernel
 * =========================
 * Formula: output[i] = input[i] * weight[i] / sqrt(mean(input²) + eps)
 *
 * Why this kernel exists
 * ----------------------
 * PyTorch's built-in rms_norm dispatches through ATen, which adds Python/C++
 * dispatch overhead and may not vectorize loads optimally for every shape.
 * A hand-rolled kernel lets us control thread tiling, vectorized loads (float4),
 * and the reduction strategy explicitly.
 *
 * Why this kernel is MEMORY-BOUND
 * --------------------------------
 * For shape (1, 2048, 4096) fp32:
 *   - DRAM traffic: 2 reads + 1 write = 3 × (2048×4096×4 bytes) ≈ 96 MB
 *   - Arithmetic: ≈ 3 FLOP per element (square, add-to-sum, multiply-normalize)
 *     = 3 × 2048×4096 ≈ 25 MFLOP
 *   - Arithmetic intensity: 25M / 96M ≈ 0.26 FLOP/byte
 *   - RTX 2080 Ti roofline ridge point: 13.4 TFLOP / 616 GB/s ≈ 21.8 FLOP/byte
 *   - 0.26 << 21.8 → deeply memory-bound; peak throughput is DRAM bandwidth.
 *
 * Why 256 threads per block
 * -------------------------
 * RTX 2080 Ti (Turing sm_75): 1024 max threads per SM.
 * At 256 threads/block → 4 blocks resident per SM → 100% thread occupancy.
 * At 512 threads/block → 2 blocks per SM → 50% occupancy.
 * 256 also divides evenly into hidden_dim 4096 (16 elements/thread) and
 * 8192 (32 elements/thread), keeping all threads busy with no idle lanes.
 *
 * Why shared memory for inter-warp reduction
 * ------------------------------------------
 * __shfl_down_sync exchanges register values within a single warp (32 threads).
 * After each warp computes its partial sum-of-squares, those partial sums live
 * in lane-0 of each warp — in *registers*, which are private to each thread.
 * There is no instruction that can read another warp's registers directly.
 * Shared memory is the only per-block fast storage visible to all threads after
 * a __syncthreads() barrier, so we route warp partial sums through it.
 *
 * Next optimization steps (if more time)
 * ----------------------------------------
 * 1. Persistent kernel: launch exactly 68 blocks (one per SM), loop over rows
 *    with a grid-stride loop. Eliminates per-row kernel launch overhead (~5µs).
 * 2. CUDA Graph capture: amortize Python-side launch cost across many calls.
 * 3. Single-pass with register caching: for small hidden_dim (≤ 512) all
 *    elements fit in registers; normalize in one pass by caching values.
 * 4. Fused kernel: combine RMSNorm with the following linear (matmul) layer
 *    to eliminate the write-back of normalized values to DRAM.
 * 5. fp8 support (E4M3/E5M2) for inference quantization pipelines.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ─── Type conversion helpers ─────────────────────────────────────────────────
//
// PyTorch's build system passes -D__CUDA_NO_HALF_CONVERSIONS__, which disables
// C-style (float)__half and (float)__nv_bfloat16 casts. Use these explicit
// intrinsics instead, which always work regardless of that flag.

template<typename T> __device__ __forceinline__ float   to_float(T x);
template<typename T> __device__ __forceinline__ T       from_float(float x);

template<> __device__ __forceinline__ float   to_float<float>(float x)               { return x; }
template<> __device__ __forceinline__ float   to_float<__half>(__half x)              { return __half2float(x); }
template<> __device__ __forceinline__ float   to_float<__nv_bfloat16>(__nv_bfloat16 x){ return __bfloat162float(x); }

template<> __device__ __forceinline__ float          from_float<float>(float x)          { return x; }
template<> __device__ __forceinline__ __half         from_float<__half>(float x)          { return __float2half(x); }
template<> __device__ __forceinline__ __nv_bfloat16  from_float<__nv_bfloat16>(float x)  { return __float2bfloat16(x); }

// ─── Constants ───────────────────────────────────────────────────────────────

static constexpr int BLOCK_SIZE     = 256;          // threads per block
static constexpr int WARPS_PER_BLOCK = BLOCK_SIZE / 32;  // = 8

// ─── Warp-level reduction ─────────────────────────────────────────────────────

/*
 * __shfl_down_sync(mask, val, offset):
 *   Each thread in the warp reads the value from the thread whose lane_id is
 *   (my_lane + offset). The mask 0xffffffff means all 32 lanes participate.
 *   After 5 steps (offset = 16, 8, 4, 2, 1) lane 0 holds the sum of all
 *   32 lanes' values. Other lanes hold garbage — only lane 0's result is used.
 *
 *   This is a pure register operation: no shared memory, no barrier, one cycle
 *   per step (vs. ~100 cycles for a shared-memory round-trip with __syncthreads).
 */
__device__ __forceinline__ float warp_reduce_sum(float val) {
    // Butterfly reduction: each step halves the active distance.
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val,  8);
    val += __shfl_down_sync(0xffffffff, val,  4);
    val += __shfl_down_sync(0xffffffff, val,  2);
    val += __shfl_down_sync(0xffffffff, val,  1);
    return val;  // only lane 0's return value is meaningful
}

// ─── Main kernel ──────────────────────────────────────────────────────────────

template <typename scalar_t>
__global__ void rmsnorm_kernel(
    const scalar_t* __restrict__ input,   // [n_rows, hidden_dim]
    const scalar_t* __restrict__ weight,  // [hidden_dim]
    scalar_t*       __restrict__ output,  // [n_rows, hidden_dim]
    const int   hidden_dim,
    const float eps)
{
    // One block handles one row.
    const int row_idx = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const scalar_t* row_in  = input  + row_idx * hidden_dim;
          scalar_t* row_out = output + row_idx * hidden_dim;

    // Shared memory: one float per warp to hold warp partial sums,
    // then reused to broadcast inv_rms to all threads.
    // 8 warps × 4 bytes = 32 bytes per block — negligible.
    __shared__ float smem[WARPS_PER_BLOCK];

    // ── Phase 1: accumulate sum-of-squares in a register ─────────────────────
    //
    // Each thread strides across the row in steps of BLOCK_SIZE, using
    // vectorized float4 loads (128-bit, 4 floats at once) when possible.
    // Vectorized loads improve memory bandwidth utilization because the GPU
    // can issue fewer, wider transactions to satisfy the same byte count.
    //
    // All accumulation is in float regardless of scalar_t, to maintain
    // numerical precision for fp16/bf16 inputs (fp16 max ≈ 65504;
    // sum-of-squares for hidden_dim=4096 with unit-variance inputs ≈ 4096,
    // which overflows fp16 easily).
    float sum_sq = 0.0f;

    if (hidden_dim % 4 == 0) {
        // Vectorized path: each thread processes 4 elements per iteration.
        // Thread tid handles elements: tid*4, tid*4 + BLOCK_SIZE*4, ...
        const float4* row_vec = reinterpret_cast<const float4*>(row_in);
        const int vec_len = hidden_dim / 4;
        for (int i = tid; i < vec_len; i += BLOCK_SIZE) {
            float4 v;
            if constexpr (sizeof(scalar_t) == 4) {
                // fp32: direct float4 load
                v = row_vec[i];
            } else {
                // fp16/bf16: unpack to float using explicit intrinsics.
                // C-style (float)__half is disabled by -D__CUDA_NO_HALF_CONVERSIONS__.
                const scalar_t* base = row_in + i * 4;
                v.x = to_float(base[0]);
                v.y = to_float(base[1]);
                v.z = to_float(base[2]);
                v.w = to_float(base[3]);
            }
            sum_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        // Scalar tail path for hidden_dim not divisible by 4.
        // Never reached for 4096 or 8192, but makes the kernel correct
        // for arbitrary hidden_dim (e.g., hidden_dim=2048 for small models).
        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            float v = to_float(row_in[i]);
            sum_sq += v * v;
        }
    }

    // ── Phase 2: warp-level reduction ─────────────────────────────────────────
    // Each warp reduces its 32 partial sums to lane 0 using register shuffles.
    sum_sq = warp_reduce_sum(sum_sq);

    // ── Phase 3: inter-warp reduction via shared memory ───────────────────────
    // Lane 0 of each warp writes its warp total to smem[warp_id].
    if (lane_id == 0) {
        smem[warp_id] = sum_sq;
    }
    __syncthreads();  // ensure all warp leaders have written before anyone reads

    // The first warp loads all 8 warp totals and reduces them.
    // Threads 0–7 each load one entry; threads 8–31 load zero (they won't
    // contribute to the final result, but they must participate in the shuffle).
    if (warp_id == 0) {
        float val = (lane_id < WARPS_PER_BLOCK) ? smem[lane_id] : 0.0f;
        // 3 steps suffice to reduce 8 values (WARPS_PER_BLOCK = 8):
        val += __shfl_down_sync(0xffffffff, val, 4);
        val += __shfl_down_sync(0xffffffff, val, 2);
        val += __shfl_down_sync(0xffffffff, val, 1);
        // Lane 0 now holds the total sum-of-squares for the entire row.
        if (lane_id == 0) {
            // Compute inverse RMS and broadcast via smem[0].
            // rsqrtf is a single hardware instruction on CUDA (MUFU.RSQ).
            smem[0] = rsqrtf(val / (float)hidden_dim + eps);
        }
    }
    __syncthreads();  // ensure inv_rms is visible to all threads

    const float inv_rms = smem[0];

    // ── Phase 4: normalize and write ──────────────────────────────────────────
    // Second DRAM read of input (unavoidable without caching in registers/smem).
    // Weight vector is read once per block; at 16KB for hidden_dim=4096 fp32,
    // it typically fits in L2 cache and is amortized across many row-blocks.
    if (hidden_dim % 4 == 0) {
        const int vec_len = hidden_dim / 4;
        for (int i = tid; i < vec_len; i += BLOCK_SIZE) {
            if constexpr (sizeof(scalar_t) == 4) {
                float4 in_v  = reinterpret_cast<const float4*>(row_in)[i];
                float4 wt_v  = reinterpret_cast<const float4*>(weight)[i];
                float4 out_v;
                out_v.x = in_v.x * inv_rms * wt_v.x;
                out_v.y = in_v.y * inv_rms * wt_v.y;
                out_v.z = in_v.z * inv_rms * wt_v.z;
                out_v.w = in_v.w * inv_rms * wt_v.w;
                reinterpret_cast<float4*>(row_out)[i] = out_v;
            } else {
                const scalar_t* base_in = row_in    + i * 4;
                const scalar_t* base_wt = weight    + i * 4;
                      scalar_t* base_out= row_out   + i * 4;
                base_out[0] = from_float<scalar_t>(to_float(base_in[0]) * inv_rms * to_float(base_wt[0]));
                base_out[1] = from_float<scalar_t>(to_float(base_in[1]) * inv_rms * to_float(base_wt[1]));
                base_out[2] = from_float<scalar_t>(to_float(base_in[2]) * inv_rms * to_float(base_wt[2]));
                base_out[3] = from_float<scalar_t>(to_float(base_in[3]) * inv_rms * to_float(base_wt[3]));
            }
        }
    } else {
        for (int i = tid; i < hidden_dim; i += BLOCK_SIZE) {
            row_out[i] = from_float<scalar_t>(to_float(row_in[i]) * inv_rms * to_float(weight[i]));
        }
    }
}

// ─── C++ launcher (called from rmsnorm_binding.cpp) ──────────────────────────
//
// Keeping the launcher in the .cu file means nvcc handles template
// instantiation. The .cpp binding only needs an extern declaration.

void rmsnorm_cuda_forward_fp32(
    const float* input, const float* weight, float* output,
    int n_rows, int hidden_dim, float eps, cudaStream_t stream)
{
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    const int smem_bytes = WARPS_PER_BLOCK * sizeof(float);
    rmsnorm_kernel<float><<<grid, block, smem_bytes, stream>>>(
        input, weight, output, hidden_dim, eps);
}

void rmsnorm_cuda_forward_fp16(
    const __half* input, const __half* weight, __half* output,
    int n_rows, int hidden_dim, float eps, cudaStream_t stream)
{
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    const int smem_bytes = WARPS_PER_BLOCK * sizeof(float);
    rmsnorm_kernel<__half><<<grid, block, smem_bytes, stream>>>(
        input, weight, output, hidden_dim, eps);
}

void rmsnorm_cuda_forward_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    int n_rows, int hidden_dim, float eps, cudaStream_t stream)
{
    dim3 grid(n_rows);
    dim3 block(BLOCK_SIZE);
    const int smem_bytes = WARPS_PER_BLOCK * sizeof(float);
    rmsnorm_kernel<__nv_bfloat16><<<grid, block, smem_bytes, stream>>>(
        input, weight, output, hidden_dim, eps);
}
