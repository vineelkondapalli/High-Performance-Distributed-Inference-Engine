#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Forward declarations of the per-dtype launchers defined in rmsnorm_kernel.cu.
// Keeping these as separate non-template C functions avoids having to expose
// CUDA template instantiation machinery to the g++-compiled .cpp file.
void rmsnorm_cuda_forward_fp32(
    const float* input, const float* weight, float* output,
    int n_rows, int hidden_dim, float eps, cudaStream_t stream);

void rmsnorm_cuda_forward_fp16(
    const __half* input, const __half* weight, __half* output,
    int n_rows, int hidden_dim, float eps, cudaStream_t stream);

void rmsnorm_cuda_forward_bf16(
    const __nv_bfloat16* input, const __nv_bfloat16* weight, __nv_bfloat16* output,
    int n_rows, int hidden_dim, float eps, cudaStream_t stream);

// ─── Python-facing forward function ──────────────────────────────────────────

torch::Tensor rmsnorm_forward(
    torch::Tensor input,   // [batch, seq_len, hidden_dim]
    torch::Tensor weight,  // [hidden_dim]
    double eps)
{
    // Input validation — fail early with a clear message rather than a CUDA error.
    TORCH_CHECK(input.dim()  == 3, "input must be 3-D, got ", input.dim(), "-D");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1-D, got ", weight.dim(), "-D");
    TORCH_CHECK(input.size(2) == weight.size(0),
        "hidden_dim mismatch: input.size(2)=", input.size(2),
        " vs weight.size(0)=", weight.size(0));
    TORCH_CHECK(input.device().is_cuda(),  "input must be a CUDA tensor");
    TORCH_CHECK(weight.device().is_cuda(), "weight must be a CUDA tensor");
    TORCH_CHECK(input.scalar_type() == weight.scalar_type(),
        "input and weight must have the same dtype");

    // Ensure contiguous layout (C-order, no strides).
    // .contiguous() is a no-op if the tensor is already contiguous.
    auto x = input.contiguous();
    auto w = weight.contiguous();

    const int n_rows     = x.size(0) * x.size(1);
    const int hidden_dim = x.size(2);
    const float eps_f    = static_cast<float>(eps);

    auto output = torch::empty_like(x);

    // Get the current CUDA stream so the kernel respects async execution
    // contexts (e.g., if the caller is using non-default streams).
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Dispatch to the correct dtype launcher.
    // AT_DISPATCH_FLOATING_TYPES_AND2 expands to an if/else chain over
    // float, double, half, bfloat16. We don't support double (no hardware
    // benefit on CUDA for RMSNorm), so we guard against it below.
    TORCH_CHECK(x.scalar_type() != torch::kDouble,
        "double precision not supported; use float32 or float16/bfloat16");

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "rmsnorm_forward",
        [&]() {
            if constexpr (std::is_same_v<scalar_t, float>) {
                rmsnorm_cuda_forward_fp32(
                    x.data_ptr<float>(),
                    w.data_ptr<float>(),
                    output.data_ptr<float>(),
                    n_rows, hidden_dim, eps_f, stream);
            } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
                rmsnorm_cuda_forward_fp16(
                    reinterpret_cast<const __half*>(x.data_ptr<at::Half>()),
                    reinterpret_cast<const __half*>(w.data_ptr<at::Half>()),
                    reinterpret_cast<      __half*>(output.data_ptr<at::Half>()),
                    n_rows, hidden_dim, eps_f, stream);
            } else {
                rmsnorm_cuda_forward_bf16(
                    reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>()),
                    reinterpret_cast<const __nv_bfloat16*>(w.data_ptr<at::BFloat16>()),
                    reinterpret_cast<      __nv_bfloat16*>(output.data_ptr<at::BFloat16>()),
                    n_rows, hidden_dim, eps_f, stream);
            }
        });

    return output;
}

// ─── Module registration ──────────────────────────────────────────────────────

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Fused RMSNorm CUDA kernel — hand-rolled, sm_75 optimized";
    m.def("rmsnorm_forward", &rmsnorm_forward,
          "RMSNorm forward pass (CUDA)",
          py::arg("input"), py::arg("weight"), py::arg("eps") = 1e-6);
}
