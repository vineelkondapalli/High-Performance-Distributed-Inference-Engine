# High-Performance Distributed Inference Engine

A C++20 async HTTP proxy paired with a Python LLM worker, connected over a Unix Domain Socket. The C++ sidecar handles all HTTP I/O, priority queuing, and Prometheus metrics — freeing the Python side to do pure inference. The worker runs TinyLlama-1.1B-Chat quantized to INT4 (NF4) via HuggingFace + bitsandbytes. Custom CUDA and Triton kernels are included and benchmarked against PyTorch baselines.

## Architecture

```
  HTTP Clients                C++ Sidecar (Boost.Beast)        Python Worker
  ─────────────►  POST /infer  ──────────────────────────►    UDS server
  GET /metrics                 Priority Queue (max-heap)        HuggingFace
  GET /health                  Prometheus /metrics              TinyLlama INT4
                               UDS client                       Triton RoPE kernel
                                      │                         CUDA RMSNorm kernel
                                      ▼
                            /tmp/inference.sock
```

## Key Components

| Component | Details |
|---|---|
| **IPC transport** | Unix Domain Socket with length-prefixed JSON — bypasses TCP stack entirely |
| **Priority queue** | Thread-safe `std::priority_queue` (max-heap); HTTP handlers push, single worker pops |
| **CUDA RMSNorm kernel** | Hand-rolled: `float4` vectorized loads, `__shfl_down_sync` warp reduction, 2.3–4.1× faster than `torch.rms_norm` |
| **Triton RoPE kernel** | Fuses HuggingFace's ~8-launch `apply_rotary_pos_emb` into a single Triton kernel |
| **INT4 quantization** | NF4 via bitsandbytes — TinyLlama-1.1B fits in ~700 MB VRAM |
| **Observability** | Prometheus metrics: request count, latency histogram, queue depth, queue wait time |
| **CI** | GitHub Actions builds and pushes Docker images to GHCR on every push to main |

## CUDA RMSNorm Kernel

Hand-rolled from scratch targeting RTX 2080 Ti (sm_75). Benchmarked against `torch.nn.functional.rms_norm` with CUDA events (10 warmup + 100 measured iterations):

| Shape (B, S, H)  | dtype | Kernel ms | Torch ms | Speedup | DRAM BW   | % of 616 GB/s peak |
|------------------|-------|-----------|----------|---------|-----------|---------------------|
| (1, 2048, 4096)  | fp32  | 0.155     | 0.434    | **2.80×** | 437 GB/s | 71%                |
| (1, 2048, 4096)  | fp16  | 0.068     | 0.275    | **4.05×** | 501 GB/s | 81%                |
| (4, 2048, 4096)  | fp32  | 0.633     | 1.702    | **2.69×** | 426 GB/s | 69%                |
| (1, 4096, 8192)  | fp32  | 0.733     | 1.701    | **2.32×** | 367 GB/s | 60%                |
| (1, 4096, 8192)  | fp16  | 0.345     | 0.909    | **2.63×** | 391 GB/s | 64%                |

Bandwidth measured via `torch.profiler` device timing. The kernel is memory-bound (arithmetic intensity ≈ 0.26 FLOP/byte vs roofline ridge at 21.8 FLOP/byte on this GPU). PyTorch's ATen path reaches only ~25% peak bandwidth because it dispatches through multiple smaller kernels with separate launch overhead.

Source: [python-worker/src/cuda_kernels/rmsnorm_kernel.cu](python-worker/src/cuda_kernels/rmsnorm_kernel.cu)  
Design notes: [python-worker/src/cuda_kernels/README_rmsnorm.md](python-worker/src/cuda_kernels/README_rmsnorm.md)

## Running Locally (no Docker needed)

**Requirements:** CUDA 12.x, PyTorch 2.5.1+cu124, conda or pip.

```bash
pip install -r python-worker/requirements.txt
```

**Terminal 1 — Python worker:**
```bash
cd python-worker
CUDA_VISIBLE_DEVICES=0 HF_HOME=./models/hf_cache \
  SOCKET_PATH=/tmp/inference.sock \
  python src/worker.py
# Ready when: Listening on /tmp/inference.sock  (~35s on first run)
```

**Terminal 2 — C++ sidecar** (requires Boost, CMake 3.20+):
```bash
cd cpp-sidecar
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel $(nproc)
SOCKET_PATH=/tmp/inference.sock HTTP_PORT=8080 ./build/proxy_server
```

**Test:**
```bash
curl http://localhost:8080/health

curl -s -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"id":"t1","prompt":"2+2=","max_tokens":10}' | python -m json.tool

curl http://localhost:8080/metrics
```

## Running with Docker Compose

```bash
docker compose up --build
# Requires Docker with GPU passthrough (nvidia-container-runtime or CDI)
```

## Benchmarking the RMSNorm Kernel

```bash
# Build the CUDA extension:
cd python-worker/src/cuda_kernels
python setup.py build_ext --inplace

# Run benchmark from repo root:
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_rmsnorm.py
```

## IPC Protocol

Length-prefixed JSON over Unix Domain Socket:

```
[4-byte little-endian uint32 = payload length][JSON bytes]
```

Request: `{"id": str, "prompt": str, "max_tokens": int, "priority": int}`  
Response: `{"id": str, "text": str, "tokens_generated": int, "error": null | str}`

## Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `http_requests_total` | Counter | Total HTTP requests received |
| `inference_latency_ms` | Histogram | End-to-end latency (buckets: 100, 500, 1000, 2000, 5000 ms) |
| `queue_depth` | Gauge | Requests currently waiting in priority queue |
| `queue_wait_ms` | Histogram | Time spent waiting before dispatch |

## Project Structure

```
.
├── cpp-sidecar/                    # C++20 async HTTP proxy
│   ├── include/                    # queue.hpp, metrics, uds_client, server
│   ├── src/                        # Implementations + main
│   └── CMakeLists.txt
├── python-worker/
│   ├── src/
│   │   ├── worker.py               # UDS server + HuggingFace inference
│   │   ├── rope_kernel.py          # Fused Triton RoPE kernel
│   │   └── cuda_kernels/
│   │       ├── rmsnorm_kernel.cu   # Hand-rolled CUDA RMSNorm (← start here)
│   │       ├── rmsnorm_binding.cpp # PyTorch C++ extension binding
│   │       ├── setup.py            # Build: sm_75, -O3, --use_fast_math
│   │       └── README_rmsnorm.md   # Design notes + full benchmark results
│   └── requirements.txt
├── scripts/
│   ├── benchmark_rmsnorm.py        # Correctness + perf vs torch baseline
│   ├── benchmark.py                # End-to-end latency/throughput
│   └── profile_rmsnorm.sh          # Nsight Compute profiling script
├── k8s/                            # Kubernetes manifests + Prometheus/Grafana
├── load-testing/                   # Locust scenarios
└── .github/workflows/deploy.yml    # CI: build + push to GHCR on push to main
```
