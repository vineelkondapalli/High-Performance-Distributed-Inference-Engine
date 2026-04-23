# Project: Distributed LLM Inference Engine

## What this is
C++ HTTP proxy (Boost.Beast) + Python inference worker connected via Unix Domain Socket.
C++ handles HTTP and queuing. Python runs the model.

## Hardware (Linux server)
- 10x RTX 2080 Ti (11 GB each); **GPU 8 assigned to this project** via `CUDA_VISIBLE_DEVICES=8`
- GPU 8 has ~10.8 GB free — TinyLlama INT4 fits on a single card
- Inside the container GPU 8 is remapped to index 0; use `device_map="cuda:0"` (not "auto")

## Stack
- C++20, Boost.Beast + Boost.Asio, Docker-only build
- Python: HuggingFace transformers 4.40.2 + bitsandbytes INT4 NF4 + Triton 3.1
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (downloaded to `./models/hf_cache`)
- CUDA 12.6 (host driver 560.35.05); PyTorch 2.5.1+cu124
- Conda env `inference-engine` (Python 3.11) for running scripts directly on host

## IPC protocol
Length-prefixed JSON over UDS. **Do not change this — C++ side depends on it.**
```
[4-byte LE uint32 = length][JSON bytes]
```
Request:  `{"id", "prompt", "max_tokens", "priority"}`
Response: `{"id", "text", "tokens_generated", "error"}`

## Key files
| File | Purpose |
|------|---------|
| `python-worker/src/worker.py` | UDS server. Lines 41–66 (IPC helpers) and 135–178 (server loop): do not touch. |
| `python-worker/src/rope_kernel.py` | Triton RoPE kernel + wrapper + tests |
| `python-worker/src/profile_inference.py` | torch.profiler script |
| `cpp-sidecar/src/server.cpp` | Boost.Beast HTTP server |
| `cpp-sidecar/include/queue.hpp` | Priority queue (producer-consumer) |
| `docker-compose.yml` | Local dev. GPU runtime enabled. |
| `scripts/benchmark.py` | Latency + per-token stats, A/B comparison |
| `scripts/download_hf_model.py` | Pre-download model before first docker build |

## Env vars (python-worker)
| Var | Default | Notes |
|-----|---------|-------|
| `SOCKET_PATH` | `/tmp/inference.sock` | |
| `HF_MODEL_ID` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | |
| `HF_HOME` | `/models/hf_cache` | persisted in `./models` volume |
| `TRITON_ROPE` | `0` | set `1` to enable fused Triton RoPE kernel |

## Triton RoPE kernel
- Fuses HF's `apply_rotary_pos_emb` (~8 CUDA launches) into 1 launch
- Monkey-patches `transformers.models.llama.modeling_llama.apply_rotary_pos_emb`
- Patch applied before `from_pretrained()`, soft fallback if target not found
- `head_dim` must be power-of-2 (TinyLlama = 64, fine)
- Run tests before integrating: `python src/rope_kernel.py`

## Startup order
1. `python scripts/download_hf_model.py` — do once before first build
2. `docker compose up --build`
3. python-worker starts first; cpp-sidecar waits for socket healthcheck

## Running locally (no Docker needed)

**Conda env:** `/data/vineel/conda-envs/inference-engine` (Python 3.11, all deps installed)

**C++ sidecar binary:** `cpp-sidecar/build/proxy_server` (pre-built, rebuild with step below)

### Terminal 1 — Python worker
```bash
conda activate /data/vineel/conda-envs/inference-engine
cd python-worker
CUDA_VISIBLE_DEVICES=8 HF_HOME=/mnt/data/shared/vineel/hf_cache \
  SOCKET_PATH=/tmp/inference.sock TRITON_ROPE=0 \
  python src/worker.py
# Ready when you see: Listening on /tmp/inference.sock (~35s, model already cached)
```

### Terminal 2 — C++ sidecar
```bash
export LD_LIBRARY_PATH="/data/vineel/conda-envs/inference-engine/lib:$LD_LIBRARY_PATH"
SOCKET_PATH=/tmp/inference.sock HTTP_PORT=8080 NUM_WORKERS=4 IO_THREADS=2 \
  ./cpp-sidecar/build/proxy_server
```

### Test it
```bash
# Health
curl http://localhost:8080/health

# Inference
curl -s -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"id":"t1","prompt":"2+2=","max_tokens":10}' | python -m json.tool

# Prometheus metrics
curl http://localhost:8080/metrics

# Kernel correctness tests
cd python-worker && conda run -n inference-engine python src/rope_kernel.py

# Benchmark (A/B)
conda run -n inference-engine python scripts/benchmark.py --requests 15 --max-tokens 50 --compare
```

### Rebuild C++ sidecar (after source changes)
```bash
conda activate /data/vineel/conda-envs/inference-engine
cd cpp-sidecar
cmake -B build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/data/vineel/conda-envs/inference-engine
cmake --build build --parallel $(nproc)
```

## Rules
- Never change the IPC protocol or UDS server loop
- C++ sidecar does not need GPU access
- `transformers` pinned at `4.40.2` — do not upgrade without checking patch target still exists
- `accelerate` pinned at `0.34.2` — accelerate 1.x breaks dispatch_model for bitsandbytes 4-bit models with transformers 4.40.2
- Prefer editing existing files over creating new ones
