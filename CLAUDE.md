# Project: Distributed LLM Inference Engine

## What this is
C++ HTTP proxy (Boost.Beast) + Python inference worker connected via Unix Domain Socket.
C++ handles HTTP and queuing. Python runs the model.

## Hardware (Linux server)
- 2x RTX 2080 Ti (22GB VRAM total)
- Use `device_map="auto"` — accelerate splits across both GPUs

## Stack
- C++20, Boost.Beast + Boost.Asio, Docker-only build
- Python: HuggingFace transformers 4.40.2 + bitsandbytes INT4 NF4 + Triton 2.3
- Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` (downloaded to `./models/hf_cache`)
- CUDA 12.1 (host driver must be >= 525)

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

## Common commands
```bash
# Verify GPU inside container
docker compose run --rm python-worker python -c \
  "import torch; print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"

# Test inference
curl -s -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"id":"t1","prompt":"2+2=","max_tokens":5}' | python -m json.tool

# Profile
docker compose run --rm python-worker python src/profile_inference.py \
  --warmup 2 --steps 5 --max-tokens 50

# Kernel correctness tests
docker compose run --rm python-worker python src/rope_kernel.py

# Benchmark (A/B)
python scripts/benchmark.py --requests 15 --max-tokens 50 --compare
```

## Rules
- Never change the IPC protocol or UDS server loop
- C++ sidecar does not need GPU access
- `transformers` pinned at `4.40.2` — do not upgrade without checking patch target still exists
- Prefer editing existing files over creating new ones
