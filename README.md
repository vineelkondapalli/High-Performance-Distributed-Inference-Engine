# High-Performance Distributed Inference Engine

A Kubernetes-native LLM inference system using the **Sidecar Design Pattern**: a C++20 async proxy insulates a Python/llama.cpp worker from high-concurrency traffic, mirroring production architectures used by Istio and KServe.

## Architecture

```
                        ┌─────────────────────────────────────────────┐
                        │              Kubernetes Pod                  │
                        │                                             │
  HTTP Clients          │  ┌─────────────────┐    Unix Domain Socket  │
  ──────────────────►   │  │  C++ Sidecar    │ ──────────────────►   │
  POST /infer           │  │  (Boost.Beast)  │  /tmp/inference.sock  │
  GET  /metrics         │  │                 │ ◄──────────────────   │
  GET  /health          │  │ • Async HTTP    │                        │
                        │  │ • Priority Queue│  ┌─────────────────┐  │
                        │  │ • Prometheus    │  │  Python Worker  │  │
  Prometheus ──────────►  │  │   /metrics     │  │ (llama-cpp-py)  │  │
                        │  └─────────────────┘  │                 │  │
                        │                        │ TinyLlama-1.1B  │  │
                        │                        │ 4-bit Quantized │  │
                        │                        └─────────────────┘  │
                        │                                             │
                        │         Shared emptyDir Volume (/tmp)       │
                        └─────────────────────────────────────────────┘
```

## Why This Architecture?

| Problem | Solution |
|---------|----------|
| Python GIL limits concurrency | C++ handles all HTTP I/O; Python does pure inference |
| TCP overhead for intra-pod comms | Unix Domain Sockets (kernel IPC, zero network stack) |
| Requests lost when model is busy | Thread-safe priority queue buffers up to N requests |
| No visibility into system behavior | Prometheus metrics + Grafana dashboards |

## Tech Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Orchestration | Kubernetes (Minikube) | Cloud-native standard |
| Proxy | C++20 + Boost.Beast/Asio | Async I/O, zero GIL |
| Model serving | Python + llama-cpp-python | Best LLM ecosystem |
| IPC | Unix Domain Sockets | OS-level IPC, no TCP overhead |
| Observability | Prometheus + Grafana | Industry standard MLOps |
| CI/CD | GitHub Actions → GHCR | Automated image builds |
| Load testing | Locust | Realistic concurrent load |

## Quickstart (Docker Compose)

### Prerequisites
- Docker Desktop (Windows/Mac/Linux)
- 4GB+ RAM available to Docker

### 1. Download the model
```bash
bash scripts/download_model.sh
# Downloads TinyLlama-1.1B-Chat Q4_K_M GGUF (~700MB) to ./models/
```

### 2. Build and run
```bash
docker compose up --build
```

### 3. Test
```bash
# Inference
curl -X POST http://localhost:8080/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Explain quantum computing in one sentence.", "max_tokens": 80}'

# Health check
curl http://localhost:8080/health

# Prometheus metrics
curl http://localhost:8080/metrics
```

## Quickstart (Kubernetes / Minikube)

```bash
# Start cluster
minikube start --memory=4096 --cpus=4

# Build images into minikube's Docker daemon
eval $(minikube docker-env)
docker build -t cpp-sidecar:latest ./cpp-sidecar
docker build -t python-worker:latest ./python-worker

# Deploy
kubectl apply -f k8s/

# Access
kubectl port-forward svc/inference-engine 8080:8080

# Deploy monitoring
kubectl apply -f k8s/monitoring/
kubectl port-forward svc/grafana 3000:3000  # admin/admin
```

## IPC Protocol

Containers communicate over a Unix Domain Socket using **length-prefixed JSON**:

```
┌──────────────────┬──────────────────────────────────────────────────┐
│  4 bytes (LE)    │  JSON payload                                    │
│  payload length  │                                                  │
└──────────────────┴──────────────────────────────────────────────────┘
```

**Request** (C++ → Python):
```json
{"id": "550e8400-e29b-41d4-a716-446655440000", "prompt": "Hello", "max_tokens": 128, "priority": 0}
```

**Response** (Python → C++):
```json
{"id": "550e8400-e29b-41d4-a716-446655440000", "text": "Hello! How can I help?", "tokens_generated": 7, "error": null}
```

## Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `http_requests_total` | Counter | Total HTTP requests received |
| `inference_latency_ms` | Histogram | End-to-end inference time (buckets: 100, 500, 1000, 2000, 5000ms) |
| `queue_depth` | Gauge | Current number of requests waiting in queue |
| `queue_wait_ms` | Histogram | Time requests spend waiting in queue |

## Load Testing

```bash
# Install locust
pip install locust

# Run load test (100 users, 10 spawn/sec)
locust -f load-testing/locustfile.py --host=http://localhost:8080 \
  --users 100 --spawn-rate 10 --run-time 60s --headless

# Or open the Locust web UI
locust -f load-testing/locustfile.py --host=http://localhost:8080
# → http://localhost:8089
```

## Project Structure

```
.
├── cpp-sidecar/           # C++20 async HTTP proxy
│   ├── include/           # Headers: queue, metrics, uds_client, server
│   ├── src/               # Implementations + main
│   ├── CMakeLists.txt     # C++20 build, Boost.Beast
│   └── Dockerfile         # Multi-stage: gcc:13 → debian:bookworm-slim
├── python-worker/         # Python LLM worker
│   ├── src/worker.py      # UDS server + llama-cpp-python
│   ├── requirements.txt
│   └── Dockerfile
├── k8s/                   # Kubernetes manifests
│   ├── deployment.yaml    # Pod with sidecar + shared volume
│   ├── service.yaml       # NodePort service
│   ├── configmap.yaml     # Environment configuration
│   └── monitoring/        # Prometheus + Grafana
├── load-testing/          # Locust scenarios
├── .github/workflows/     # CI/CD: build + push to GHCR
├── models/                # GGUF model files (gitignored)
├── scripts/               # download_model.sh, benchmark.py
└── docker-compose.yml     # Local development
```

## Benchmarks

Run `python scripts/benchmark.py` after a load test to generate comparison graphs.

Expected results (approximate, CPU-only):
| Setup | Throughput | p95 Latency |
|-------|-----------|-------------|
| Python direct (TCP) | ~1.2 req/s | ~8500ms |
| C++ Sidecar | ~1.2 req/s | ~8200ms |
| C++ Sidecar under burst | Queue absorbs spike | No dropped requests |

> Note: Throughput is bottlenecked by model inference time (CPU). The C++ proxy's advantage shows in **queue depth management** and **zero dropped connections** under burst traffic.
