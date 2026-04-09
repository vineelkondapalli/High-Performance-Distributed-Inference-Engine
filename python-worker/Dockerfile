# CUDA 12.1 runtime — required for bitsandbytes INT4 and Triton kernels
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Python 3.11 + libgomp (required by PyTorch OpenMP threading)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3-pip \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

# Install dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/

# Environment defaults
ENV SOCKET_PATH=/tmp/inference.sock \
    HF_MODEL_ID=TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    HF_HOME=/models/hf_cache \
    N_CTX=2048 \
    TRITON_ROPE=0

CMD ["python", "src/worker.py"]
