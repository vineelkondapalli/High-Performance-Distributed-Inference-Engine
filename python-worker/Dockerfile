# python:3.11-slim — small base, no heavy ML frameworks needed (llama.cpp is self-contained)
FROM python:3.11-slim

# Build tools needed to compile llama-cpp-python's C extension
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/

# Environment defaults
ENV SOCKET_PATH=/tmp/inference.sock \
    MODEL_PATH=/models/tinyllama-1.1b-q4.gguf \
    N_CTX=2048

CMD ["python", "src/worker.py"]
