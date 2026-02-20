# ── Stage 1: Build ────────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS builder

# Install build toolchain + dependencies.
# ubuntu:24.04 ships GCC 13 and Boost 1.83 — no version mismatch.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        libboost-all-dev \
        nlohmann-json3-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source tree
COPY CMakeLists.txt .
COPY include/ include/
COPY src/     src/

# Configure and build (Release mode)
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/install \
    && cmake --build build --parallel $(nproc) \
    && cmake --install build

# ── Stage 2: Runtime ──────────────────────────────────────────────────────────
FROM ubuntu:24.04 AS runtime

# Runtime shared libraries — versions must match the builder stage.
# curl is included for the docker-compose healthcheck (GET /health).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libboost-system1.83.0 \
        libboost-thread1.83.0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy just the binary from the builder stage
COPY --from=builder /install/bin/proxy_server /usr/local/bin/proxy_server

# Environment defaults (overridden by docker-compose / k8s)
ENV HTTP_PORT=8080 \
    SOCKET_PATH=/tmp/inference.sock \
    NUM_WORKERS=4 \
    IO_THREADS=2

EXPOSE 8080

ENTRYPOINT ["/usr/local/bin/proxy_server"]
