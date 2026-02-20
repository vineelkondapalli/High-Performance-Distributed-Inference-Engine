"""
Python LLM Worker — Unix Domain Socket Server
=============================================
Listens on a Unix Domain Socket, receives length-prefixed JSON requests
from the C++ sidecar, runs inference with llama-cpp-python, and returns
length-prefixed JSON responses.

IPC Protocol (shared with C++ uds_client.cpp):
  [4 bytes little-endian uint32 = payload length][JSON bytes]

Request JSON:
  {"id": str, "prompt": str, "max_tokens": int, "priority": int}

Response JSON:
  {"id": str, "text": str, "tokens_generated": int, "error": null | str}
"""

import json
import logging
import os
import signal
import socket
import struct
import sys

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")

SOCKET_PATH = os.getenv("SOCKET_PATH", "/tmp/inference.sock")
MODEL_PATH  = os.getenv("MODEL_PATH",  "/models/tinyllama-1.1b-q4.gguf")
N_CTX       = int(os.getenv("N_CTX",   "2048"))
N_THREADS   = int(os.getenv("N_THREADS", str(os.cpu_count() or 4)))


# ── IPC helpers ──────────────────────────────────────────────────────────────

def recv_message(conn: socket.socket) -> dict:
    """Read a length-prefixed JSON message from the socket."""
    raw_len = _recv_exact(conn, 4)
    length = struct.unpack("<I", raw_len)[0]
    if length == 0 or length > 10 * 1024 * 1024:
        raise ValueError(f"Invalid message length: {length}")
    payload = _recv_exact(conn, length)
    return json.loads(payload)


def send_message(conn: socket.socket, data: dict) -> None:
    """Write a length-prefixed JSON message to the socket."""
    payload = json.dumps(data).encode("utf-8")
    header  = struct.pack("<I", len(payload))
    conn.sendall(header + payload)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    """Read exactly n bytes, raising on EOF."""
    buf = b""
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise EOFError("Connection closed before all bytes received.")
        buf += chunk
    return buf


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """Load TinyLlama-1.1B (4-bit quantized) into memory."""
    try:
        from llama_cpp import Llama
    except ImportError:
        log.error("llama-cpp-python is not installed. Run: pip install llama-cpp-python")
        sys.exit(1)

    if not os.path.exists(MODEL_PATH):
        log.error("Model file not found: %s", MODEL_PATH)
        log.error("Run: bash scripts/download_model.sh")
        sys.exit(1)

    log.info("Loading model: %s", MODEL_PATH)
    log.info("Context length: %d tokens | Threads: %d", N_CTX, N_THREADS)

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        verbose=False,
    )
    log.info("Model loaded successfully.")
    return llm


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(llm, request: dict) -> dict:
    """Run a single inference request and return a response dict."""
    prompt     = request["prompt"]
    max_tokens = request.get("max_tokens", 128)
    req_id     = request.get("id", "unknown")

    try:
        output = llm(
            prompt,
            max_tokens=max_tokens,
            stop=["</s>", "\n\n"],   # TinyLlama stop tokens
            echo=False,
        )
        text             = output["choices"][0]["text"]
        tokens_generated = output["usage"]["completion_tokens"]

        log.info("Request %s: generated %d tokens", req_id, tokens_generated)
        return {
            "id":               req_id,
            "text":             text,
            "tokens_generated": tokens_generated,
            "error":            None,
        }

    except Exception as exc:  # noqa: BLE001
        log.exception("Inference failed for request %s", req_id)
        return {
            "id":               req_id,
            "text":             "",
            "tokens_generated": 0,
            "error":            str(exc),
        }


# ── Socket server ─────────────────────────────────────────────────────────────

def run_server(llm) -> None:
    """Accept connections in a loop, handle one request per connection."""
    # Remove stale socket file if it exists.
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)

    # Handle SIGTERM gracefully (sent by Docker/k8s on shutdown).
    running = True
    def _handle_sigterm(sig, frame):
        nonlocal running
        log.info("SIGTERM received — shutting down.")
        running = False

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT,  _handle_sigterm)

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as srv:
        srv.bind(SOCKET_PATH)
        # Set permissions so only the container user can connect.
        os.chmod(SOCKET_PATH, 0o600)
        srv.listen(1)  # backlog=1: intentional — one request at a time
        srv.settimeout(1.0)  # allow checking `running` flag periodically

        log.info("Listening on %s", SOCKET_PATH)

        while running:
            try:
                conn, _ = srv.accept()
            except socket.timeout:
                continue  # loop back and check `running`

            with conn:
                try:
                    request  = recv_message(conn)
                    response = run_inference(llm, request)
                    send_message(conn, response)
                except Exception as exc:  # noqa: BLE001
                    log.exception("Error handling connection: %s", exc)

    # Clean up socket file on exit.
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    log.info("Worker exited cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    llm = load_model()
    run_server(llm)
