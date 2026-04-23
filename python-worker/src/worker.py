"""
Python LLM Worker — Unix Domain Socket Server
=============================================
Listens on a Unix Domain Socket, receives length-prefixed JSON requests
from the C++ sidecar, runs inference with HuggingFace transformers (INT4
bitsandbytes NF4), and returns length-prefixed JSON responses.

IPC Protocol (shared with C++ uds_client.cpp):
  [4 bytes little-endian uint32 = payload length][JSON bytes]

Request JSON:
  {"id": str, "prompt": str, "max_tokens": int, "priority": int}

Response JSON:
  {"id": str, "text": str, "tokens_generated": int, "error": null | str}

Environment variables:
  SOCKET_PATH   — UDS socket path (default: /tmp/inference.sock)
  HF_MODEL_ID   — HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  HF_HOME       — HuggingFace cache dir (default: /models/hf_cache)
  N_CTX         — max context length (default: 2048, informational only)
  TRITON_ROPE   — set to "1" to enable fused Triton RoPE kernel (default: 0)
"""

import json
import logging
import os
import signal
import socket
import struct
import sys
import torch

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("worker")

SOCKET_PATH = os.getenv("SOCKET_PATH", "/tmp/inference.sock")
HF_MODEL_ID = os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0")
N_CTX       = int(os.getenv("N_CTX", "2048"))
TRITON_ROPE = os.getenv("TRITON_ROPE", "0") == "1"


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
    buf = bytearray(n)
    view = memoryview(buf)
    pos = 0
    while pos < n:
        chunk = conn.recv_into(view[pos:], n - pos)
        if not chunk:
            raise EOFError("Connection closed before all bytes received.")
        pos += chunk
    return bytes(buf)


# ── Triton RoPE patch helpers ─────────────────────────────────────────────────

def _apply_triton_rope_patch() -> bool:
    """
    Monkey-patch HuggingFace's apply_rotary_pos_emb with our Triton kernel.

    The target is the module-level function in transformers.models.llama.modeling_llama.
    LlamaAttention.forward() calls it via LOAD_GLOBAL, so replacing it in the
    module dict before any forward pass redirects all subsequent calls.

    Returns True if the patch was applied, False if skipped (soft fallback).
    """
    try:
        import transformers.models.llama.modeling_llama as llama_module
    except ImportError:
        log.warning("[Triton RoPE] Could not import modeling_llama — skipping patch.")
        return False

    if not hasattr(llama_module, "apply_rotary_pos_emb"):
        log.warning(
            "[Triton RoPE] apply_rotary_pos_emb not found in modeling_llama "
            "(transformers version may have changed) — falling back to HF native RoPE."
        )
        return False

    try:
        from src.rope_kernel import apply_rope_triton
    except ImportError:
        log.warning("[Triton RoPE] rope_kernel.py not found — falling back to HF native RoPE.")
        return False

    original = llama_module.apply_rotary_pos_emb
    llama_module.apply_rotary_pos_emb = apply_rope_triton
    log.info("[Triton RoPE] Monkey-patch applied: apply_rotary_pos_emb -> apply_rope_triton")

    # Sanity-check: run a tiny dummy tensor through both and compare
    try:
        import torch
        if torch.cuda.is_available():
            q   = torch.randn(1, 4, 4, 64, device="cuda", dtype=torch.float16)
            k   = torch.randn_like(q)
            cos = torch.randn(1, 4, 64, device="cuda", dtype=torch.float16)
            sin = torch.randn_like(cos)
            q_ref, k_ref = original(q.clone(), k.clone(), cos, sin)
            q_tri, k_tri = apply_rope_triton(q.clone(), k.clone(), cos, sin)
            if not torch.allclose(q_ref, q_tri, atol=1e-2):
                log.error(
                    "[Triton RoPE] Sanity check FAILED (max err=%.4f) — "
                    "reverting to HF native RoPE.",
                    (q_ref - q_tri).abs().max().item(),
                )
                llama_module.apply_rotary_pos_emb = original
                return False
            log.info("[Triton RoPE] Sanity check passed.")
    except Exception as exc:
        log.warning("[Triton RoPE] Sanity check error: %s — reverting.", exc)
        llama_module.apply_rotary_pos_emb = original
        return False

    return True


def _warmup_triton_kernel() -> None:
    """
    Trigger Triton JIT compilation before serving traffic.
    First call compiles and caches the binary (~5-30s); subsequent calls are fast.
    """
    try:
        import torch
        from src.rope_kernel import apply_rope_triton
        q  = torch.randn(1, 32, 1, 64, device="cuda", dtype=torch.float16)
        k  = torch.randn_like(q)
        cs = torch.randn(1, 1, 64, device="cuda", dtype=torch.float16)
        apply_rope_triton(q, k, cs, cs)
        torch.cuda.synchronize()
        log.info("[Triton RoPE] Kernel compiled and warmed up.")
    except Exception as exc:
        log.warning("[Triton RoPE] Warmup failed: %s", exc)


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load TinyLlama-1.1B via HuggingFace transformers with INT4 bitsandbytes NF4.
    Returns (model, tokenizer) tuple.
    """
    try:
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    except ImportError as exc:
        log.error("Missing dependency: %s — check requirements.txt", exc)
        sys.exit(1)

    if not torch.cuda.is_available():
        log.warning("CUDA not available — INT4 quantization requires a GPU. Expect errors.")

    # Apply Triton RoPE patch BEFORE from_pretrained so the module-level
    # function is replaced before any LlamaAttention.forward() is ever called.
    triton_active = False
    if TRITON_ROPE:
        triton_active = _apply_triton_rope_patch()
        if not triton_active:
            log.warning("[Triton RoPE] Patch not applied — running with HF native RoPE.")

    log.info("Loading model: %s", HF_MODEL_ID)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # NormalFloat4 — best quality/size tradeoff
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,       # saves ~0.4 bits/param extra
    )

    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",          # only one GPU visible inside container; auto picks cuda:0
        torch_dtype=torch.float16,
    )
    model.eval()

    if triton_active:
        _warmup_triton_kernel()

    log.info("Model loaded on device: %s", next(model.parameters()).device)
    return model, tokenizer


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(llm, request: dict) -> dict:
    """Run a single inference request and return a response dict."""
    model, tokenizer = llm
    prompt     = request["prompt"]
    max_tokens = request.get("max_tokens", 128)
    req_id     = request.get("id", "unknown")

    try:
        inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,                        # greedy — deterministic, faster
                pad_token_id=tokenizer.eos_token_id,   # TinyLlama has no dedicated pad token
                eos_token_id=tokenizer.eos_token_id,
            )

        # Slice off the prompt tokens — return only the newly generated tokens
        generated_ids    = output_ids[0, input_len:]
        text             = tokenizer.decode(generated_ids, skip_special_tokens=True)
        tokens_generated = len(generated_ids)

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
                except Exception:  # noqa: BLE001
                    log.exception("Error handling connection")

    # Clean up socket file on exit.
    if os.path.exists(SOCKET_PATH):
        os.unlink(SOCKET_PATH)
    log.info("Worker exited cleanly.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model, tokenizer = load_model()
    run_server((model, tokenizer))
