"""
Inference Profiler — torch.profiler
====================================
Profiles the HuggingFace TinyLlama inference pipeline to identify where
GPU time is spent (matmul, RoPE, attention, quantized ops, etc.).

Output:
  Chrome trace JSON  — open at https://ui.perfetto.dev (drag & drop)
  Top-ops table      — printed to stdout, sorted by CUDA self time

Usage (inside container or on GPU machine):
  python src/profile_inference.py --warmup 2 --steps 5 --max-tokens 50

  # With Triton RoPE patch active:
  TRITON_ROPE=1 python src/profile_inference.py --warmup 2 --steps 5 --max-tokens 50

  # Via docker compose:
  docker compose run --rm python-worker python src/profile_inference.py \\
      --warmup 2 --steps 5 --max-tokens 50

Arguments:
  --warmup      Number of un-profiled warmup steps (default: 2)
  --steps       Number of profiled steps (default: 5)
  --max-tokens  Max tokens to generate per step (default: 50)
  --trace-path  Output path for Chrome trace JSON (default: /tmp/trace.json)
  --model-id    HuggingFace model ID (default: from HF_MODEL_ID env or TinyLlama)
  --triton-rope Apply Triton RoPE patch before profiling (same as TRITON_ROPE=1)
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile LLM inference with torch.profiler")
    p.add_argument("--warmup",      type=int, default=2,
                   help="Warmup steps (not profiled)")
    p.add_argument("--steps",       type=int, default=5,
                   help="Profiled steps")
    p.add_argument("--max-tokens",  type=int, default=50,
                   help="Max new tokens per inference call")
    p.add_argument("--trace-path",  default="/tmp/trace.json",
                   help="Output path for Chrome trace JSON")
    p.add_argument("--model-id",
                   default=os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
                   help="HuggingFace model ID")
    p.add_argument("--triton-rope", action="store_true",
                   help="Apply Triton RoPE monkey-patch before profiling")
    return p.parse_args()


def load_model_and_tokenizer(model_id: str):
    """
    Load the model identically to worker.py so profiles reflect production behaviour.
    Must be kept in sync with worker.py's load_model().
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.eval()
    return model, tokenizer


def apply_triton_patch() -> bool:
    """Apply rope_kernel.py monkey-patch. Returns True if successful."""
    try:
        import transformers.models.llama.modeling_llama as llama_module
        from rope_kernel import apply_rope_triton
        if not hasattr(llama_module, "apply_rotary_pos_emb"):
            print("[profiler] WARNING: apply_rotary_pos_emb not found — skipping patch.")
            return False
        llama_module.apply_rotary_pos_emb = apply_rope_triton
        print("[profiler] Triton RoPE patch applied.")
        return True
    except ImportError as e:
        print(f"[profiler] WARNING: Could not apply Triton patch: {e}")
        return False


def run_single(model, tokenizer, prompt: str, max_tokens: int) -> int:
    """Run one inference call. Returns number of tokens generated."""
    import torch
    inputs    = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return len(output_ids[0, input_len:])


def main() -> None:
    import torch
    from torch.profiler import profile, record_function, ProfilerActivity

    args = parse_args()

    if not torch.cuda.is_available():
        print("[profiler] ERROR: CUDA not available. Profiling requires a GPU.")
        sys.exit(1)

    if args.triton_rope or os.getenv("TRITON_ROPE", "0") == "1":
        apply_triton_patch()

    print(f"[profiler] Loading model: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id)
    print(f"[profiler] Model on device: {next(model.parameters()).device}")

    prompt = (
        "Explain the attention mechanism in transformer models. "
        "What is the role of queries, keys, and values?"
    )

    # ── Warmup — not profiled, ensures CUDA kernels are compiled/cached ──────
    print(f"[profiler] Warmup ({args.warmup} step(s))...")
    for _ in range(args.warmup):
        run_single(model, tokenizer, prompt, args.max_tokens)
    torch.cuda.synchronize()

    # ── Profiled steps ───────────────────────────────────────────────────────
    print(f"[profiler] Profiling {args.steps} step(s) x {args.max_tokens} max tokens...")

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,     # capture tensor shapes per op
        profile_memory=True,    # track CUDA memory allocations per op
        with_stack=False,       # set True for Python call stack (significantly slower)
    ) as prof:
        for step in range(args.steps):
            with record_function(f"inference_step_{step}"):
                tokens = run_single(model, tokenizer, prompt, args.max_tokens)
            prof.step()          # marks profiler step boundary for averaging
        torch.cuda.synchronize()

    # ── Export Chrome trace ──────────────────────────────────────────────────
    prof.export_chrome_trace(args.trace_path)
    print(f"\n[profiler] Chrome trace saved: {args.trace_path}")
    print(f"           View at: https://ui.perfetto.dev  (drag & drop the JSON file)")

    # ── Top-ops tables ───────────────────────────────────────────────────────
    sep = "=" * 80

    print(f"\n{sep}")
    print("TOP 20 OPS BY CUDA SELF TIME  (what to optimize — no children)")
    print(sep)
    print(prof.key_averages(group_by_input_shape=False).table(
        sort_by="self_cuda_time_total",
        row_limit=20,
    ))

    print(f"\n{sep}")
    print("TOP 10 OPS BY CPU SELF TIME")
    print(sep)
    print(prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=10,
    ))

    print(f"\n{sep}")
    print("TOP 10 OPS BY CUDA MEMORY USAGE")
    print(sep)
    print(prof.key_averages().table(
        sort_by="self_cuda_memory_usage",
        row_limit=10,
    ))

    # ── What to look for ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("INTERPRETATION GUIDE")
    print(sep)
    print("""
  aten::mm / aten::bmm       — Matmul ops (will dominate; driven by bitsandbytes dequant)
  aten::mul + aten::cat      — Likely RoPE rotate_half (target for Triton kernel)
  aten::add                  — Residual connections, RoPE apply
  aten::softmax              — Attention score normalisation
  aten::scaled_dot_product_attention — Flash-attention path (if enabled)
  bitsandbytes ops           — INT4 dequantisation

  RoPE typically appears as several small aten::mul + aten::cat ops.
  After applying TRITON_ROPE=1, those should collapse into a single
  'triton_' op with lower cumulative CUDA time.
    """.strip())


if __name__ == "__main__":
    main()
