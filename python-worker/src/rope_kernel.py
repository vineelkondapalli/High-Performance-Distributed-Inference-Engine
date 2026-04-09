"""
Fused RoPE (Rotary Position Embedding) Triton Kernel
=====================================================
Replaces HuggingFace's apply_rotary_pos_emb() with a single fused GPU kernel
that performs the full rotate_half + multiply-add in one pass.

HuggingFace issues ~8 separate CUDA kernel launches per token for RoPE:
  rotate_half: 2 slices + 1 cat
  apply:       2 mul + 1 cat + 2 mul + 1 add  (×2 for q and k)

This kernel fuses all of that into one launch per (batch, head, seq_pos) row.

Tensor layouts (HuggingFace LlamaAttention convention):
  q, k    : [batch, num_heads, seq_len, head_dim]
  cos, sin: [batch, seq_len, head_dim]   (before the unsqueeze in HF's version)

Usage:
  from rope_kernel import apply_rope_triton
  q_out, k_out = apply_rope_triton(q, k, cos, sin)

Standalone correctness test:
  python src/rope_kernel.py
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton Kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _rope_fwd_kernel(
    # Pointers to input/output tensors in GPU DRAM
    Q_ptr,       # q input:   [batch, n_heads, seq_len, head_dim]
    K_ptr,       # k input:   [batch, n_heads, seq_len, head_dim]
    COS_ptr,     # cos input: [batch, seq_len, head_dim]
    SIN_ptr,     # sin input: [batch, seq_len, head_dim]
    Q_out_ptr,   # q output:  same shape as Q
    K_out_ptr,   # k output:  same shape as K

    # Strides for q/k tensors — how many elements to advance per index
    # For a contiguous [B, H, S, D] tensor: stride(i) = product of dims after i
    stride_qb,   # stride along batch dim
    stride_qh,   # stride along head dim
    stride_qs,   # stride along seq_len dim
    stride_qd,   # stride along head_dim (== 1 for contiguous row-major)

    # Strides for cos/sin tensors [B, S, D]
    stride_cb,   # stride along batch dim
    stride_cs,   # stride along seq_len dim
    stride_cd,   # stride along head_dim (== 1 for contiguous)

    # Shape values — n_heads and seq_len are runtime values, HEAD_DIM is constexpr
    n_heads,
    seq_len,
    HEAD_DIM: tl.constexpr,  # head dimension — must be power-of-2, known at compile time
    HALF_DIM: tl.constexpr,  # HEAD_DIM // 2
):
    """
    Each program instance processes ONE row: one (batch, head, seq_pos) triple.

    Grid: grid = (batch * n_heads * seq_len,)
    Each program gets a unique pid in [0, batch*n_heads*seq_len).

    Why one program per row?
      Each row is head_dim elements (64 for TinyLlama). At 64 fp16 values = 128 bytes,
      this fits in a single 128-byte coalesced memory transaction. Mapping one warp
      per row maximises memory bandwidth utilisation with no wasted threads.
    """

    # ── Decode (batch, head, seq_pos) from the flat program ID ──────────────
    # tl.program_id(axis=0) returns this program's index in the 1D grid.
    # We encode: pid = b * (n_heads * seq_len) + h * seq_len + s
    pid       = tl.program_id(axis=0)
    b         = pid // (n_heads * seq_len)
    remainder = pid % (n_heads * seq_len)
    h         = remainder // seq_len
    s         = remainder % seq_len

    # ── Compute base offsets for this row ────────────────────────────────────
    # All pointer arithmetic is in units of elements (not bytes).
    q_base   = b * stride_qb + h * stride_qh + s * stride_qs
    cos_base = b * stride_cb + s * stride_cs

    # ── Index vectors for the two halves ─────────────────────────────────────
    # tl.arange(0, N) produces a compile-time vector [0, 1, ..., N-1].
    # N must be a power-of-2 constexpr — this is why HEAD_DIM is constexpr.
    half_dims = tl.arange(0, HALF_DIM)   # [0, 1, ..., HALF_DIM-1]

    # ── Load the first half [0:H/2] and second half [H/2:H] of each tensor ──
    # tl.load(ptr + offsets) performs a vectorised load from DRAM to registers.
    # Adding a scalar base offset to a vector of half_dims gives the address
    # of each element to load — this is the vectorised pointer arithmetic.
    q_first  = tl.load(Q_ptr   + q_base   + half_dims)
    q_second = tl.load(Q_ptr   + q_base   + half_dims + HALF_DIM)
    k_first  = tl.load(K_ptr   + q_base   + half_dims)
    k_second = tl.load(K_ptr   + q_base   + half_dims + HALF_DIM)

    cos_first  = tl.load(COS_ptr + cos_base + half_dims)
    cos_second = tl.load(COS_ptr + cos_base + half_dims + HALF_DIM)
    sin_first  = tl.load(SIN_ptr + cos_base + half_dims)
    sin_second = tl.load(SIN_ptr + cos_base + half_dims + HALF_DIM)

    # ── Fused rotate_half + RoPE in registers ─────────────────────────────────
    # HuggingFace reference:
    #   rotate_half(x) = cat(-x[H/2:], x[:H/2])
    #   q_out = q * cos + rotate_half(q) * sin
    #
    # Expanded per half:
    #   out[:H/2] = q[:H/2] * cos[:H/2]  +  (-q[H/2:]) * sin[:H/2]
    #             = q[:H/2] * cos[:H/2]  -    q[H/2:]  * sin[:H/2]
    #   out[H/2:] = q[H/2:] * cos[H/2:] +    q[:H/2]  * sin[H/2:]
    #
    # All arithmetic happens in registers — no intermediate tensors in DRAM.
    q_out_first  = q_first  * cos_first  - q_second * sin_first
    q_out_second = q_second * cos_second + q_first  * sin_second
    k_out_first  = k_first  * cos_first  - k_second * sin_first
    k_out_second = k_second * cos_second + k_first  * sin_second

    # ── Store results back to DRAM ────────────────────────────────────────────
    # tl.store(ptr + offsets, values) writes the register values back to DRAM.
    # Uses the same pointer arithmetic as the loads above.
    tl.store(Q_out_ptr + q_base + half_dims,              q_out_first)
    tl.store(Q_out_ptr + q_base + half_dims + HALF_DIM,   q_out_second)
    tl.store(K_out_ptr + q_base + half_dims,              k_out_first)
    tl.store(K_out_ptr + q_base + half_dims + HALF_DIM,   k_out_second)


# ─────────────────────────────────────────────────────────────────────────────
# Python Wrapper — drop-in for HuggingFace's apply_rotary_pos_emb()
# ─────────────────────────────────────────────────────────────────────────────

def apply_rope_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    position_ids=None,     # accepted but unused — cos/sin already positional
    unsqueeze_dim: int = 1,  # accepted but unused — we handle dims internally
) -> tuple:
    """
    Drop-in replacement for HuggingFace's apply_rotary_pos_emb().

    Args:
        q:   [batch, n_heads, seq_len, head_dim]  — float16, CUDA
        k:   [batch, n_heads, seq_len, head_dim]
        cos: [batch, seq_len, head_dim]            — from LlamaRotaryEmbedding.forward()
        sin: [batch, seq_len, head_dim]
        position_ids:  ignored (cos/sin already computed for the right positions)
        unsqueeze_dim: ignored (we handle the head dim internally)

    Returns:
        (q_embed, k_embed) — same shapes and dtype as q, k

    Constraints:
        - head_dim must be a power of 2 (TinyLlama: 64 ✓)
        - tensors must be on CUDA
        - dtype should be float16 or bfloat16
    """
    # Ensure contiguous memory layout — Triton stride math assumes row-major
    q   = q.contiguous()
    k   = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    batch, n_heads, seq_len, head_dim = q.shape
    assert head_dim % 2 == 0, f"head_dim must be even, got {head_dim}"
    assert (head_dim & (head_dim - 1)) == 0, (
        f"head_dim must be a power of 2 for the Triton kernel, got {head_dim}. "
        f"For non-power-of-2 head dims, use the HuggingFace reference implementation."
    )

    half_dim = head_dim // 2

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)

    # One program per (batch, head, seq_pos) row
    grid = (batch * n_heads * seq_len,)

    _rope_fwd_kernel[grid](
        q, k, cos, sin, q_out, k_out,
        # q/k strides (in elements)
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        # cos/sin strides
        cos.stride(0), cos.stride(1), cos.stride(2),
        # runtime shape values
        n_heads=n_heads,
        seq_len=seq_len,
        # compile-time constexprs — Triton generates a specialised kernel per unique combo
        HEAD_DIM=head_dim,
        HALF_DIM=half_dim,
    )

    return q_out, k_out


# ─────────────────────────────────────────────────────────────────────────────
# Correctness Tests — run standalone: python src/rope_kernel.py
# ─────────────────────────────────────────────────────────────────────────────

def _hf_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """HuggingFace reference rotate_half (from modeling_llama.py)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _hf_apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> tuple:
    """HuggingFace reference apply_rotary_pos_emb (from modeling_llama.py)."""
    cos = cos.unsqueeze(unsqueeze_dim)  # [B, 1, S, D]
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (_hf_rotate_half(q) * sin)
    k_embed = (k * cos) + (_hf_rotate_half(k) * sin)
    return q_embed, k_embed


def test_correctness() -> None:
    """Verify Triton output matches HuggingFace reference within fp16 tolerance."""
    torch.manual_seed(42)
    B, H, S, D = 1, 32, 64, 64   # TinyLlama-like dims
    device = "cuda"
    dtype  = torch.float16

    q   = torch.randn(B, H, S, D, device=device, dtype=dtype)
    k   = torch.randn(B, H, S, D, device=device, dtype=dtype)
    cos = torch.randn(B, S, D,    device=device, dtype=dtype)
    sin = torch.randn(B, S, D,    device=device, dtype=dtype)

    q_ref, k_ref = _hf_apply_rotary_pos_emb(q, k, cos, sin)
    q_tri, k_tri = apply_rope_triton(q, k, cos, sin)

    # fp16 accumulation can give ~1e-3 relative error per op — allow atol=1e-2
    q_ok = torch.allclose(q_ref, q_tri, atol=1e-2, rtol=1e-3)
    k_ok = torch.allclose(k_ref, k_tri, atol=1e-2, rtol=1e-3)

    if q_ok and k_ok:
        print("[PASS] Basic correctness: Triton matches HF reference within fp16 tolerance.")
    else:
        max_q = (q_ref - q_tri).abs().max().item()
        max_k = (k_ref - k_tri).abs().max().item()
        print(f"[FAIL] q_match={q_ok}  k_match={k_ok}")
        print(f"       Max q error: {max_q:.6f}   Max k error: {max_k:.6f}")
        raise AssertionError("Triton RoPE kernel output does not match HF reference.")


def test_shapes() -> None:
    """Test correctness across several shapes to catch stride/indexing bugs."""
    device = "cuda"
    dtype  = torch.float16
    configs = [
        (1, 4,  1, 64),    # single-token decode step (batch=1, seq=1)
        (1, 32, 1, 64),    # TinyLlama single-token decode
        (2, 32, 128, 64),  # typical prefill batch
        (1, 32, 512, 64),  # long prompt
    ]
    for B, H, S, D in configs:
        q   = torch.randn(B, H, S, D, device=device, dtype=dtype)
        k   = torch.randn(B, H, S, D, device=device, dtype=dtype)
        cos = torch.randn(B, S, D,    device=device, dtype=dtype)
        sin = torch.randn(B, S, D,    device=device, dtype=dtype)
        q_ref, k_ref = _hf_apply_rotary_pos_emb(q, k, cos, sin)
        q_tri, k_tri = apply_rope_triton(q, k, cos, sin)
        assert torch.allclose(q_ref, q_tri, atol=1e-2), f"Shape {(B,H,S,D)}: q mismatch"
        assert torch.allclose(k_ref, k_tri, atol=1e-2), f"Shape {(B,H,S,D)}: k mismatch"
        print(f"  [PASS] shape {(B, H, S, D)}")


def test_numerical_stability() -> None:
    """Check that large-magnitude inputs don't produce NaN or Inf."""
    device = "cuda"
    dtype  = torch.float16
    B, H, S, D = 1, 32, 64, 64

    # fp16 max is ~65504 — use values near but within range
    q   = torch.full((B, H, S, D), 100.0, device=device, dtype=dtype)
    k   = torch.full((B, H, S, D), 100.0, device=device, dtype=dtype)
    cos = torch.ones(B, S, D, device=device, dtype=dtype)
    sin = torch.zeros(B, S, D, device=device, dtype=dtype)

    q_out, k_out = apply_rope_triton(q, k, cos, sin)
    assert not torch.isnan(q_out).any(), "NaN in q_out"
    assert not torch.isinf(q_out).any(), "Inf in q_out"
    # With cos=1, sin=0: output should equal input
    assert torch.allclose(q_out, q, atol=1e-2), "Identity test failed (cos=1, sin=0)"
    print("[PASS] Numerical stability: no NaN/Inf, identity check passed.")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("[SKIP] CUDA not available — tests require a GPU.")
        raise SystemExit(0)

    print("=== Triton RoPE Kernel Correctness Tests ===\n")
    test_correctness()

    print("\n=== Shape Coverage Tests ===")
    test_shapes()

    print("\n=== Numerical Stability Test ===")
    test_numerical_stability()

    print("\n[All tests passed]")
