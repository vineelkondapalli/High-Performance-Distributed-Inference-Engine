"""
Benchmark: Latency, Per-Token Throughput, and Triton RoPE A/B Comparison
=========================================================================
Sends sequential requests to the inference endpoint and reports:
  - End-to-end latency (mean / median / p95 / p99 / min / max)
  - Per-token latency (ms/token) and throughput (tokens/sec)
  - Optional A/B comparison between HF baseline and Triton RoPE

Requirements:
  pip install requests matplotlib numpy

Usage:
  # Start docker compose first
  docker compose up -d

  # Baseline benchmark
  python scripts/benchmark.py --requests 20 --max-tokens 50

  # Label results as Triton RoPE run (worker must already run with TRITON_ROPE=1):
  python scripts/benchmark.py --requests 20 --max-tokens 50 --triton-rope

  # Interactive A/B: runs baseline, prompts you to restart worker, runs Triton, prints diff
  python scripts/benchmark.py --requests 20 --max-tokens 50 --compare

  # Also benchmark the legacy Python-direct endpoint (if running):
  python scripts/benchmark.py --requests 20 --python-url http://localhost:8081/infer

  Output: benchmark_results.png (if matplotlib is available)
"""

import argparse
import statistics
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import requests

try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("matplotlib/numpy not installed — will print results only.")

PROMPTS = [
    "What is artificial intelligence?",
    "Explain cloud computing briefly.",
    "What is the difference between TCP and UDP?",
    "Describe supervised learning.",
    "What is a Kubernetes pod?",
    "What is backpropagation?",
    "Explain gradient descent in one paragraph.",
    "What is a transformer model?",
    "What is quantization in machine learning?",
    "Describe the RoPE positional encoding method.",
]


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    latency_ms: float
    tokens_generated: int

    @property
    def ms_per_token(self) -> float:
        return self.latency_ms / self.tokens_generated if self.tokens_generated > 0 else float("inf")

    @property
    def tokens_per_sec(self) -> float:
        return self.tokens_generated / (self.latency_ms / 1000.0) if self.tokens_generated > 0 else 0.0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _percentile(data: list, p: float) -> float:
    """Return the p-th percentile (0–100) of a list."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * p / 100.0
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


# ── Core benchmark logic ──────────────────────────────────────────────────────

def run_sequential(url: str, n: int, max_tokens: int,
                   label: str = "") -> list:
    """Send n sequential requests, return list of RequestResult."""
    results = []
    tag = f"[{label}]" if label else ""
    for i in range(n):
        prompt  = PROMPTS[i % len(PROMPTS)]
        payload = {"prompt": prompt, "max_tokens": max_tokens}

        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except Exception as exc:
            print(f"  {tag} Request {i+1} FAILED: {exc}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000

        body   = resp.json()
        tokens = body.get("tokens_generated", 0)
        r      = RequestResult(latency_ms=elapsed_ms, tokens_generated=tokens)
        results.append(r)

        print(f"  {tag} [{i+1:>3}/{n}] "
              f"{elapsed_ms:7.0f} ms | "
              f"{tokens:3d} tok | "
              f"{r.ms_per_token:6.1f} ms/tok | "
              f"{r.tokens_per_sec:6.1f} tok/s")

    return results


def compute_stats(label: str, results: list) -> dict:
    """Compute and return a stats dict for the result set."""
    if not results:
        return {"label": label}

    latencies     = [r.latency_ms      for r in results]
    ms_per_tokens = [r.ms_per_token    for r in results if r.tokens_generated > 0]
    toks_per_sec  = [r.tokens_per_sec  for r in results if r.tokens_generated > 0]
    total_tokens  = sum(r.tokens_generated for r in results)

    return {
        "label":            label,
        "n":                len(results),
        "total_tokens":     total_tokens,
        "mean_ms":          statistics.mean(latencies),
        "median_ms":        statistics.median(latencies),
        "p95_ms":           _percentile(latencies, 95),
        "p99_ms":           _percentile(latencies, 99),
        "min_ms":           min(latencies),
        "max_ms":           max(latencies),
        "mean_ms_per_tok":  statistics.mean(ms_per_tokens) if ms_per_tokens else 0.0,
        "mean_tok_per_sec": statistics.mean(toks_per_sec)  if toks_per_sec  else 0.0,
        "throughput_rps":   len(results) / (sum(latencies) / 1000),
    }


def print_stats(stats: dict) -> None:
    if not stats or "n" not in stats:
        print(f"{stats.get('label', '?')}: no data")
        return
    print(f"\n{'─'*60}")
    print(f"  {stats['label']}")
    print(f"{'─'*60}")
    print(f"  Requests completed  : {stats['n']}")
    print(f"  Total tokens        : {stats['total_tokens']}")
    print(f"  Mean latency        : {stats['mean_ms']:.1f} ms")
    print(f"  Median latency      : {stats['median_ms']:.1f} ms")
    print(f"  p95 latency         : {stats['p95_ms']:.1f} ms")
    print(f"  p99 latency         : {stats['p99_ms']:.1f} ms")
    print(f"  Min / Max           : {stats['min_ms']:.1f} / {stats['max_ms']:.1f} ms")
    print(f"  Mean ms/token       : {stats['mean_ms_per_tok']:.2f}")
    print(f"  Mean tokens/sec     : {stats['mean_tok_per_sec']:.1f}")
    print(f"  Throughput          : {stats['throughput_rps']:.3f} req/s")


def print_comparison(baseline: dict, triton: dict) -> None:
    """Print a before/after delta table for two stats dicts."""
    if not baseline or not triton or "n" not in baseline or "n" not in triton:
        return

    metrics = [
        ("mean_ms",          "Mean latency",      ".1f",  "ms",    True),
        ("p95_ms",           "p95 latency",       ".1f",  "ms",    True),
        ("p99_ms",           "p99 latency",       ".1f",  "ms",    True),
        ("mean_ms_per_tok",  "ms / token",        ".2f",  "ms",    True),
        ("mean_tok_per_sec", "Tokens / sec",      ".1f",  "tok/s", False),
        ("throughput_rps",   "Throughput (req/s)",",.3f", "r/s",   False),
    ]

    print(f"\n{'='*72}")
    print(f"  A/B COMPARISON")
    print(f"  Baseline : {baseline['label']}")
    print(f"  Triton   : {triton['label']}")
    print(f"{'='*72}")
    print(f"  {'Metric':<22} {'Baseline':>12}  {'Triton RoPE':>12}  {'Delta':>10}  Note")
    print(f"  {'-'*68}")

    for key, label, fmt, unit, lower_is_better in metrics:
        b_val = baseline.get(key, 0)
        t_val = triton.get(key, 0)
        if b_val == 0:
            print(f"  {label:<22} {'N/A':>12}  {'N/A':>12}")
            continue
        diff = t_val - b_val
        pct  = diff / b_val * 100
        sign = "+" if diff > 0 else ""
        better = (diff < 0) if lower_is_better else (diff > 0)
        note = "BETTER" if better else ("WORSE" if diff != 0 else "SAME")
        print(f"  {label:<22} {b_val:>10{fmt}}{unit:<4}  "
              f"{t_val:>10{fmt}}{unit:<4}  "
              f"{sign}{pct:>7.1f}%   {note}")

    print(f"{'='*72}")


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_results(result_sets: list, labels: list,
                 output_path: str = "benchmark_results.png") -> None:
    """Plot latency-over-time and distribution for one or more result sets."""
    if not HAS_PLOT or not result_sets:
        return

    n_sets = len(result_sets)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Inference Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                 fontsize=13)

    colors  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    markers = ["o", "s", "^", "D"]

    # Latency over time
    ax1 = axes[0]
    for i, (results, label) in enumerate(zip(result_sets, labels)):
        latencies = [r.latency_ms for r in results]
        ax1.plot(latencies, label=label, marker=markers[i % 4],
                 color=colors[i % 4], linewidth=2, markersize=4)
    ax1.set_xlabel("Request #")
    ax1.set_ylabel("End-to-end latency (ms)")
    ax1.set_title("Latency Per Request")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Distribution box plot
    ax2 = axes[1]
    data = [[r.latency_ms for r in rs] for rs in result_sets]
    bp   = ax2.boxplot(data, labels=labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark inference endpoints with per-token latency stats")
    parser.add_argument("--requests",    type=int, default=10,
                        help="Sequential requests per run (default: 10)")
    parser.add_argument("--max-tokens",  type=int, default=50,
                        help="Max tokens per response (default: 50)")
    parser.add_argument("--url",         default="http://localhost:8080/infer",
                        help="Primary inference endpoint (C++ sidecar)")
    parser.add_argument("--python-url",  default=None,
                        help="Optional Python-direct endpoint for comparison")
    parser.add_argument("--triton-rope", action="store_true",
                        help="Label this run as 'Triton RoPE' (worker must already have patch)")
    parser.add_argument("--compare",     action="store_true",
                        help="Run baseline, pause, run Triton run, print diff table")
    parser.add_argument("--output",      default="benchmark_results.png",
                        help="Path for plot image (default: benchmark_results.png)")
    args = parser.parse_args()

    label = "Triton RoPE" if args.triton_rope else "HF Baseline"

    print("=" * 65)
    print(f"  Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Mode         : {label}")
    print(f"  Requests     : {args.requests}")
    print(f"  Max tokens   : {args.max_tokens}")
    print(f"  Endpoint     : {args.url}")
    print("=" * 65)

    all_results = []
    all_labels  = []

    # ── Primary endpoint ──────────────────────────────────────────────────────
    print(f"\nRunning {args.requests} requests ({label})...")
    results_a = run_sequential(args.url, args.requests, args.max_tokens, label)
    stats_a   = compute_stats(label, results_a)
    print_stats(stats_a)
    all_results.append(results_a)
    all_labels.append(label)

    # ── Optional A/B comparison ───────────────────────────────────────────────
    if args.compare:
        input(
            "\n>>> Restart the worker with TRITON_ROPE=1, wait for it to be healthy, "
            "then press Enter to run the Triton RoPE benchmark..."
        )
        triton_label = "Triton RoPE"
        print(f"\nRunning {args.requests} requests ({triton_label})...")
        results_b = run_sequential(args.url, args.requests, args.max_tokens, triton_label)
        stats_b   = compute_stats(triton_label, results_b)
        print_stats(stats_b)
        print_comparison(stats_a, stats_b)
        all_results.append(results_b)
        all_labels.append(triton_label)

    # ── Optional Python-direct endpoint ──────────────────────────────────────
    if args.python_url:
        py_label = "Python Direct"
        print(f"\nRunning {args.requests} requests ({py_label})...")
        results_py = run_sequential(args.python_url, args.requests, args.max_tokens, py_label)
        stats_py   = compute_stats(py_label, results_py)
        print_stats(stats_py)
        all_results.append(results_py)
        all_labels.append(py_label)

    # ── Plot all result sets ──────────────────────────────────────────────────
    plot_results(all_results, all_labels, output_path=args.output)


if __name__ == "__main__":
    main()
