"""
Benchmark: Python-direct vs C++ Sidecar
========================================
Sends sequential requests to both endpoints and plots throughput + latency.

Requirements:
  pip install requests matplotlib numpy

Usage:
  # Start docker compose first
  docker compose up -d

  # Run benchmark (sequential, not concurrent — isolates inference latency)
  python scripts/benchmark.py --requests 20 --max-tokens 50

  # Output: benchmark_results.png
"""

import argparse
import json
import statistics
import time
from datetime import datetime

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
]


def run_sequential(url: str, n: int, max_tokens: int) -> list[float]:
    """Send n sequential requests, return list of latencies in ms."""
    latencies = []
    for i in range(n):
        prompt = PROMPTS[i % len(PROMPTS)]
        payload = {"prompt": prompt, "max_tokens": max_tokens}

        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, timeout=300)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Request {i+1} FAILED: {e}")
            continue
        elapsed_ms = (time.perf_counter() - start) * 1000

        latencies.append(elapsed_ms)
        print(f"  [{i+1:>3}/{n}] {elapsed_ms:7.0f} ms")

    return latencies


def print_stats(label: str, latencies: list[float]) -> None:
    if not latencies:
        print(f"{label}: no data")
        return
    print(f"\n{label}:")
    print(f"  Requests completed : {len(latencies)}")
    print(f"  Mean latency       : {statistics.mean(latencies):.0f} ms")
    print(f"  Median latency     : {statistics.median(latencies):.0f} ms")
    print(f"  p95 latency        : {sorted(latencies)[int(len(latencies)*0.95)]:.0f} ms")
    print(f"  Min / Max          : {min(latencies):.0f} / {max(latencies):.0f} ms")
    throughput = len(latencies) / (sum(latencies) / 1000)
    print(f"  Effective throughput: {throughput:.3f} req/s")


def plot_results(cpp_latencies: list[float], python_latencies: list[float]) -> None:
    if not HAS_PLOT:
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("C++ Sidecar vs Python Direct — Benchmark Results", fontsize=14)

    # ── Latency over time ─────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(cpp_latencies,    label="C++ Sidecar", marker="o", linewidth=2)
    ax1.plot(python_latencies, label="Python Direct", marker="s", linewidth=2, linestyle="--")
    ax1.set_xlabel("Request #")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency Per Request")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # ── Box plot ──────────────────────────────────────────────────────────────
    ax2 = axes[1]
    data   = [cpp_latencies, python_latencies]
    labels = ["C++ Sidecar", "Python Direct"]
    bp     = ax2.boxplot(data, labels=labels, patch_artist=True)
    colors = ["#4C72B0", "#DD8452"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax2.set_ylabel("Latency (ms)")
    ax2.set_title("Latency Distribution")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = "benchmark_results.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nChart saved to: {output_path}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Benchmark inference endpoints")
    parser.add_argument("--requests",   type=int, default=10,
                        help="Number of sequential requests per endpoint (default: 10)")
    parser.add_argument("--max-tokens", type=int, default=50,
                        help="Max tokens per response (default: 50)")
    parser.add_argument("--cpp-url",    default="http://localhost:8080/infer",
                        help="C++ sidecar URL")
    parser.add_argument("--python-url", default="http://localhost:8081/infer",
                        help="Python direct URL (runs without C++ proxy)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Benchmark — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Requests per endpoint : {args.requests}")
    print(f"  Max tokens            : {args.max_tokens}")
    print("=" * 60)

    print(f"\n[1/2] C++ Sidecar ({args.cpp_url})")
    cpp_latencies = run_sequential(args.cpp_url, args.requests, args.max_tokens)

    print(f"\n[2/2] Python Direct ({args.python_url})")
    python_latencies = run_sequential(args.python_url, args.requests, args.max_tokens)

    print("\n" + "=" * 60)
    print_stats("C++ Sidecar",   cpp_latencies)
    print_stats("Python Direct", python_latencies)
    print("=" * 60)

    plot_results(cpp_latencies, python_latencies)


if __name__ == "__main__":
    main()
