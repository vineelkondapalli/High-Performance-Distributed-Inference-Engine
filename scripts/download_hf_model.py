"""
Pre-download HuggingFace model to local cache before starting containers.
========================================================================
Run this ONCE on your Linux GPU machine before `docker compose up --build`.
It downloads the model into ./models/hf_cache so the container's first
start isn't blocked waiting for a ~2GB download.

The ./models/hf_cache directory is mounted into the container as HF_HOME,
so the downloaded weights are immediately usable without re-downloading.

Usage:
  python scripts/download_hf_model.py

  # Different model:
  python scripts/download_hf_model.py --model-id TinyLlama/TinyLlama-1.1B-Chat-v1.0

  # Custom cache dir:
  python scripts/download_hf_model.py --cache-dir /data/hf_cache

Requirements:
  pip install transformers huggingface_hub
"""

import argparse
import os
import sys


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-download a HuggingFace model to the local cache")
    p.add_argument(
        "--model-id",
        default=os.getenv("HF_MODEL_ID", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="HuggingFace model ID (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)",
    )
    p.add_argument(
        "--cache-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "hf_cache"),
        help="Local cache directory (default: ./models/hf_cache)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve and create cache dir
    cache_dir = os.path.abspath(args.cache_dir)
    os.makedirs(cache_dir, exist_ok=True)

    print(f"Model ID  : {args.model_id}")
    print(f"Cache dir : {cache_dir}")
    print()

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("ERROR: transformers is not installed.")
        print("Run: pip install transformers huggingface_hub")
        sys.exit(1)

    # Set HF_HOME so the download lands in our target directory
    os.environ["HF_HOME"] = cache_dir

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=cache_dir)
    print("Tokenizer downloaded.")

    print("Downloading model weights (this may take a few minutes)...")
    # Download config + weights in safetensors format; no need to load into memory
    from huggingface_hub import snapshot_download
    local_path = snapshot_download(
        repo_id=args.model_id,
        cache_dir=cache_dir,
        ignore_patterns=["*.msgpack", "*.h5", "flax_*"],  # skip JAX/Flax weights
    )
    print(f"Model downloaded to: {local_path}")

    # Report disk usage
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, filenames in os.walk(cache_dir)
        for f in filenames
    )
    print(f"\nTotal cache size: {total_bytes / 1024**3:.2f} GB")
    print("\nDone. You can now run:")
    print("  docker compose up --build")
    print("The container will use the cached weights from ./models/hf_cache")
    print("without downloading them again.")


if __name__ == "__main__":
    main()
