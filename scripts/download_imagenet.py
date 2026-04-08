"""Download the full ImageNet-1K dataset to the NAS data volume.

Downloads the dataset in HuggingFace's native Arrow/Parquet format — no
individual image extraction needed.  The download is **resumable**: any
parquet shards already on disk are skipped automatically.

Usage:
    python scripts/download_imagenet.py                 # train + validation
    python scripts/download_imagenet.py --split train   # train only

Downloads to: /mnt/NAS/data/ds5725/visaebench/datasets/imagenet-1k/
Requires HF_TOKEN for gated dataset access.

After downloading, use cache_activations.py with --use_hf_local:
    python -m src.caching.cache_activations \\
        --backbone dinov2_vitb14 \\
        --use_hf_local \\
        --output_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \\
        --num_images 10000
"""

import argparse
import os
import sys
import time

# Disable hf_transfer (XET protocol) — it stalls on NFS for large files.
# Standard HTTP downloads are slower per-file but reliable and resumable.
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

# Ensure HF caches go to the data volume
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import src.utils.paths  # noqa: F401  — sets HF_HOME env vars on import

from src.utils.paths import DATASET_ROOT

from datasets import load_dataset
from dotenv import load_dotenv
from huggingface_hub import login


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download ImageNet-1K to NAS data volume.")
    parser.add_argument(
        "--split",
        type=str,
        nargs="+",
        default=["train", "validation"],
        help="Which split(s) to download (default: train validation).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        login(token=hf_token)
        print("Logged in to HuggingFace Hub")
    else:
        print("ERROR: HF_TOKEN not found. ImageNet-1K is a gated dataset.")
        print("  1. Accept the license at https://huggingface.co/datasets/imagenet-1k")
        print("  2. Set HF_TOKEN in your .env file or environment")
        sys.exit(1)

    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    print(f"HF_HOME: {hf_home}")
    print(f"Splits to download: {args.split}")
    print("Already-downloaded shards will be skipped (resumable).\n")

    for split in args.split:
        print(f"{'='*60}")
        print(f"  Downloading '{split}' split ...")
        print(f"{'='*60}")
        t0 = time.time()

        ds = load_dataset(
            "imagenet-1k",
            split=split,
        )

        elapsed = time.time() - t0
        print(f"\n  ✅ '{split}' ready — {len(ds):,} examples")
        print(f"     Time: {elapsed/60:.1f} min\n") 

    print("=" * 60)
    print("Download complete!")
    print(f"Dataset stored in HF_HOME: {hf_home}")
    print()
    print("To cache activations from the local dataset:")
    print("  python -m src.caching.cache_activations \\")
    print("      --backbone dinov2_vitb14 \\")
    print("      --use_hf_local \\")
    print(f"      --output_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \\")
    print("      --num_images 10000")


if __name__ == "__main__":
    main()
