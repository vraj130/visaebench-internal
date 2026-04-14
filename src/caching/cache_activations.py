"""Extract and cache layer activations from a ViT backbone on an image dataset.

Usage (local image directory):
    python -m src.caching.cache_activations \
        --backbone clip_vitb16 \
        --dataset_dir /data/imagenet/train \
        --output_dir /mnt/NAS/data/ds5725/visaebench/activations/clip_vitb16/layer_11/ \
        --num_images 10000

Usage (locally downloaded HF dataset — recommended):
    python -m src.caching.cache_activations \
        --backbone dinov2_vitb14 \
        --use_hf_local \
        --output_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \
        --num_images 10000 \
        --batch_size 64

Usage (HuggingFace streaming — slower, no local copy kept):
    python -m src.caching.cache_activations \
        --backbone dinov2_vitb14 \
        --output_dir /mnt/NAS/data/ds5725/visaebench/activations/dinov2_vitb14/layer_11/ \
        --use_hf_streaming \
        --num_images 10000 \
        --shard_size 5000 \
        --batch_size 64

Requires HF_TOKEN env var for gated imagenet-1k access.
"""

import argparse
import json
import os
import time

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Subset

import src.utils.paths  # noqa: F401  — redirects HF caches to data volume
from src.backbones import load_backbone
from src.caching.dataset import ImageFolderForCaching
from src.caching.shard_utils import WelfordAccumulator, save_shard


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache ViT backbone patch-token activations as sharded .pt files."
    )
    parser.add_argument(
        "--backbone",
        type=str,
        required=True,
        choices=["clip_vitb16", "dinov2_vitb14", "siglip_vitb16", "mae_vitb16", "deit_vitb16"],
        help="Backbone to extract activations from.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=None,
        help="Path to dataset root (ImageNet-style or flat directory of images). "
             "Not required when --use_hf_streaming is set.",
    )
    parser.add_argument(
        "--use_hf_local",
        action="store_true",
        default=False,
        help="Load from locally downloaded HF dataset cache at "
             "DATASET_ROOT/imagenet-1k (run scripts/download_imagenet.py first).",
    )
    parser.add_argument(
        "--use_hf_streaming",
        action="store_true",
        default=False,
        help="Stream imagenet-1k from HuggingFace Datasets instead of loading "
             "from a local directory. Requires HF_TOKEN env var.",
    )
    parser.add_argument(
        "--hf_dataset",
        type=str,
        default="imagenet-1k",
        help="HuggingFace dataset name to stream (default: imagenet-1k).",
    )
    parser.add_argument(
        "--hf_split",
        type=str,
        default="train",
        help="Dataset split to use when streaming from HuggingFace (default: train).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save shard .pt files and stats.json.",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=11,
        help="0-indexed transformer block to extract (default: 11).",
    )
    parser.add_argument(
        "--num_images",
        type=int,
        default=None,
        help="If set, only process this many images (for pilot runs).",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=5000,
        help="Number of images per shard file (default: 5000).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="DataLoader batch size (default: 64).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="DataLoader num_workers (default: 4).",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        default=False,
        help="Sample exactly num_images // 1000 images per ImageNet class instead of "
             "taking the first N sequentially. Requires --num_images and works with "
             "--use_hf_local and --use_hf_streaming.",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        default=False,
        help="Save activations as float16 instead of float32 (halves disk usage). "
             "Training code casts to float32 on load, so this is safe.",
    )
    args = parser.parse_args()
    if args.balanced and args.num_images is None:
        parser.error("--balanced requires --num_images to be set.")
    if args.balanced and args.dataset_dir is not None and not args.use_hf_local and not args.use_hf_streaming:
        parser.error("--balanced is only supported with --use_hf_local or --use_hf_streaming.")
    if not args.use_hf_streaming and not args.use_hf_local and args.dataset_dir is None:
        parser.error("--dataset_dir is required unless --use_hf_streaming or --use_hf_local is set.")
    return args


def _hf_balanced_batches(dataset_name: str, split: str, batch_size: int, num_images: int):
    """Yield batches of PIL images sampled equally across all 1000 ImageNet classes.

    Uses indexed access (non-streaming) to avoid scanning the full dataset:
    1. Load the dataset with Arrow/mmap backing.
    2. Pull the label column in one shot to build a class→indices map.
    3. Randomly sample ``num_images // 1000`` indices per class.
    4. Shuffle the combined indices and iterate only over those rows.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "imagenet-1k").
        split:        Dataset split (e.g. "train").
        batch_size:   Number of images per yielded batch.
        num_images:   Total images to return; must be a multiple of 1000.
    """
    import random
    from datasets import load_dataset

    per_class = num_images // 1000
    num_classes = 1000
    print(
        f"[cache_activations] Balanced sampling: {per_class} images × {num_classes} classes "
        f"= {per_class * num_classes} total"
    )

    print("[cache_activations] Loading dataset index (label column only)...")
    hf_ds = load_dataset(dataset_name, split=split)

    # Pull all labels in one vectorised call — fast with Arrow backing
    all_labels: list[int] = hf_ds["label"]

    # Build class → list-of-indices map
    class_indices: list[list[int]] = [[] for _ in range(num_classes)]
    for idx, lbl in enumerate(all_labels):
        class_indices[lbl].append(idx)

    # Sample per_class indices per class
    rng = random.Random(42)
    selected: list[int] = []
    for lbl in range(num_classes):
        pool = class_indices[lbl]
        chosen = rng.sample(pool, min(per_class, len(pool)))
        selected.extend(chosen)

    total = len(selected)
    print(f"[cache_activations] Balanced: {total} indices selected, building contiguous subset...")

    # hf_ds.select() writes a contiguous Arrow table — may internally sort indices
    # for read efficiency, so we apply a second shuffle on the subset row positions
    # to ensure shards saved to disk are class-interleaved, not class-clustered.
    subset = hf_ds.select(selected)
    row_order = list(range(total))
    rng.shuffle(row_order)
    subset = subset.select(row_order)
    print(f"[cache_activations] Subset ready (row-shuffled), iterating {total} images in batches of {batch_size}...")

    # Prefetch the next decoded batch on a background thread so GPU is never
    # idle waiting for JPEG decoding on the CPU.
    import queue
    import threading

    _SENTINEL = object()

    def _decode_worker(ds, bs, q):
        for arrow_batch in ds.iter(batch_size=bs, drop_last_batch=False):
            pil_images = []
            for img in arrow_batch["image"]:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)
            q.put(pil_images)
        q.put(_SENTINEL)

    prefetch_q: queue.Queue = queue.Queue(maxsize=4)
    t = threading.Thread(target=_decode_worker, args=(subset, batch_size, prefetch_q), daemon=True)
    t.start()

    while True:
        item = prefetch_q.get()
        if item is _SENTINEL:
            break
        yield item


def _hf_streaming_batches(dataset_name: str, split: str, batch_size: int, num_images: int | None):
    """Yield batches of PIL images from a HuggingFace streaming dataset.

    Each yielded item is a list of PIL.Image.Image objects. Preprocessing is
    intentionally deferred to extract_patch_activations so the adapter's
    processor is called exactly once.

    Args:
        dataset_name: HuggingFace dataset identifier (e.g. "imagenet-1k").
        split:        Dataset split (e.g. "train").
        batch_size:   Number of images per yielded batch.
        num_images:   Stop after this many images total (None = no limit).
    """
    from datasets import load_dataset

    hf_ds = load_dataset(dataset_name, split=split, streaming=True)

    pil_buffer: list = []
    images_seen = 0

    for example in hf_ds:
        if num_images is not None and images_seen >= num_images:
            break
        img = example["image"]
        if img.mode != "RGB":
            img = img.convert("RGB")
        pil_buffer.append(img)
        images_seen += 1

        if len(pil_buffer) == batch_size:
            yield pil_buffer
            pil_buffer = []
            if num_images is not None and images_seen >= num_images:
                break

    if pil_buffer:
        remaining = (num_images - (images_seen - len(pil_buffer))) if num_images is not None else len(pil_buffer)
        yield pil_buffer[:remaining]


def _hf_local_batches(hf_ds, batch_size: int, num_prefetch: int = 4):
    """Yield batches of PIL images from a locally loaded HuggingFace dataset.

    Uses a background thread to decode JPEG images ahead of the GPU so the
    forward pass never stalls waiting for CPU-side image decoding.

    Args:
        hf_ds:         A datasets.Dataset object (already loaded into memory/mmap).
        batch_size:    Number of images per yielded batch.
        num_prefetch:  Number of batches to decode ahead in background.
    """
    import queue
    import threading

    _SENTINEL = object()

    def _decode_worker(ds, bs, q):
        for arrow_batch in ds.iter(batch_size=bs, drop_last_batch=False):
            pil_images = []
            for img in arrow_batch["image"]:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                pil_images.append(img)
            q.put(pil_images)
        q.put(_SENTINEL)

    prefetch_q: queue.Queue = queue.Queue(maxsize=num_prefetch)
    t = threading.Thread(target=_decode_worker, args=(hf_ds, batch_size, prefetch_q), daemon=True)
    t.start()

    while True:
        item = prefetch_q.get()
        if item is _SENTINEL:
            break
        yield item


@torch.no_grad()
def run(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[cache_activations] device={device}  backbone={args.backbone}  layer={args.layer}")

    print(f"[cache_activations] Loading backbone '{args.backbone}'...")
    adapter = load_backbone(args.backbone, device=device)

    if args.use_hf_local or args.use_hf_streaming:
        source = "local parquet files" if args.use_hf_local else "HuggingFace"
        hf_home = os.environ.get("HF_HOME", "")
        total_images = args.num_images

        if args.balanced:
            per_class = args.num_images // 1000
            total_images = per_class * 1000
            print(
                f"[cache_activations] Balanced mode: {per_class} images/class × 1000 classes "
                f"= {total_images} total from {source} (HF_HOME={hf_home})"
            )
            batch_iter = _hf_balanced_batches(
                args.hf_dataset, args.hf_split, args.batch_size, args.num_images
            )
        elif args.use_hf_local:
            # Fast indexed path: Arrow-backed mmap, prefetched decoding
            from datasets import load_dataset
            print(f"[cache_activations] Loading indexed dataset '{args.hf_dataset}' (split={args.hf_split}) ...")
            hf_ds = load_dataset(args.hf_dataset, split=args.hf_split)
            if total_images is not None:
                hf_ds = hf_ds.select(range(min(total_images, len(hf_ds))))
            else:
                total_images = len(hf_ds)
            print(f"[cache_activations] {total_images} images, prefetched batches of {args.batch_size}")
            batch_iter = _hf_local_batches(hf_ds, args.batch_size)
        else:
            # True streaming from HuggingFace (no local copy)
            print(
                f"[cache_activations] Streaming '{args.hf_dataset}' (split={args.hf_split}) "
                f"from {source} (HF_HOME={hf_home})"
            )
            if total_images is not None:
                print(f"[cache_activations] Will process up to {total_images} images.")
            else:
                print("[cache_activations] No --num_images limit; will process full split.")
            batch_iter = _hf_streaming_batches(
                args.hf_dataset, args.hf_split, args.batch_size, args.num_images
            )
    else:
        print(f"[cache_activations] Building dataset from '{args.dataset_dir}'...")
        dataset = ImageFolderForCaching(root_dir=args.dataset_dir, processor=adapter.processor)
        if args.num_images is not None:
            n = min(args.num_images, len(dataset))
            dataset = Subset(dataset, list(range(n)))
            print(f"[cache_activations] Limiting to {n} images (--num_images={args.num_images}).")
        total_images = len(dataset)
        print(f"[cache_activations] Total images to process: {total_images}")
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=(device == "cuda"),
            drop_last=False,
        )
        batch_iter = loader

    os.makedirs(args.output_dir, exist_ok=True)

    welford = WelfordAccumulator()
    shard_buffer: list[torch.Tensor] = []
    shard_idx = 0
    images_processed = 0
    t_start = time.time()

    total_batches = (
        (total_images + args.batch_size - 1) // args.batch_size
        if total_images is not None else None
    )
    pbar = tqdm(batch_iter, total=total_batches, unit="batch",
                desc="Caching activations", dynamic_ncols=True)

    use_amp = device == "cuda"

    for batch in pbar:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            if isinstance(batch, list):
                activations = adapter.extract_patch_activations(batch, layer=args.layer)
            else:
                activations = adapter.extract_patch_activations(batch.to(device), layer=args.layer)
        activations_cpu = activations.cpu().float()

        B, P, D = activations_cpu.shape
        welford.update(activations_cpu.reshape(B * P, D))

        shard_buffer.append(activations_cpu)
        images_processed += B

        pbar.set_postfix(images=images_processed, img_s=f"{images_processed/(time.time()-t_start):.0f}")

        buffered = sum(t.shape[0] for t in shard_buffer)
        while buffered >= args.shard_size:
            full = torch.cat(shard_buffer, dim=0)
            shard_tensor = full[: args.shard_size]
            remainder = full[args.shard_size :]

            path = save_shard(shard_tensor, args.output_dir, shard_idx, half=args.half)
            elapsed = time.time() - t_start
            rate = images_processed / elapsed if elapsed > 0 else float("inf")
            if total_images is not None:
                remaining_images = total_images - images_processed
                eta = remaining_images / rate if rate > 0 else float("nan")
                eta_str = f"  |  ETA: {eta:.0f}s"
                total_str = f"/{total_images}"
            else:
                eta_str = ""
                total_str = ""
            print(
                f"  Saved {path}  |  images processed: {images_processed}{total_str}"
                + eta_str
            )
            shard_idx += 1

            if remainder.shape[0] > 0:
                shard_buffer = [remainder]
                buffered = remainder.shape[0]
            else:
                shard_buffer = []
                buffered = 0

    if shard_buffer:
        final_tensor = torch.cat(shard_buffer, dim=0)
        path = save_shard(final_tensor, args.output_dir, shard_idx, half=args.half)
        total_str = f"/{total_images}" if total_images is not None else ""
        print(
            f"  Saved final shard {path}  |  images processed: {images_processed}{total_str}"
        )

    stats = welford.to_dict()
    stats.update(
        {
            "num_images": images_processed,
            "patch_count": adapter.patch_count,
            "d_model": adapter.d_model,
            "backbone": args.backbone,
            "layer": args.layer,
        }
    )
    stats_path = os.path.join(args.output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[cache_activations] Wrote stats to {stats_path}")

    elapsed_total = time.time() - t_start
    print(
        f"[cache_activations] Done. {images_processed} images in {elapsed_total:.1f}s "
        f"({images_processed / elapsed_total:.1f} img/s). "
        f"{shard_idx + (1 if shard_buffer else 0)} shards written."
    )


if __name__ == "__main__":
    args = parse_args()
    run(args)
