"""Run M3 sparse probing on a single trained SAE.

Usage:
    python scripts/run_sparse_probing.py \
        --sae_checkpoint /mnt/NAS/data/.../sae.pt \
        --sae_config /mnt/NAS/data/.../config.yaml \
        --activation_dir /mnt/NAS/data/.../activations_val/dinov2_vitb14/layer_11/ \
        --output_path results/raw/dinov2_vitb14_topk_16x_k192_sparse_probing.json
"""

import argparse
import json
import os

import torch
import yaml

from src.evaluation.concept_detection.sparse_probing import (
    SparseProbing,
    discover_shards,
    load_imagenet_val_labels,
    load_stats,
)

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M3 sparse probing on a trained SAE.")
    p.add_argument("--sae_checkpoint", type=str, required=True,
                   help="Path to sae.pt state dict.")
    p.add_argument("--sae_config", type=str, required=True,
                   help="Path to config.yaml (architecture, d_model, dict_size, k, etc.).")
    p.add_argument("--activation_dir", type=str, required=True,
                   help="Path to val activation shards (shard_*.pt + stats.json).")
    p.add_argument("--output_path", type=str, required=True,
                   help="Where to save the results JSON.")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: auto-detect cuda/cpu).")
    return p.parse_args()


def load_sae(
    config_path: str, checkpoint_path: str, device: str,
    eval_batch_size: int = 512,
) -> tuple[torch.nn.Module, dict]:
    """Reconstruct an SAE from config.yaml and load weights.

    For BatchTopKSAE, top_k = k * eval_batch_size so the per-sample
    sparsity matches what was used during training.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from overcomplete import BatchTopKSAE, TopKSAE
    try:
        from overcomplete import JumpReLUSAE
    except ImportError:
        JumpReLUSAE = None

    d_model = cfg["d_model"]
    dict_size = d_model * cfg["expansion_factor"]
    k = cfg["k"]
    arch = cfg["architecture"]

    if arch == "batchtopk":
        sae = BatchTopKSAE(
            input_shape=d_model,
            nb_concepts=dict_size,
            top_k=k * eval_batch_size,
            device=device,
        )
    elif arch == "topk":
        sae = TopKSAE(
            input_shape=d_model,
            nb_concepts=dict_size,
            top_k=k,
            device=device,
        )
    elif arch == "jumprelu":
        if JumpReLUSAE is None:
            raise ImportError("JumpReLUSAE not found in installed overcomplete version.")
        sae = JumpReLUSAE(
            input_shape=d_model,
            nb_concepts=dict_size,
            device=device,
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Restore running_threshold for BatchTopKSAE — it's saved under a
    # special key because it's not part of nn.Module state_dict.
    saved_threshold = state_dict.pop("_running_threshold", None)
    sae.load_state_dict(state_dict)

    if arch == "batchtopk":
        if saved_threshold is not None:
            sae.running_threshold = saved_threshold.to(device)
        else:
            # Old checkpoint without saved threshold — calibrate with a
            # dummy forward pass in training mode.
            sae.train()
            with torch.no_grad():
                dummy = torch.randn(eval_batch_size, d_model, device=device)
                sae(dummy)
            del dummy

    sae.eval()
    return sae, cfg


def main() -> None:
    args = parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sparse_probing] device={device}")

    # Load SAE — encode_batch_size must match between load_sae and SparseProbing
    encode_batch_size = 512
    sae, cfg = load_sae(args.sae_config, args.sae_checkpoint, device,
                        eval_batch_size=encode_batch_size)
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    print(f"[sparse_probing] SAE: {cfg['architecture']} d={cfg['d_model']} "
          f"dict={dict_size} k={cfg['k']}")

    # Load val shards and stats
    shard_paths = discover_shards(args.activation_dir)
    mean, std = load_stats(args.activation_dir)
    print(f"[sparse_probing] {len(shard_paths)} val shards from {args.activation_dir}")

    # Load labels
    print("[sparse_probing] Loading ImageNet val labels from HuggingFace...")
    labels = load_imagenet_val_labels()
    print(f"[sparse_probing] {len(labels)} labels loaded")

    # Run metric
    metric = SparseProbing(encode_batch_size=encode_batch_size)
    results = metric.evaluate(
        sae=sae,
        shard_paths=shard_paths,
        mean=mean,
        std=std,
        device=device,
        dict_size=dict_size,
        labels=labels,
    )

    # Build output
    # Derive SAE config name from checkpoint path (e.g. "topk_16x_k192")
    ckpt_dir = os.path.basename(os.path.dirname(args.sae_checkpoint))
    output = {
        "metric": "sparse_probing",
        "backbone": cfg.get("backbone", "unknown"),
        "sae_config": ckpt_dir,
        "sparse_probing": results,
    }

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print()
    print("=" * 50)
    print("M3 Sparse Probing Results")
    print("=" * 50)
    for k in sorted(k for k in results if k.startswith("k_")):
        print(f"  {k}: {results[k]:.4f}")
    print(f"  AUC: {results['auc']:.4f}")
    print(f"  Images: {results['num_images']}") 
    print(f"  Features: {results['num_features']}")
    print("=" * 50)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
