"""Run M2 downstream preservation on a single trained SAE.

Usage:
    python scripts/run_downstream_preservation.py \
        --sae_checkpoint /mnt/NAS/data/.../sae.pt \
        --sae_config /mnt/NAS/data/.../config.yaml \
        --activation_dir /mnt/NAS/data/.../activations_val/dinov2_vitb14/layer_11/ \
        --output_path results/raw/dinov2_vitb14_batchtopk_16x_k192_downstream.json
"""

import argparse
import json
import os

import torch
import yaml

from src.evaluation.concept_detection.sparse_probing import (
    discover_shards,
    load_imagenet_val_labels,
    load_stats,
)
from src.evaluation.reconstruction.downstream_preservation import DownstreamPreservation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M2 downstream preservation.")
    p.add_argument("--sae_checkpoint", type=str, required=True)
    p.add_argument("--sae_config", type=str, required=True)
    p.add_argument("--activation_dir", type=str, required=True)
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae(
    config_path: str, checkpoint_path: str, device: str,
    eval_batch_size: int = 512,
) -> tuple[torch.nn.Module, dict]:
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
        sae = BatchTopKSAE(input_shape=d_model, nb_concepts=dict_size,
                           top_k=k * eval_batch_size, device=device)
    elif arch == "topk":
        sae = TopKSAE(input_shape=d_model, nb_concepts=dict_size,
                      top_k=k, device=device)
    elif arch == "jumprelu":
        if JumpReLUSAE is None:
            raise ImportError("JumpReLUSAE not found.")
        sae = JumpReLUSAE(input_shape=d_model, nb_concepts=dict_size, device=device)
    else:
        raise ValueError(f"Unknown architecture: {arch}")

    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_threshold = state_dict.pop("_running_threshold", None)
    sae.load_state_dict(state_dict)

    if arch == "batchtopk":
        if saved_threshold is not None:
            sae.running_threshold = saved_threshold.to(device)
        else:
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
    encode_batch_size = 512

    print(f"[downstream] device={device}")

    sae, cfg = load_sae(args.sae_config, args.sae_checkpoint, device,
                        eval_batch_size=encode_batch_size)
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    print(f"[downstream] SAE: {cfg['architecture']} d={cfg['d_model']} "
          f"dict={dict_size} k={cfg['k']}")

    shard_paths = discover_shards(args.activation_dir)
    mean, std = load_stats(args.activation_dir)
    print(f"[downstream] {len(shard_paths)} val shards")

    print("[downstream] Loading ImageNet val labels...")
    labels = load_imagenet_val_labels()
    print(f"[downstream] {len(labels)} labels loaded")

    metric = DownstreamPreservation(encode_batch_size=encode_batch_size)
    results = metric.evaluate(
        sae=sae, shard_paths=shard_paths, mean=mean, std=std,
        device=device, dict_size=dict_size, labels=labels,
    )

    ckpt_dir = os.path.basename(os.path.dirname(args.sae_checkpoint))
    output = {
        "metric": "downstream_preservation",
        "backbone": cfg.get("backbone", "unknown"),
        "sae_config": ckpt_dir,
        **results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    print()
    print("=" * 50)
    print("M2 Downstream Preservation")
    print("=" * 50)
    print(f"  Accuracy (original):      {results['accuracy_original']:.4f}")
    print(f"  Accuracy (reconstructed): {results['accuracy_reconstructed']:.4f}")
    print(f"  Preservation ratio:       {results['preservation_ratio']:.4f}")
    print(f"  Accuracy gap:             {results['accuracy_gap']:.4f}")
    print(f"  Images:                   {results['num_images']}")
    print("=" * 50)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
