"""Run M4 monosemanticity score on a single trained SAE.

Usage:
    python scripts/run_monosemanticity.py \
        --sae_checkpoint /mnt/NAS/data/.../sae.pt \
        --sae_config /mnt/NAS/data/.../config.yaml \
        --activation_dir /mnt/NAS/data/.../activations_val/dinov2_vitb14/layer_11/ \
        --backbone_name dinov2_vitb14 \
        --output_path results/raw/dinov2_vitb14_batchtopk_16x_k192_monosemanticity.json
"""

import argparse
import json
import os

import torch
import yaml

from src.evaluation.concept_detection.monosemanticity import (
    CROSS_MODEL_MAP,
    MonosemanticityScore,
)
from src.evaluation.concept_detection.sparse_probing import discover_shards, load_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run M4 monosemanticity score on a trained SAE.")
    p.add_argument("--sae_checkpoint", type=str, required=True)
    p.add_argument("--sae_config", type=str, required=True)
    p.add_argument("--activation_dir", type=str, required=True,
                   help="Path to val activation shards.")
    p.add_argument("--backbone_name", type=str, required=True,
                   choices=list(CROSS_MODEL_MAP.keys()),
                   help="Backbone whose SAE is being evaluated.")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--max_features", type=int, default=2048,
                   help="Max features to score (default 2048).")
    p.add_argument("--top_k_images", type=int, default=16,
                   help="Top-k images per feature (default 16).")
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def load_sae(
    config_path: str, checkpoint_path: str, device: str,
    eval_batch_size: int = 512,
) -> tuple[torch.nn.Module, dict]:
    """Reconstruct SAE from config.yaml and load weights."""
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

    print(f"[monosemanticity] device={device}")
    print(f"[monosemanticity] cross-model: {args.backbone_name} → {CROSS_MODEL_MAP[args.backbone_name]}")

    # Load SAE
    sae, cfg = load_sae(args.sae_config, args.sae_checkpoint, device,
                        eval_batch_size=encode_batch_size)
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    print(f"[monosemanticity] SAE: {cfg['architecture']} d={cfg['d_model']} "
          f"dict={dict_size} k={cfg['k']}")

    # Load val shards and stats
    shard_paths = discover_shards(args.activation_dir)
    mean, std = load_stats(args.activation_dir)
    print(f"[monosemanticity] {len(shard_paths)} val shards")

    # Run metric
    metric = MonosemanticityScore(
        top_k_images=args.top_k_images,
        max_features=args.max_features,
        encode_batch_size=encode_batch_size,
    )
    results = metric.evaluate(
        sae=sae,
        shard_paths=shard_paths,
        mean=mean,
        std=std,
        device=device,
        dict_size=dict_size,
        backbone_name=args.backbone_name,
    )

    # Build output
    ckpt_dir = os.path.basename(os.path.dirname(args.sae_checkpoint))
    output = {
        "metric": "monosemanticity_score",
        "backbone": cfg.get("backbone", args.backbone_name),
        "sae_config": ckpt_dir,
        **results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print()
    print("=" * 50)
    print("M4 Monosemanticity Score")
    print("=" * 50)
    ms = results["monosemanticity_score"]
    print(f"  MS (mean):     {ms:.4f}" if ms is not None else "  MS (mean):     N/A")
    print(f"  MS (median):   {results['ms_median']:.4f}" if results["ms_median"] is not None else "  MS (median):   N/A")
    print(f"  MS (std):      {results['ms_std']:.4f}" if results["ms_std"] is not None else "  MS (std):      N/A")
    print(f"  Features scored: {results['num_features_scored']}")
    print(f"  Dead features:   {results['num_dead_features']}")
    print(f"  Cross-model:     {results['cross_model']}")
    print("=" * 50)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
