"""Run M6 feature localization (Moran's I) on a single trained SAE.

Usage:
    python scripts/run_spatial_coherence.py \
        --sae_checkpoint /mnt/NAS/data/.../sae.pt \
        --sae_config /mnt/NAS/data/.../config.yaml \
        --activation_dir /mnt/NAS/data/.../activations_val/dinov2_vitb14/layer_11/ \
        --output_path results/raw/dinov2_vitb14_batchtopk_16x_k192_spatial_coherence.json
"""

import argparse
import json
import os

import torch
import yaml

from src.evaluation.concept_detection.sparse_probing import discover_shards, load_stats
from src.evaluation.spatial_coherence.localization import FeatureLocalizationScore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run M6 feature localization (Moran's I) on a trained SAE.",
    )
    p.add_argument("--sae_checkpoint", type=str, required=True,
                   help="Path to sae.pt state dict.")
    p.add_argument("--sae_config", type=str, required=True,
                   help="Path to config.yaml.")
    p.add_argument("--activation_dir", type=str, required=True,
                   help="Path to val activation shards (shard_*.pt + stats.json).")
    p.add_argument("--output_path", type=str, required=True,
                   help="Where to save results JSON.")
    p.add_argument("--grid_h", type=int, default=None,
                   help="Patch grid height (default: auto-detect from shard shape).")
    p.add_argument("--grid_w", type=int, default=None,
                   help="Patch grid width (default: auto-detect from shard shape).")
    p.add_argument("--min_active_patches", type=int, default=5,
                   help="Min nonzero patches per image for a feature to count (default: 5).")
    p.add_argument("--min_valid_images", type=int, default=50,
                   help="Min valid images for a feature to be evaluable (default: 50).")
    p.add_argument("--batch_size_images", type=int, default=64,
                   help="Images per Moran's I batch (default: 64).")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: auto-detect).")
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

    # Restore running_threshold for BatchTopKSAE (not in nn.Module state_dict)
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

    print(f"[spatial_coherence] device={device}")

    # Load SAE
    sae, cfg = load_sae(args.sae_config, args.sae_checkpoint, device,
                        eval_batch_size=encode_batch_size)
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    print(f"[spatial_coherence] SAE: {cfg['architecture']} d={cfg['d_model']} "
          f"dict={dict_size} k={cfg['k']}")

    # Load val shards and stats
    shard_paths = discover_shards(args.activation_dir)
    mean, std = load_stats(args.activation_dir)
    print(f"[spatial_coherence] {len(shard_paths)} val shards from {args.activation_dir}")

    # Auto-detect grid size from shard shape if not specified
    grid_h, grid_w = args.grid_h, args.grid_w
    if grid_h is None or grid_w is None:
        peek = torch.load(shard_paths[0], map_location="cpu", weights_only=True)
        P = peek.shape[1]
        del peek
        if P == 256:
            grid_h, grid_w = 16, 16
        elif P == 196:
            grid_h, grid_w = 14, 14
        else:
            raise ValueError(
                f"Cannot auto-detect grid size for {P} patches. "
                f"Pass --grid_h and --grid_w explicitly."
            )
    print(f"[spatial_coherence] grid: {grid_h}x{grid_w} = {grid_h * grid_w} patches")

    # Run metric
    metric = FeatureLocalizationScore(
        grid_h=grid_h,
        grid_w=grid_w,
        min_active_patches=args.min_active_patches,
        min_valid_images=args.min_valid_images,
        encode_batch_size=encode_batch_size,
        batch_size_images=args.batch_size_images,
    )
    results = metric.evaluate(
        sae=sae,
        shard_paths=shard_paths,
        mean=mean,
        std=std,
        device=device,
        dict_size=dict_size,
    )

    # Build output
    ckpt_dir = os.path.basename(os.path.dirname(args.sae_checkpoint))
    output = {
        "metric": "feature_localization_score",
        "backbone": cfg.get("backbone", "unknown"),
        "sae_config": ckpt_dir,
        **results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    r = results["results"]
    print()
    print("=" * 50)
    print("M6 Feature Localization Score (Moran's I)")
    print("=" * 50)
    print(f"  Mean Moran's I:     {r['mean_morans_i']:.4f}" if r["mean_morans_i"] is not None else "  Mean Moran's I:     N/A")
    print(f"  Median Moran's I:   {r['median_morans_i']:.4f}" if r["median_morans_i"] is not None else "  Median Moran's I:   N/A")
    print(f"  Std:                {r['std_morans_i']:.4f}" if r["std_morans_i"] is not None else "  Std:                N/A")
    print(f"  25th percentile:    {r['percentile_25']:.4f}" if r["percentile_25"] is not None else "  25th percentile:    N/A")
    print(f"  75th percentile:    {r['percentile_75']:.4f}" if r["percentile_75"] is not None else "  75th percentile:    N/A")
    print(f"  Evaluable features: {r['num_evaluable_features']} / {r['num_total_features']} "
          f"({r['frac_evaluable']:.1%})")
    print("=" * 50)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
