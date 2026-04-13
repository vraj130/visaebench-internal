"""Run M5 cross-domain generalization on a single trained SAE.

Usage:
    python scripts/run_cross_domain.py \
        --sae_checkpoint /mnt/NAS/data/.../sae.pt \
        --sae_config /mnt/NAS/data/.../config.yaml \
        --activation_dir /mnt/NAS/data/.../activations_val/dinov2_vitb14/layer_11/ \
        --backbone_name dinov2_vitb14 \
        --datasets eurosat \
        --output_path results/raw/dinov2_vitb14_batchtopk_16x_k192_cross_domain.json
"""

import argparse
import json
import os

import torch
import yaml

from src.evaluation.concept_detection.cross_domain import CrossDomainGeneralization
from src.evaluation.concept_detection.sparse_probing import load_stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run M5 cross-domain generalization on a trained SAE.",
    )
    p.add_argument(
        "--sae_checkpoint", type=str, required=True,
        help="Path to sae.pt state dict.",
    )
    p.add_argument(
        "--sae_config", type=str, required=True,
        help="Path to config.yaml (architecture, d_model, dict_size, k, etc.).",
    )
    p.add_argument(
        "--activation_dir", type=str, required=True,
        help="Path to ImageNet activation dir (for stats.json only — need mean/std).",
    )
    p.add_argument(
        "--backbone_name", type=str, required=True,
        help="Backbone name, e.g. dinov2_vitb14.",
    )
    p.add_argument(
        "--datasets", type=str, default="eurosat",
        help="Comma-separated OOD dataset names, e.g. 'eurosat' or 'eurosat,inaturalist'.",
    )
    p.add_argument(
        "--output_path", type=str, required=True,
        help="Where to save the results JSON.",
    )
    p.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect cuda/cpu).",
    )
    p.add_argument(
        "--max_images", type=int, default=10000,
        help="Max images per OOD dataset (default: 10000).",
    )
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
    print(f"[cross_domain] device={device}")

    # Load SAE
    encode_batch_size = 512
    sae, cfg = load_sae(
        args.sae_config, args.sae_checkpoint, device,
        eval_batch_size=encode_batch_size,
    )
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    print(f"[cross_domain] SAE: {cfg['architecture']} d={cfg['d_model']} "
          f"dict={dict_size} k={cfg['k']}")

    # Load ImageNet stats (mean/std for normalization)
    mean, std = load_stats(args.activation_dir)
    print(f"[cross_domain] Loaded ImageNet stats from {args.activation_dir}")

    # Parse dataset list
    dataset_names = [d.strip() for d in args.datasets.split(",")]
    print(f"[cross_domain] OOD datasets: {dataset_names}")

    # Run metric
    metric = CrossDomainGeneralization(
        encode_batch_size=encode_batch_size,
        max_images=args.max_images,
    )
    results = metric.evaluate(
        sae=sae,
        shard_paths=[],  # unused — OOD activations extracted live
        mean=mean,
        std=std,
        device=device,
        dict_size=dict_size,
        backbone_name=args.backbone_name,
        datasets=dataset_names,
    )

    # Build output
    ckpt_dir = os.path.basename(os.path.dirname(args.sae_checkpoint))
    output = {
        "metric": "cross_domain_generalization",
        "backbone": cfg.get("backbone", args.backbone_name),
        "sae_config": ckpt_dir,
    }
    output.update(results)

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print()
    print("=" * 60)
    print("M5 Cross-Domain Generalization Results")
    print("=" * 60)
    for ds_name, ds_results in results.items():
        print(f"\n  Dataset: {ds_name}")
        print(f"    Images: {ds_results['num_images']}, Classes: {ds_results['num_classes']}")
        print(f"    Raw baseline accuracy: {ds_results['raw_accuracy']:.4f}")
        for k in sorted(k for k in ds_results if k.startswith("sae_k")):
            print(f"    {k}: {ds_results[k]:.4f}")
        for k in sorted(k for k in ds_results if k.startswith("preservation_")):
            print(f"    {k}: {ds_results[k]:.3f}")
    print("=" * 60)
    print(f"Saved to {args.output_path}")


if __name__ == "__main__":
    main()
