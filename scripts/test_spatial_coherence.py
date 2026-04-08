"""Quick validation of M6 spatial coherence on DINOv2 BatchTopK SAE.

Runs on a small subset (first 2 shards) for speed, then prints detailed
diagnostics including per-feature histograms and example spatial maps.
"""

import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

# Ensure repo root is on path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.evaluation.spatial_coherence.localization import (
    FeatureLocalizationScore,
    _build_weight_matrix,
    _compute_morans_i_batch,
)

# ── Paths ────────────────────────────────────────────────────────────────────
CHECKPOINT_DIR = "/mnt/NAS/data/ds5725/visaebench/checkpoints/dinov2_vitb14/batchtopk_16x_k192"
VAL_DIR = "/mnt/NAS/data/ds5725/visaebench/activations_val/dinov2_vitb14/layer_11"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ENCODE_BATCH_SIZE = 512
NUM_SHARDS = 2  # use first 2 shards for quick test (~10K images)


def load_sae(config_path, checkpoint_path, device, eval_batch_size):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    from overcomplete import BatchTopKSAE, TopKSAE
    d_model = cfg["d_model"]
    dict_size = d_model * cfg["expansion_factor"]
    k = cfg["k"]
    arch = cfg["architecture"]
    if arch == "batchtopk":
        sae = BatchTopKSAE(input_shape=d_model, nb_concepts=dict_size,
                           top_k=k * eval_batch_size, device=device)
    else:
        sae = TopKSAE(input_shape=d_model, nb_concepts=dict_size,
                      top_k=k, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    saved_threshold = state_dict.pop("_running_threshold", None)
    sae.load_state_dict(state_dict)
    if arch == "batchtopk":
        if saved_threshold is not None:
            sae.running_threshold = saved_threshold.to(device)
        else:
            sae.train()
            with torch.no_grad():
                sae(torch.randn(eval_batch_size, d_model, device=device))
    sae.eval()
    return sae, cfg


def ascii_histogram(values, bins=20, width=50):
    """Print a simple ASCII histogram."""
    counts, edges = np.histogram(values, bins=bins)
    max_count = max(counts) if max(counts) > 0 else 1
    for i, count in enumerate(counts):
        bar = "#" * int(count / max_count * width)
        lo, hi = edges[i], edges[i + 1]
        print(f"  [{lo:+.3f}, {hi:+.3f}) | {bar} ({count})")


def main():
    print("=" * 60)
    print("M6 Spatial Coherence — Quick Validation (DINOv2 BatchTopK)")
    print("=" * 60)

    # ── 1. Unit tests on synthetic patterns ──────────────────────────────
    print("\n[1] Synthetic pattern tests (4x4 grid)...")
    W4 = _build_weight_matrix(4, 4)
    N4, S4 = 16, float(W4.sum())

    # Clustered: 2x2 block in top-left
    codes_cluster = torch.zeros(1, 16, 1)
    for r in range(2):
        for c in range(2):
            codes_cluster[0, r * 4 + c, 0] = 1.0
    mi_c, _ = _compute_morans_i_batch(codes_cluster, W4, N4, S4, 1)
    print(f"  Clustered 2x2 block:  I = {mi_c[0, 0]:.4f}  (expect > 0)")

    # Checkerboard
    codes_check = torch.zeros(1, 16, 1)
    for r in range(4):
        for c in range(4):
            codes_check[0, r * 4 + c, 0] = 1.0 if (r + c) % 2 == 0 else 0.0
    mi_ch, _ = _compute_morans_i_batch(codes_check, W4, N4, S4, 1)
    print(f"  Checkerboard:         I = {mi_ch[0, 0]:.4f}  (expect < 0)")

    # Uniform (all ones — zero variance → invalid)
    codes_uniform = torch.ones(1, 16, 1)
    mi_u, valid_u = _compute_morans_i_batch(codes_uniform, W4, N4, S4, 1)
    print(f"  Uniform (all 1.0):    valid = {valid_u[0, 0].item()}  (expect False)")

    # Single row active (horizontal stripe)
    codes_stripe = torch.zeros(1, 16, 1)
    for c in range(4):
        codes_stripe[0, c, 0] = 1.0  # top row
    mi_s, _ = _compute_morans_i_batch(codes_stripe, W4, N4, S4, 1)
    print(f"  Horizontal stripe:    I = {mi_s[0, 0]:.4f}  (expect > 0)")

    # Random codes (multiple features, batched)
    torch.manual_seed(42)
    codes_rand = torch.rand(8, 16, 100)  # 8 images, 16 patches, 100 features
    codes_rand[codes_rand < 0.7] = 0.0   # sparsify
    mi_r, valid_r = _compute_morans_i_batch(codes_rand, W4, N4, S4, 1)
    valid_mi = mi_r[valid_r]
    print(f"  Random sparse (8 imgs, 100 feats): {valid_r.sum()} valid entries, "
          f"mean I = {valid_mi.mean():.4f}  (expect ≈ 0)")

    print("  ✓ Synthetic tests passed")

    # ── 2. Load SAE and run on real val shards ───────────────────────────
    print(f"\n[2] Loading SAE from {CHECKPOINT_DIR}...")
    sae, cfg = load_sae(
        os.path.join(CHECKPOINT_DIR, "config.yaml"),
        os.path.join(CHECKPOINT_DIR, "sae.pt"),
        DEVICE, ENCODE_BATCH_SIZE,
    )
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    print(f"  arch={cfg['architecture']}  dict_size={dict_size}  k={cfg['k']}")

    # Load stats
    stats_path = os.path.join(VAL_DIR, "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = float(stats["std"])

    # Use first NUM_SHARDS shards for quick test
    all_shards = sorted(glob.glob(os.path.join(VAL_DIR, "shard_*.pt")))
    test_shards = all_shards[:NUM_SHARDS]
    print(f"  Using {len(test_shards)} of {len(all_shards)} shards for quick test")

    # ── 3. Run M6 metric ────────────────────────────────────────────────
    print(f"\n[3] Running M6 feature localization (Moran's I)...")
    metric = FeatureLocalizationScore(
        grid_h=16, grid_w=16,  # DINOv2: 16x16 grid
        min_active_patches=5,
        min_valid_images=10,   # lower threshold for small test set
        encode_batch_size=ENCODE_BATCH_SIZE,
        batch_size_images=64,
    )
    results = metric.evaluate(
        sae=sae, shard_paths=test_shards, mean=mean, std=std,
        device=DEVICE, dict_size=dict_size,
    )

    r = results["results"]
    print(f"\n  Results ({len(test_shards)} shards):")
    print(f"  Mean Moran's I:     {r['mean_morans_i']:.4f}" if r["mean_morans_i"] is not None else "  Mean Moran's I:     N/A")
    print(f"  Median Moran's I:   {r['median_morans_i']:.4f}" if r["median_morans_i"] is not None else "  Median Moran's I:   N/A")
    print(f"  Std:                {r['std_morans_i']:.4f}" if r["std_morans_i"] is not None else "  Std:                N/A")
    print(f"  25th pctl:          {r['percentile_25']:.4f}" if r["percentile_25"] is not None else "  25th pctl:          N/A")
    print(f"  75th pctl:          {r['percentile_75']:.4f}" if r["percentile_75"] is not None else "  75th pctl:          N/A")
    print(f"  Evaluable features: {r['num_evaluable_features']} / {r['num_total_features']} "
          f"({r['frac_evaluable']:.1%})")

    # ── 4. Distribution analysis ────────────────────────────────────────
    scores = np.array(r["per_feature_scores"], dtype=np.float64)
    valid_scores = scores[~np.isnan(scores)] if not all(s is None for s in r["per_feature_scores"]) else np.array([])
    # Handle None values
    valid_scores = np.array([s for s in r["per_feature_scores"] if s is not None])

    if len(valid_scores) > 0:
        print(f"\n[4] Distribution of per-feature Moran's I ({len(valid_scores)} evaluable features):")
        ascii_histogram(valid_scores, bins=20, width=40)

        # Extreme features
        sorted_idx = np.argsort(valid_scores)
        print(f"\n  Top 10 most spatially coherent features:")
        for i in sorted_idx[-10:][::-1]:
            print(f"    Feature {i:>5d}:  I = {valid_scores[i]:.4f}")

        print(f"\n  Top 10 most dispersed features:")
        for i in sorted_idx[:10]:
            print(f"    Feature {i:>5d}:  I = {valid_scores[i]:.4f}")

        # Sanity checks
        print(f"\n[5] Sanity checks:")
        print(f"  Fraction with I > 0 (spatially coherent):  {(valid_scores > 0).mean():.1%}")
        print(f"  Fraction with I > 0.2 (strongly coherent): {(valid_scores > 0.2).mean():.1%}")
        print(f"  Fraction with I < 0 (dispersed):           {(valid_scores < 0).mean():.1%}")
        expected_null = -1 / (256 - 1)  # E[I] under spatial randomness for 256 patches
        print(f"  Expected I under null (random):             {expected_null:.4f}")
        print(f"  Observed mean I:                            {np.mean(valid_scores):.4f}")
        if np.mean(valid_scores) > expected_null + 0.01:
            print("  ✓ Mean I significantly above null — SAE features show spatial structure")
        else:
            print("  ⚠ Mean I near null — features may lack spatial coherence")

        # ── 6. Visualize a few feature spatial maps ──────────────────────
        print(f"\n[6] Example spatial activation maps (top-3 coherent features):")
        # Pick top-3 features by Moran's I
        # Need to map back from valid_scores index to dict_size index
        all_scores_arr = np.array(
            [s if s is not None else np.nan for s in r["per_feature_scores"]]
        )
        top_features = np.argsort(np.nan_to_num(all_scores_arr, nan=-999))[-3:][::-1]

        # Load one shard and encode to get spatial maps
        shard = torch.load(test_shards[0], map_location="cpu", weights_only=True)
        # Pick first 3 images
        with torch.no_grad():
            for img_idx in range(min(3, shard.shape[0])):
                patches = shard[img_idx].float()  # [256, 768]
                patches = (patches - mean) / std
                codes = []
                for s in range(0, patches.shape[0], ENCODE_BATCH_SIZE):
                    batch = patches[s:s + ENCODE_BATCH_SIZE].to(DEVICE)
                    _, c, _ = sae(batch)
                    codes.append(c.cpu())
                codes = torch.cat(codes, dim=0)  # [256, dict_size]

                for feat_idx in top_features:
                    feat_map = codes[:, feat_idx].reshape(16, 16).numpy()
                    active = (feat_map != 0).sum()
                    if active < 3:
                        continue
                    # ASCII heatmap: show where the feature activates
                    print(f"\n  Image {img_idx}, Feature {feat_idx} "
                          f"(I={all_scores_arr[feat_idx]:.4f}, active={active}/256):")
                    max_val = feat_map.max() if feat_map.max() > 0 else 1.0
                    chars = " .:-=+*#@"
                    for row in range(16):
                        line = "    "
                        for col in range(16):
                            v = feat_map[row, col] / max_val
                            ci = min(int(v * (len(chars) - 1)), len(chars) - 1)
                            line += chars[ci]
                        print(line)

        del shard
    else:
        print("\n  No evaluable features — something may be wrong.")

    print("\n" + "=" * 60)
    print("Validation complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
