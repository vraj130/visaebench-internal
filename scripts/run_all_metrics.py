"""Run all 7 evaluation metrics (M1-M7) on a single SAE checkpoint.

Loads the SAE once, encodes val data once (shared across metrics that need
image codes), then runs each metric sequentially.  This avoids redundant
encoding passes — the shared encoding step produces:

  - image_codes:          [num_images, dict_size]  (max-pooled SAE codes)
  - pooled_original:      [num_images, 768]        (max-pooled normalised activations)
  - pooled_reconstructed: [num_images, 768]        (max-pooled SAE reconstructions)

Metrics that cannot use precomputed data (M1 FVU, M5 cross-domain, M6
localization) run their own shard passes.

Usage:
    python scripts/run_all_metrics.py \
        --sae_checkpoint /mnt/NAS/data/.../sae.pt \
        --sae_config /mnt/NAS/data/.../config.yaml \
        --activation_dir /mnt/NAS/data/.../activations_val/dinov2_vitb14/layer_11/ \
        --backbone_name dinov2_vitb14 \
        --output_dir /mnt/NAS/data/.../results/raw/ \
        --device cuda
"""

import argparse
import json
import os
import tempfile
import time
import traceback

import numpy as np
import torch
import yaml

from src.evaluation.concept_detection.sparse_probing import (
    discover_shards,
    load_imagenet_val_labels,
    load_stats,
)


# ---------------------------------------------------------------------------
# SAE loading (same pattern used across all run_*.py scripts)
# ---------------------------------------------------------------------------

def load_sae(
    config_path: str,
    checkpoint_path: str,
    device: str,
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


# ---------------------------------------------------------------------------
# Shared encoding: encode val data once for M2, M3, M4, M7
# ---------------------------------------------------------------------------

def encode_val_data(
    sae: torch.nn.Module,
    shard_paths: list[str],
    mean: torch.Tensor,
    std: float,
    device: str,
    dict_size: int,
    encode_batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Process all val shards through the SAE once.

    Returns:
        image_codes:          np.ndarray [num_images, dict_size]  — max-pooled SAE codes
        pooled_original:      np.ndarray [num_images, 768]        — max-pooled original activations
        pooled_reconstructed: np.ndarray [num_images, 768]        — max-pooled SAE reconstructions
    """
    from tqdm import tqdm

    sae.eval()
    d_model = mean.shape[0]

    # Count total images across shards for memmap pre-allocation
    shard_sizes = []
    for sp in tqdm(shard_paths, desc="Counting images"):
        s = torch.load(sp, map_location="cpu", weights_only=True)
        shard_sizes.append(s.shape[0])
        del s
    total_images = sum(shard_sizes)

    # Create memmaps to avoid accumulating multi-GB arrays in RAM
    tmp_codes = tempfile.NamedTemporaryFile(suffix="_codes.dat", delete=False)
    tmp_orig = tempfile.NamedTemporaryFile(suffix="_orig.dat", delete=False)
    tmp_recon = tempfile.NamedTemporaryFile(suffix="_recon.dat", delete=False)
    tmp_codes_path, tmp_orig_path, tmp_recon_path = (
        tmp_codes.name, tmp_orig.name, tmp_recon.name,
    )
    tmp_codes.close()
    tmp_orig.close()
    tmp_recon.close()

    codes_mmap = np.memmap(tmp_codes_path, dtype=np.float32, mode="w+",
                           shape=(total_images, dict_size))
    orig_mmap = np.memmap(tmp_orig_path, dtype=np.float32, mode="w+",
                          shape=(total_images, d_model))
    recon_mmap = np.memmap(tmp_recon_path, dtype=np.float32, mode="w+",
                           shape=(total_images, d_model))

    group_size = 512
    write_idx = 0

    pbar = tqdm(total=total_images, desc="Shared encoding", unit="img")

    with torch.no_grad():
        for shard_path in shard_paths:
            shard = torch.load(shard_path, map_location="cpu", weights_only=True)
            N, P, D = shard.shape

            for g_start in range(0, N, group_size):
                g_end = min(g_start + group_size, N)
                group = shard[g_start:g_end].float()  # [G, P, D]
                G = group.shape[0]

                # Normalize
                normed = (group - mean) / std  # [G, P, D]

                # Max-pool original normalised activations
                orig_pooled = normed.max(dim=1).values  # [G, D]

                # Flatten for SAE encoding
                tokens = normed.reshape(G * P, D)

                code_chunks: list[torch.Tensor] = []
                recon_chunks: list[torch.Tensor] = []

                for s in range(0, tokens.shape[0], encode_batch_size):
                    batch = tokens[s : s + encode_batch_size].to(device)
                    _pre, codes, x_hat = sae(batch)
                    code_chunks.append(codes.cpu())
                    recon_chunks.append(x_hat.cpu())
                    del batch, _pre, codes, x_hat

                # Codes: [G*P, dict_size] -> [G, P, dict_size] -> max-pool -> [G, dict_size]
                all_codes = torch.cat(code_chunks, dim=0).reshape(G, P, -1)
                pooled_codes = all_codes.max(dim=1).values.numpy()

                # Reconstructions: [G*P, D] -> [G, P, D] -> max-pool -> [G, D]
                all_recon = torch.cat(recon_chunks, dim=0).reshape(G, P, D)
                recon_pooled = all_recon.max(dim=1).values.numpy()

                codes_mmap[write_idx : write_idx + G] = pooled_codes
                orig_mmap[write_idx : write_idx + G] = orig_pooled.numpy()
                recon_mmap[write_idx : write_idx + G] = recon_pooled

                write_idx += G
                pbar.update(G)

                del group, normed, tokens, code_chunks, recon_chunks
                del all_codes, pooled_codes, all_recon, recon_pooled, orig_pooled

            del shard

    pbar.close()

    if device == "cuda":
        torch.cuda.empty_cache()

    # Flush and reopen as read-only
    codes_mmap.flush()
    orig_mmap.flush()
    recon_mmap.flush()

    image_codes = np.memmap(tmp_codes_path, dtype=np.float32, mode="r",
                            shape=(total_images, dict_size))
    pooled_original = np.memmap(tmp_orig_path, dtype=np.float32, mode="r",
                                shape=(total_images, d_model))
    pooled_reconstructed = np.memmap(tmp_recon_path, dtype=np.float32, mode="r",
                                     shape=(total_images, d_model))

    import atexit
    atexit.register(lambda: os.unlink(tmp_codes_path))
    atexit.register(lambda: os.unlink(tmp_orig_path))
    atexit.register(lambda: os.unlink(tmp_recon_path))

    return image_codes, pooled_original, pooled_reconstructed


# ---------------------------------------------------------------------------
# Per-metric runners
# ---------------------------------------------------------------------------

def run_m1_fvu(sae, shard_paths, mean, std, device, dict_size, encode_batch_size):
    """M1: FVU — needs its own shard-by-shard pass (incremental variance)."""
    from src.evaluation.reconstruction.fvu import FVUMetric
    metric = FVUMetric(batch_size=encode_batch_size)
    return metric.evaluate(
        sae=sae, shard_paths=shard_paths, mean=mean, std=std,
        device=device, dict_size=dict_size,
    )


def run_m2_downstream(sae, shard_paths, mean, std, device, dict_size,
                      labels, pooled_original, pooled_reconstructed,
                      encode_batch_size):
    """M2: Downstream Preservation — uses precomputed pooled data."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    num_images = pooled_original.shape[0]
    seed = 42
    test_size = 0.2

    if num_images < len(labels):
        print(f"[M2] Using first {num_images} of {len(labels)} labels")
        labels = labels[:num_images]

    idx_train, idx_test, y_train, y_test = train_test_split(
        np.arange(num_images), labels,
        test_size=test_size, stratify=labels, random_state=seed,
    )

    probe_steps = tqdm(
        [("original", pooled_original), ("reconstructed", pooled_reconstructed)],
        desc="M2 probes",
    )
    accuracies = {}
    for name, data in probe_steps:
        probe_steps.set_postfix(probe=name)
        clf = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                                 random_state=seed)
        clf.fit(data[idx_train], y_train)
        accuracies[name] = float(clf.score(data[idx_test], y_test))
        del clf

    acc_orig = accuracies["original"]
    acc_recon = accuracies["reconstructed"]
    preservation = acc_recon / acc_orig if acc_orig > 0 else 0.0

    return {
        "accuracy_original": acc_orig,
        "accuracy_reconstructed": acc_recon,
        "preservation_ratio": preservation,
        "accuracy_gap": acc_orig - acc_recon,
        "num_images": num_images,
    }


def run_m3_sparse_probing(image_codes, labels, dict_size):
    """M3: Sparse Probing — uses precomputed image codes."""
    from sklearn.feature_selection import f_classif
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from tqdm import tqdm

    k_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    seed = 42
    num_images = image_codes.shape[0]

    if num_images < len(labels):
        print(f"[M3] Using first {num_images} of {len(labels)} labels")
        labels = labels[:num_images]

    print("[M3] Ranking features by F-statistic...")
    f_scores, _ = f_classif(image_codes, labels)
    f_scores = np.nan_to_num(f_scores, nan=-np.inf)
    ranked_indices = np.argsort(f_scores)[::-1]
    print(f"[M3] Feature ranking done ({dict_size} features)")

    X_train, X_test, y_train, y_test = train_test_split(
        image_codes, labels,
        test_size=0.2, stratify=labels, random_state=seed,
    )

    results: dict = {}
    for k in tqdm(k_values, desc="M3 sparse probes"):
        top_k_idx = ranked_indices[:k]
        X_tr_k = X_train[:, top_k_idx]
        X_te_k = X_test[:, top_k_idx]

        clf = LogisticRegression(solver="lbfgs", max_iter=500, C=1.0,
                                 random_state=seed)
        clf.fit(X_tr_k, y_train)
        acc = float(clf.score(X_te_k, y_test))
        results[f"k_{k}_accuracy"] = acc

    ks = sorted(k_values)
    accs = [results[f"k_{k}_accuracy"] for k in ks]
    auc = float(np.trapezoid(accs, x=ks) / (ks[-1] - ks[0])) if len(ks) > 1 else accs[0]

    results["auc"] = auc
    results["num_images"] = num_images
    results["num_features"] = dict_size
    return results


def run_m4_monosemanticity(sae, shard_paths, mean, std, device, dict_size,
                           backbone_name, image_codes, max_features):
    """M4: Monosemanticity — uses precomputed image codes for top-k selection,
    but still needs to load cross-model backbone for embeddings."""
    from src.evaluation.concept_detection.monosemanticity import MonosemanticityScore

    metric = MonosemanticityScore(
        max_features=max_features,
        encode_batch_size=512,
    )

    # Monkey-patch _encode_shards to return precomputed codes instead of re-encoding
    original_encode = metric._encode_shards

    def _precomputed_encode(sae, shard_paths, mean, std, device, dict_size):
        return image_codes

    metric._encode_shards = _precomputed_encode

    result = metric.evaluate(
        sae=sae, shard_paths=shard_paths, mean=mean, std=std,
        device=device, dict_size=dict_size, backbone_name=backbone_name,
    )

    # Restore original method
    metric._encode_shards = original_encode
    return result


def run_m5_cross_domain(sae, shard_paths, mean, std, device, dict_size,
                        backbone_name, datasets):
    """M5: Cross-Domain — independent of val encoding, loads OOD data."""
    from src.evaluation.concept_detection.cross_domain import CrossDomainGeneralization
    metric = CrossDomainGeneralization()
    return metric.evaluate(
        sae=sae, shard_paths=shard_paths, mean=mean, std=std,
        device=device, dict_size=dict_size,
        backbone_name=backbone_name, datasets=datasets,
    )


def run_m6_localization(sae, shard_paths, mean, std, device, dict_size,
                        encode_batch_size):
    """M6: Localization — needs spatial (non-pooled) codes, runs its own pass."""
    from src.evaluation.spatial_coherence.localization import FeatureLocalizationScore

    # Determine grid size from the first shard
    first_shard = torch.load(shard_paths[0], map_location="cpu", weights_only=True)
    num_patches = first_shard.shape[1]
    del first_shard

    # Infer grid dimensions (square grids: 14x14=196 or 16x16=256)
    grid_side = int(num_patches ** 0.5)
    assert grid_side * grid_side == num_patches, (
        f"Non-square patch grid: {num_patches} patches"
    )

    metric = FeatureLocalizationScore(
        grid_h=grid_side,
        grid_w=grid_side,
        encode_batch_size=encode_batch_size,
    )
    return metric.evaluate(
        sae=sae, shard_paths=shard_paths, mean=mean, std=std,
        device=device, dict_size=dict_size,
    )


def run_m7_absorption(image_codes, labels, dict_size):
    """M7: Absorption — uses precomputed image codes."""
    from src.evaluation.disentanglement.absorption import FeatureAbsorption
    metric = FeatureAbsorption()
    # Pass precomputed_codes to skip re-encoding
    return metric.evaluate(
        sae=None, shard_paths=[], mean=None, std=0.0,
        device="cpu", dict_size=dict_size,
        labels=labels, precomputed_codes=image_codes,
    )


# ---------------------------------------------------------------------------
# Save result JSON
# ---------------------------------------------------------------------------

def save_result(result: dict, output_dir: str, backbone: str,
                sae_config_name: str, metric_name: str, metric_label: str):
    """Save one metric result as JSON."""
    os.makedirs(output_dir, exist_ok=True)
    filename = f"{backbone}_{sae_config_name}_{metric_name}.json"
    path = os.path.join(output_dir, filename)
    output = {
        "metric": metric_label,
        "backbone": backbone,
        "sae_config": sae_config_name,
        **result,
    }
    with open(path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run all 7 metrics (M1-M7) on a single SAE checkpoint.",
    )
    p.add_argument("--sae_checkpoint", type=str, required=True,
                   help="Path to sae.pt state dict.")
    p.add_argument("--sae_config", type=str, required=True,
                   help="Path to config.yaml.")
    p.add_argument("--activation_dir", type=str, required=True,
                   help="Path to val activation shards (shard_*.pt + stats.json).")
    p.add_argument("--backbone_name", type=str, required=True,
                   help="Backbone name (e.g., dinov2_vitb14).")
    p.add_argument("--output_dir", type=str, required=True,
                   help="Directory to save all metric result JSONs.")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device (default: cuda).")
    p.add_argument("--metrics", type=str, default="m1,m2,m3,m4,m5,m6,m7",
                   help="Comma-separated list of metrics to run (default: all).")
    p.add_argument("--max_features_m4", type=int, default=2048,
                   help="Max features to score for M4 monosemanticity (default: 2048).")
    p.add_argument("--m5_datasets", type=str, default="eurosat",
                   help="Comma-separated OOD datasets for M5 (default: eurosat).")
    return p.parse_args()


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}min"
    hours = minutes / 60
    return f"{hours:.1f}h"


def main() -> None:
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[warn] CUDA not available, falling back to CPU")
        device = "cpu"

    metrics_to_run = set(args.metrics.lower().split(","))
    m5_datasets = [d.strip() for d in args.m5_datasets.split(",")]
    encode_batch_size = 512

    print("=" * 60)
    print(f"Running metrics: {sorted(metrics_to_run)}")
    print(f"Backbone: {args.backbone_name}")
    print(f"Checkpoint: {args.sae_checkpoint}")
    print(f"Device: {device}")
    print("=" * 60)

    # ── Load SAE ─────────────────────────────────────────────────────────
    sae, cfg = load_sae(args.sae_config, args.sae_checkpoint, device,
                        eval_batch_size=encode_batch_size)
    dict_size = cfg["d_model"] * cfg["expansion_factor"]
    backbone_name = args.backbone_name
    sae_config_name = os.path.basename(os.path.dirname(args.sae_checkpoint))

    print(f"SAE: {cfg['architecture']} d={cfg['d_model']} "
          f"dict={dict_size} k={cfg['k']}")

    # ── Discover shards and load stats ───────────────────────────────────
    shard_paths = discover_shards(args.activation_dir)
    mean, std = load_stats(args.activation_dir)
    print(f"{len(shard_paths)} val shards from {args.activation_dir}")

    # ── Load labels (needed by M2, M3, M7) ───────────────────────────────
    needs_labels = metrics_to_run & {"m2", "m3", "m7"}
    labels = None
    if needs_labels:
        print("Loading ImageNet val labels...")
        labels = load_imagenet_val_labels()
        print(f"{len(labels)} labels loaded")

    # ── Shared encoding (needed by M2, M3, M4, M7) ──────────────────────
    needs_shared_encoding = metrics_to_run & {"m2", "m3", "m4", "m7"}
    image_codes = None
    pooled_original = None
    pooled_reconstructed = None

    if needs_shared_encoding:
        print("\n--- Shared Encoding Pass ---")
        t0 = time.time()
        image_codes, pooled_original, pooled_reconstructed = encode_val_data(
            sae=sae,
            shard_paths=shard_paths,
            mean=mean,
            std=std,
            device=device,
            dict_size=dict_size,
            encode_batch_size=encode_batch_size,
        )
        enc_time = time.time() - t0
        print(f"Shared encoding: {image_codes.shape[0]} images in {format_time(enc_time)}")
        print(f"  image_codes:          {image_codes.shape}")
        print(f"  pooled_original:      {pooled_original.shape}")
        print(f"  pooled_reconstructed: {pooled_reconstructed.shape}")

    # ── Run each metric ──────────────────────────────────────────────────
    from tqdm import tqdm

    summary: dict[str, tuple[str, float]] = {}  # metric -> (headline, elapsed)
    all_results: dict[str, dict] = {}

    # Build ordered list of metrics to run for the overall progress bar
    metric_order = ["m1", "m2", "m3", "m4", "m5", "m6", "m7"]
    metric_labels = {
        "m1": "M1 FVU", "m2": "M2 Downstream", "m3": "M3 Sparse Probing",
        "m4": "M4 Monosemanticity", "m5": "M5 Cross-Domain",
        "m6": "M6 Localization", "m7": "M7 Absorption",
    }
    active_metrics = [m for m in metric_order if m in metrics_to_run]
    overall_pbar = tqdm(active_metrics, desc="Overall progress", unit="metric",
                        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} metrics [{elapsed}<{remaining}]")

    # M1: FVU
    if "m1" in metrics_to_run:
        overall_pbar.set_description("M1 FVU")
        print("\n--- M1: FVU ---")
        t0 = time.time()
        try:
            result = run_m1_fvu(sae, shard_paths, mean, std, device,
                                dict_size, encode_batch_size)
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m1_fvu", "fvu")
            headline = f"{result['fvu']:.4f} (L0={result['l0']:.1f}, dead={result['dead_pct']:.1f}%)"
            summary["M1 FVU"] = (headline, elapsed)
            all_results["m1"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m1_fvu", "fvu")
            summary["M1 FVU"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    # M2: Downstream Preservation
    if "m2" in metrics_to_run:
        overall_pbar.set_description("M2 Downstream")
        print("\n--- M2: Downstream Preservation ---")
        t0 = time.time()
        try:
            result = run_m2_downstream(
                sae, shard_paths, mean, std, device, dict_size,
                labels.copy(), pooled_original, pooled_reconstructed,
                encode_batch_size,
            )
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m2_downstream", "downstream_preservation")
            headline = f"ratio={result['preservation_ratio']:.3f} (orig={result['accuracy_original']:.3f}, recon={result['accuracy_reconstructed']:.3f})"
            summary["M2 Preservation"] = (headline, elapsed)
            all_results["m2"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m2_downstream", "downstream_preservation")
            summary["M2 Preservation"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    # M3: Sparse Probing
    if "m3" in metrics_to_run:
        overall_pbar.set_description("M3 Sparse Probing")
        print("\n--- M3: Sparse Probing ---")
        t0 = time.time()
        try:
            result = run_m3_sparse_probing(image_codes, labels.copy(), dict_size)
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m3_sparse_probing", "sparse_probing")
            headline = f"AUC={result['auc']:.3f}"
            if "k_32_accuracy" in result:
                headline += f" (k32={result['k_32_accuracy']:.3f})"
            if "k_128_accuracy" in result:
                headline += f" (k128={result['k_128_accuracy']:.3f})"
            summary["M3 Sparse Probing"] = (headline, elapsed)
            all_results["m3"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m3_sparse_probing", "sparse_probing")
            summary["M3 Sparse Probing"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    # M4: Monosemanticity
    if "m4" in metrics_to_run:
        overall_pbar.set_description("M4 Monosemanticity")
        print("\n--- M4: Monosemanticity ---")
        t0 = time.time()
        try:
            result = run_m4_monosemanticity(
                sae, shard_paths, mean, std, device, dict_size,
                backbone_name, image_codes, args.max_features_m4,
            )
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m4_monosemanticity", "monosemanticity")
            ms = result.get("monosemanticity_score")
            headline = f"{ms:.3f}" if ms is not None else "N/A"
            summary["M4 Monosemanticity"] = (headline, elapsed)
            all_results["m4"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m4_monosemanticity", "monosemanticity")
            summary["M4 Monosemanticity"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    # M5: Cross-Domain
    if "m5" in metrics_to_run:
        overall_pbar.set_description("M5 Cross-Domain")
        print("\n--- M5: Cross-Domain ---")
        t0 = time.time()
        try:
            result = run_m5_cross_domain(
                sae, shard_paths, mean, std, device, dict_size,
                backbone_name, m5_datasets,
            )
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m5_cross_domain", "cross_domain")
            # Build headline from first dataset
            ds_name = m5_datasets[0]
            if ds_name in result and "preservation_k128" in result[ds_name]:
                headline = f"{ds_name} preservation_k128={result[ds_name]['preservation_k128']:.3f}"
            elif ds_name in result and "raw_accuracy" in result[ds_name]:
                headline = f"{ds_name} raw={result[ds_name]['raw_accuracy']:.3f}"
            else:
                headline = "done"
            summary["M5 Cross-Domain"] = (headline, elapsed)
            all_results["m5"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m5_cross_domain", "cross_domain")
            summary["M5 Cross-Domain"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    # M6: Localization
    if "m6" in metrics_to_run:
        overall_pbar.set_description("M6 Localization")
        print("\n--- M6: Feature Localization ---")
        t0 = time.time()
        try:
            result = run_m6_localization(
                sae, shard_paths, mean, std, device, dict_size,
                encode_batch_size,
            )
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m6_localization", "localization")
            mean_mi = result.get("results", {}).get("mean_morans_i")
            headline = f"{mean_mi:.3f}" if mean_mi is not None else "N/A"
            summary["M6 Localization"] = (headline, elapsed)
            all_results["m6"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m6_localization", "localization")
            summary["M6 Localization"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    # M7: Absorption
    if "m7" in metrics_to_run:
        overall_pbar.set_description("M7 Absorption")
        print("\n--- M7: Feature Absorption ---")
        t0 = time.time()
        try:
            result = run_m7_absorption(image_codes, labels.copy(), dict_size)
            elapsed = time.time() - t0
            save_result(result, args.output_dir, backbone_name,
                        sae_config_name, "m7_absorption", "feature_absorption")
            headline = f"rate={result['absorption_rate']:.3f} (tests={result['num_tests']})"
            summary["M7 Absorption"] = (headline, elapsed)
            all_results["m7"] = result
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  ERROR: {e}")
            traceback.print_exc()
            save_result({"error": str(e)}, args.output_dir, backbone_name,
                        sae_config_name, "m7_absorption", "feature_absorption")
            summary["M7 Absorption"] = (f"FAILED: {e}", elapsed)
        overall_pbar.update(1)

    overall_pbar.set_description("Done")
    overall_pbar.close()

    # ── Summary ──────────────────────────────────────────────────────────
    total_time = sum(t for _, t in summary.values())
    print()
    print("=" * 60)
    print(f"Evaluation Summary: {backbone_name} / {sae_config_name}")
    print("=" * 60)
    for metric_name, (headline, elapsed) in summary.items():
        print(f"  {metric_name:25s} {headline:40s} ({format_time(elapsed)})")
    print("-" * 60)
    print(f"  {'Total':25s} {'':40s} ({format_time(total_time)})")
    print("=" * 60)


if __name__ == "__main__":
    main()
