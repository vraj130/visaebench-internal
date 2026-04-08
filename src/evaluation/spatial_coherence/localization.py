"""M6: Feature Localization Score (Moran's I spatial autocorrelation).

Measures whether SAE features activate on spatially coherent regions of the
ViT patch grid.  Moran's I quantifies spatial autocorrelation: values near
+1 indicate clustered activation (e.g., a contiguous object region), 0 means
random spatial arrangement, and -1 means a dispersed checkerboard pattern.

This is a novel application of Moran's I (Moran, 1950) to SAE feature
evaluation in vision transformers.

Algorithm:
  1. Build an 8-connectivity (queen contiguity) spatial weight matrix for the
     patch grid (14x14 or 16x16).
  2. For each image batch, encode patches through the SAE to get per-feature
     activation maps across the 2D grid.
  3. Compute Moran's I for every feature on every image simultaneously using
     vectorized matrix operations:  I = (N/S) * (x_c^T W x_c) / (x_c^T x_c)
  4. Average per feature across valid images (enough active patches, nonzero
     variance) and aggregate across features for the SAE-level score.
"""

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.base import MetricBase


class FeatureLocalizationScore(MetricBase):
    """M6: Feature localization via Moran's I spatial autocorrelation.

    For each SAE feature, computes Moran's I on the feature's activation
    pattern across the 2D patch grid of each image.  High Moran's I means
    the feature fires on spatially contiguous regions (e.g., an object),
    while low values indicate scattered or random activation.

    Results dict keys (under "results"):
        mean_morans_i          -- mean M6 across evaluable features
        median_morans_i        -- median M6
        std_morans_i           -- standard deviation
        percentile_25          -- 25th percentile
        percentile_75          -- 75th percentile
        num_evaluable_features -- features with >= min_valid_images valid images
        num_total_features     -- total SAE dictionary size
        frac_evaluable         -- fraction of evaluable features
        per_feature_scores     -- list of per-feature mean Moran's I (None if not evaluable)
    """

    def __init__(
        self,
        grid_h: int = 14,
        grid_w: int = 14,
        min_active_patches: int = 5,
        min_valid_images: int = 50,
        encode_batch_size: int = 512,
        batch_size_images: int = 64,
    ):
        self.grid_h = grid_h
        self.grid_w = grid_w
        self.min_active_patches = min_active_patches
        self.min_valid_images = min_valid_images
        self.encode_batch_size = encode_batch_size
        self.batch_size_images = batch_size_images

    # -- MetricBase interface --------------------------------------------------

    def evaluate(
        self,
        sae: torch.nn.Module,
        shard_paths: list[str],
        mean: torch.Tensor,
        std: float,
        device: str,
        **kwargs,
    ) -> dict:
        """Compute M6 feature localization scores.

        Required kwargs:
            dict_size (int): SAE dictionary size.
        """
        dict_size: int = kwargs["dict_size"]
        num_patches = self.grid_h * self.grid_w

        sae.eval()

        # Build spatial weight matrix (8-connectivity) once, move to device
        W = _build_weight_matrix(self.grid_h, self.grid_w).to(device)
        N = num_patches
        S = float(W.sum())

        # Running accumulators (float64 for precision over many images)
        morans_sum = np.zeros(dict_size, dtype=np.float64)
        valid_count = np.zeros(dict_size, dtype=np.int64)

        with torch.no_grad():
            for shard_path in tqdm(shard_paths, desc="M6 localization"):
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                N_shard, P, D = shard.shape
                assert P == num_patches, (
                    f"Shard patch count {P} != expected {num_patches} "
                    f"(grid {self.grid_h}x{self.grid_w})"
                )

                for g_start in range(0, N_shard, self.batch_size_images):
                    g_end = min(g_start + self.batch_size_images, N_shard)
                    group = shard[g_start:g_end]  # [G, P, D]
                    G = group.shape[0]

                    # Normalize and flatten for SAE encoding
                    tokens = group.reshape(G * P, D).float()
                    tokens = (tokens - mean) / std

                    # Encode through SAE in sub-batches, keep on device
                    code_chunks: list[torch.Tensor] = []
                    for s in range(0, tokens.shape[0], self.encode_batch_size):
                        batch = tokens[s : s + self.encode_batch_size].to(device)
                        _pre, codes, _xhat = sae(batch)
                        code_chunks.append(codes)
                        del batch, _pre, _xhat

                    all_codes = torch.cat(code_chunks, dim=0)  # [G*P, F]
                    del code_chunks
                    all_codes = all_codes.reshape(G, P, -1)  # [G, P, F]

                    # Vectorized Moran's I for all images × all features
                    mi, valid = _compute_morans_i_batch(
                        all_codes, W, N, S, self.min_active_patches,
                    )

                    # Accumulate on CPU
                    mi_np = mi.cpu().numpy()        # [G, F]
                    valid_np = valid.cpu().numpy()   # [G, F]
                    mi_np = np.where(valid_np, mi_np, 0.0)
                    morans_sum += mi_np.sum(axis=0)
                    valid_count += valid_np.astype(np.int64).sum(axis=0)

                    del all_codes, mi, valid, group, tokens

                del shard

        if device == "cuda":
            torch.cuda.empty_cache()

        # ── Per-feature average Moran's I ────────────────────────────────
        per_feature = np.full(dict_size, np.nan)
        evaluable_mask = valid_count >= self.min_valid_images
        per_feature[evaluable_mask] = (
            morans_sum[evaluable_mask] / valid_count[evaluable_mask]
        )

        evaluable_scores = per_feature[evaluable_mask]
        num_evaluable = int(evaluable_mask.sum())

        # NaN → None for JSON serialisation
        per_feature_list = [
            None if np.isnan(x) else float(x) for x in per_feature
        ]

        return {
            "config": {
                "grid_size": [self.grid_h, self.grid_w],
                "connectivity": 8,
                "min_active_patches": self.min_active_patches,
                "min_valid_images": self.min_valid_images,
            },
            "results": {
                "mean_morans_i": float(np.mean(evaluable_scores)) if num_evaluable > 0 else None,
                "median_morans_i": float(np.median(evaluable_scores)) if num_evaluable > 0 else None,
                "std_morans_i": float(np.std(evaluable_scores)) if num_evaluable > 0 else None,
                "percentile_25": float(np.percentile(evaluable_scores, 25)) if num_evaluable > 0 else None,
                "percentile_75": float(np.percentile(evaluable_scores, 75)) if num_evaluable > 0 else None,
                "num_evaluable_features": num_evaluable,
                "num_total_features": dict_size,
                "frac_evaluable": num_evaluable / dict_size if dict_size > 0 else 0.0,
                "per_feature_scores": per_feature_list,
            },
        }


# -- Vectorized Moran's I computation ----------------------------------------

def _compute_morans_i_batch(
    codes: torch.Tensor,
    W: torch.Tensor,
    N: int,
    S: float,
    min_active_patches: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Moran's I for a batch of images, all features at once.

    Uses a single matrix multiply ``W @ x_c`` (broadcast over the batch dim)
    to compute the spatial autocorrelation numerator for every feature
    simultaneously.  No Python loops over features or images.

    Args:
        codes: [B, P, F] activation codes on device.
        W: [P, P] spatial weight matrix on device.
        N: Number of patches (P).
        S: Sum of all weights in W.
        min_active_patches: Minimum nonzero patches for a feature to be valid.

    Returns:
        morans_i: [B, F] Moran's I values (garbage where ``valid`` is False).
        valid: [B, F] boolean mask of evaluable entries.
    """
    # Active patches per image per feature
    active_count = (codes != 0).sum(dim=1)  # [B, F]

    # Center codes per image
    x_bar = codes.mean(dim=1, keepdim=True)  # [B, 1, F]
    x_c = codes - x_bar                      # [B, P, F]

    # Denominator: sum of squared deviations
    denom = (x_c ** 2).sum(dim=1)            # [B, F]

    # Numerator: spatial autocorrelation via weight matrix
    # torch.matmul broadcasts W [P, P] across batch dim of x_c [B, P, F]
    #   result[b, p, f] = Σ_q W[p, q] * x_c[b, q, f]
    wx_c = torch.matmul(W, x_c)             # [B, P, F]
    numer = (x_c * wx_c).sum(dim=1)         # [B, F]

    # Moran's I  (clamp denom to avoid inf; invalid entries are masked out)
    morans_i = (N / S) * numer / denom.clamp(min=1e-10)

    # Validity: enough active patches AND non-zero variance
    valid = (active_count >= min_active_patches) & (denom > 1e-10)

    return morans_i, valid


# -- Spatial weight matrix construction ---------------------------------------

def _build_weight_matrix(grid_h: int, grid_w: int) -> torch.Tensor:
    """Build an 8-connectivity (queen contiguity) spatial weight matrix.

    Entry ``W[i, j] = 1`` if patches *i* and *j* are neighbors on the 2D
    grid (including diagonals).  The matrix is symmetric.

    Interior cells have 8 neighbours, edge cells 5, corner cells 3.

    Args:
        grid_h: Grid height (e.g. 14 for 14x14).
        grid_w: Grid width.

    Returns:
        Dense float32 tensor of shape ``[grid_h * grid_w, grid_h * grid_w]``.
    """
    num_patches = grid_h * grid_w
    W = torch.zeros(num_patches, num_patches, dtype=torch.float32)
    for r in range(grid_h):
        for c in range(grid_w):
            i = r * grid_w + c
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid_h and 0 <= nc < grid_w:
                        j = nr * grid_w + nc
                        W[i, j] = 1.0
    return W
