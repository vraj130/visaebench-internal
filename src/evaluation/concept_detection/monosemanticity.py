"""M4: Monosemanticity Score (MS) based on Pach et al. (2025).

Measures whether each SAE feature responds to semantically coherent images.
For each feature, find the top-k most-activating images, embed them using a
*cross-model* backbone (different from the one that produced the SAE), and
compute the activation-weighted mean pairwise cosine similarity of embeddings.

Reference:
    Pach et al. (2025), "Sparse Autoencoders Learn Monosemantic Features in
    Vision-Language Models", NeurIPS 2025.
    https://github.com/ExplainableML/sae-for-vlm/blob/main/metric.py
"""

import glob
import json
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.evaluation.base import MetricBase

# Cross-model map: SAE backbone → embedding backbone for MS computation.
# The embedding backbone must differ from the SAE backbone to avoid
# circularity (measuring whether the SAE preserves *its own* features).
CROSS_MODEL_MAP = {
    "clip_vitb16": "dinov2_vitb14",
    "dinov2_vitb14": "clip_vitb16",
    "siglip_vitb16": "dinov2_vitb14",
    "mae_vitb16": "clip_vitb16",
    "deit_vitb16": "clip_vitb16",
}


class MonosemanticityScore(MetricBase):
    """M4: Monosemanticity Score.

    For each SAE feature j:
      1. Find the top-k images with highest (max-pooled) activation.
      2. Normalise activations to [0, 1] (min-max across all images).
      3. Embed the top-k images with a cross-model backbone.
      4. Compute MS_j = Σ_{i<k} w_i·w_k·cos(e_i,e_k) / Σ_{i<k} w_i·w_k
         where w_i is the normalised activation weight.

    The overall MS is the mean of MS_j across scored features.
    """

    def __init__(
        self,
        top_k_images: int = 16,
        max_features: int = 2048,
        encode_batch_size: int = 512,
    ):
        self.top_k_images = top_k_images
        self.max_features = max_features
        self.encode_batch_size = encode_batch_size

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
        """Compute M4 monosemanticity score.

        Required kwargs:
            dict_size (int): SAE dictionary size.
            backbone_name (str): Name of the backbone whose SAE is evaluated.
        Optional kwargs:
            seed (int): Random seed for feature subsampling (default 42).
        """
        dict_size: int = kwargs["dict_size"]
        backbone_name: str = kwargs["backbone_name"]
        seed: int = kwargs.get("seed", 42)

        cross_model_name = CROSS_MODEL_MAP[backbone_name]

        # ── Step 1: Encode all val images → [num_images, dict_size] ──────
        print(f"[M4] Encoding val images through SAE...")
        image_codes = self._encode_shards(sae, shard_paths, mean, std, device, dict_size)
        num_images = image_codes.shape[0]
        print(f"[M4] {num_images} images encoded, dict_size={dict_size}")

        # ── Step 2: Identify live features and subsample ─────────────────
        # A feature is "dead" if it never activates (max activation == 0)
        max_per_feature = np.max(image_codes, axis=0)  # [dict_size]
        live_mask = max_per_feature > 0
        live_indices = np.where(live_mask)[0]
        num_dead = int((~live_mask).sum())
        print(f"[M4] Live features: {len(live_indices)}, dead: {num_dead}")

        rng = np.random.RandomState(seed)
        if len(live_indices) > self.max_features:
            sampled = rng.choice(live_indices, size=self.max_features, replace=False)
            sampled.sort()
        else:
            sampled = live_indices
        print(f"[M4] Scoring {len(sampled)} features (max_features={self.max_features})")

        # ── Step 3: For each sampled feature, get top-k image indices ────
        # Also compute min-max normalised activations for weights
        min_per_feature = np.min(image_codes, axis=0)  # [dict_size]
        range_per_feature = max_per_feature - min_per_feature
        range_per_feature[range_per_feature == 0] = 1.0  # avoid div by zero

        # Collect all unique image indices needed for embedding
        feature_topk: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        all_needed_indices: set[int] = set()

        for feat_idx in sampled:
            col = image_codes[:, feat_idx]
            # Top-k by activation value
            k = min(self.top_k_images, num_images)
            topk_idx = np.argpartition(col, -k)[-k:]
            topk_idx = topk_idx[np.argsort(col[topk_idx])[::-1]]  # sort descending

            # Normalised weights
            weights = (col[topk_idx] - min_per_feature[feat_idx]) / range_per_feature[feat_idx]
            feature_topk[feat_idx] = (topk_idx, weights.astype(np.float32))
            all_needed_indices.update(topk_idx.tolist())

        print(f"[M4] Need embeddings for {len(all_needed_indices)} unique images")

        # ── Step 4: Load cross-model backbone and embed needed images ────
        print(f"[M4] Loading cross-model backbone: {cross_model_name}")
        embeddings = self._embed_images(
            sorted(all_needed_indices), cross_model_name, device,
        )
        # embeddings: dict[int, np.ndarray]  (image_idx → [768] vector)

        # ── Step 5: Compute per-feature MS ───────────────────────────────
        print(f"[M4] Computing MS for {len(sampled)} features...")
        ms_scores = np.full(len(sampled), np.nan, dtype=np.float64)

        for i, feat_idx in enumerate(tqdm(sampled, desc="M4 scoring")):
            topk_idx, weights = feature_topk[feat_idx]
            k = len(topk_idx)

            # Gather embeddings for this feature's top-k images
            embeds = np.stack([embeddings[idx] for idx in topk_idx])  # [k, 768]
            embeds_t = torch.from_numpy(embeds).float()
            # Normalise for cosine similarity
            embeds_t = F.normalize(embeds_t, dim=1)

            weights_t = torch.from_numpy(weights).float()

            # Pairwise weighted cosine similarity (upper triangle)
            numer = 0.0
            denom = 0.0
            for a in range(k):
                for b in range(a + 1, k):
                    cos_ab = float((embeds_t[a] * embeds_t[b]).sum())
                    w_ab = float(weights_t[a] * weights_t[b])
                    numer += w_ab * cos_ab
                    denom += w_ab

            if denom > 0:
                ms_scores[i] = numer / denom

        valid_ms = ms_scores[~np.isnan(ms_scores)]
        print(f"[M4] Scored {len(valid_ms)} features successfully")

        return {
            "monosemanticity_score": float(np.mean(valid_ms)) if len(valid_ms) > 0 else None,
            "ms_std": float(np.std(valid_ms)) if len(valid_ms) > 0 else None,
            "ms_median": float(np.median(valid_ms)) if len(valid_ms) > 0 else None,
            "num_features_scored": int(len(valid_ms)),
            "num_dead_features": num_dead,
            "cross_model": cross_model_name,
        }

    # -- Internal helpers ------------------------------------------------------

    def _encode_shards(
        self,
        sae: torch.nn.Module,
        shard_paths: list[str],
        mean: torch.Tensor,
        std: float,
        device: str,
        dict_size: int,
    ) -> np.ndarray:
        """Encode shards → max-pooled image codes via memmap (same as M3)."""
        sae.eval()

        shard_sizes = []
        for sp in shard_paths:
            s = torch.load(sp, map_location="cpu", weights_only=True)
            shard_sizes.append(s.shape[0])
            del s
        total_images = sum(shard_sizes)

        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_path = tmp.name
        tmp.close()

        codes_mmap = np.memmap(
            tmp_path, dtype=np.float32, mode="w+",
            shape=(total_images, dict_size),
        )

        group_size = 16
        write_idx = 0

        with torch.no_grad():
            for shard_path in tqdm(shard_paths, desc="M4 encoding shards"):
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                N, P, D = shard.shape

                for g_start in range(0, N, group_size):
                    g_end = min(g_start + group_size, N)
                    group = shard[g_start:g_end]
                    G = group.shape[0]
                    tokens = group.reshape(G * P, D).float()
                    tokens = (tokens - mean) / std

                    code_chunks: list[torch.Tensor] = []
                    for start in range(0, tokens.shape[0], self.encode_batch_size):
                        batch = tokens[start : start + self.encode_batch_size].to(device)
                        _pre, codes, _xhat = sae(batch)
                        code_chunks.append(codes.cpu())
                        del batch, _pre, codes, _xhat

                    group_codes = torch.cat(code_chunks, dim=0)
                    group_codes = group_codes.reshape(G, P, -1)
                    pooled = group_codes.max(dim=1).values.numpy()

                    codes_mmap[write_idx : write_idx + G] = pooled
                    write_idx += G

                    del group, tokens, code_chunks, group_codes, pooled
                del shard

        if device == "cuda":
            torch.cuda.empty_cache()

        codes_mmap.flush()
        result = np.memmap(tmp_path, dtype=np.float32, mode="r",
                           shape=(total_images, dict_size))

        import atexit
        atexit.register(lambda p=tmp_path: os.unlink(p))
        return result

    def _embed_images(
        self,
        image_indices: list[int],
        cross_model_name: str,
        device: str,
    ) -> dict[int, np.ndarray]:
        """Load images from HuggingFace and embed via cross-model backbone.

        Returns dict mapping image index → L2-normalised embedding [768].
        """
        from src.backbones import load_backbone
        from src.evaluation.concept_detection.sparse_probing import load_imagenet_val_labels

        # Load HF val dataset (images only, via arrow cache)
        import src.utils.paths  # noqa: F401 — sets HF_HOME
        from datasets import load_dataset

        print(f"[M4] Loading ImageNet val images from HF cache...")
        ds = load_dataset("imagenet-1k", split="validation")

        # Load cross-model backbone
        adapter = load_backbone(cross_model_name, device=device)

        embeddings: dict[int, np.ndarray] = {}
        batch_size = 32  # images per forward pass through embedding backbone

        # Process in batches for efficiency
        for batch_start in tqdm(range(0, len(image_indices), batch_size),
                                desc=f"M4 embedding ({cross_model_name})"):
            batch_idx = image_indices[batch_start : batch_start + batch_size]
            pil_images = [ds[int(i)]["image"].convert("RGB") for i in batch_idx]

            with torch.no_grad():
                # [B, patches, 768] → mean-pool → [B, 768]
                patch_acts = adapter.extract_patch_activations(pil_images, layer=11)
                img_embeds = patch_acts.mean(dim=1)  # [B, 768]
                img_embeds = F.normalize(img_embeds, dim=1)  # L2 normalise
                img_embeds_np = img_embeds.cpu().numpy()

            for j, idx in enumerate(batch_idx):
                embeddings[idx] = img_embeds_np[j]

            del pil_images, patch_acts, img_embeds

        # Free the backbone
        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()

        return embeddings
