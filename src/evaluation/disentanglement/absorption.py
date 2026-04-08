"""M7: Feature Absorption Rate.

Adapted from SAEBench (Karvonen et al., ICML 2025) for the vision domain.

Feature absorption occurs when a general SAE feature absorbs a more specific
concept, preventing it from being represented by its own dedicated feature.
We test this using ImageNet's WordNet class hierarchy: for sibling classes
that share a parent (e.g., different dog breeds), we check whether a single
SAE feature can discriminate each specific class, or whether discrimination
requires multiple features (indicating the primary feature absorbed the
subclass information).

Algorithm:
  1. Build sibling groups from WordNet hierarchy.
  2. For each class within a group, train binary probes (class-vs-siblings)
     using k=1 and k=4 most discriminative SAE features.
  3. If F1(k=4) - F1(k=1) > threshold, absorption is detected.
"""

import os
import tempfile

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from tqdm import tqdm

from src.evaluation.base import MetricBase
from src.evaluation.disentanglement.imagenet_hierarchy import build_sibling_groups


class FeatureAbsorption(MetricBase):
    """M7: Feature absorption rate.

    Results dict keys:
        absorption_rate       -- fraction of (group, class) tests showing absorption
        mean_f1_k1            -- average F1 with k_primary features
        mean_f1_k4            -- average F1 with k_expanded features
        mean_f1_improvement   -- average (f1_k4 - f1_k1)
        num_groups            -- number of sibling groups tested
        num_tests             -- total (group, class) pairs tested
        num_absorbed          -- how many showed absorption
    """

    def __init__(
        self,
        min_group_size: int = 3,
        absorption_threshold: float = 0.1,
        k_primary: int = 1,
        k_expanded: int = 4,
        encode_batch_size: int = 512,
    ):
        self.min_group_size = min_group_size
        self.absorption_threshold = absorption_threshold
        self.k_primary = k_primary
        self.k_expanded = k_expanded
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
        """Compute M7 feature absorption rate.

        Required kwargs:
            dict_size (int): SAE dictionary size.
            labels (np.ndarray): Class labels aligned with val shards.
        Optional kwargs:
            precomputed_codes (np.ndarray): [num_images, dict_size] image codes
                to skip re-encoding.
            seed (int): Random seed (default 42).
        """
        dict_size: int = kwargs["dict_size"]
        labels: np.ndarray = np.asarray(kwargs["labels"], dtype=np.int64)
        seed: int = kwargs.get("seed", 42)

        # ── Step 1: Get image codes ──────────────────────────────────────
        precomputed = kwargs.get("precomputed_codes", None)
        if precomputed is not None:
            image_codes = np.asarray(precomputed)
            print(f"[M7] Using precomputed codes: {image_codes.shape}")
        else:
            print(f"[M7] Encoding val images through SAE...")
            image_codes = self._encode_shards(sae, shard_paths, mean, std, device, dict_size)
            print(f"[M7] Encoded {image_codes.shape[0]} images")

        num_images = image_codes.shape[0]
        if num_images > len(labels):
            raise ValueError(f"More images ({num_images}) than labels ({len(labels)})")
        if num_images < len(labels):
            labels = labels[:num_images]

        # ── Step 2: Build sibling groups ─────────────────────────────────
        groups = build_sibling_groups(min_group_size=self.min_group_size)
        print(f"[M7] {len(groups)} sibling groups")

        # ── Step 3: Binary probes per (group, class) pair ────────────────
        f1_k1_list = []
        f1_k4_list = []
        absorbed_count = 0

        for group_name, class_indices in tqdm(groups.items(), desc="M7 absorption"):
            # Collect images belonging to any class in this group
            group_mask = np.isin(labels, class_indices)
            if group_mask.sum() < 10:
                continue  # skip tiny groups

            group_codes = image_codes[group_mask]  # [G, dict_size]
            group_labels = labels[group_mask]       # [G]

            for target_class in class_indices:
                # Binary labels: target_class vs rest-of-group
                binary_labels = (group_labels == target_class).astype(np.int64)
                n_pos = binary_labels.sum()
                n_neg = len(binary_labels) - n_pos

                # Need at least 3 positive and 3 negative samples
                if n_pos < 3 or n_neg < 3:
                    continue

                # Feature ranking for this binary task
                f_scores, _ = f_classif(group_codes, binary_labels)
                f_scores = np.nan_to_num(f_scores, nan=-np.inf)
                ranked = np.argsort(f_scores)[::-1]

                # k=1 probe
                f1_1 = self._train_probe(
                    group_codes, binary_labels, ranked, self.k_primary, seed,
                )
                # k=4 probe
                f1_4 = self._train_probe(
                    group_codes, binary_labels, ranked, self.k_expanded, seed,
                )

                f1_k1_list.append(f1_1)
                f1_k4_list.append(f1_4)

                if f1_4 - f1_1 > self.absorption_threshold:
                    absorbed_count += 1

        f1_k1_arr = np.array(f1_k1_list)
        f1_k4_arr = np.array(f1_k4_list)
        num_tests = len(f1_k1_list)

        return {
            "absorption_rate": absorbed_count / num_tests if num_tests > 0 else 0.0,
            "mean_f1_k1": float(f1_k1_arr.mean()) if num_tests > 0 else 0.0,
            "mean_f1_k4": float(f1_k4_arr.mean()) if num_tests > 0 else 0.0,
            "mean_f1_improvement": float((f1_k4_arr - f1_k1_arr).mean()) if num_tests > 0 else 0.0,
            "num_groups": len(groups),
            "num_tests": num_tests,
            "num_absorbed": absorbed_count,
        }

    # -- Internal helpers ------------------------------------------------------

    @staticmethod
    def _train_probe(
        X: np.ndarray,
        y: np.ndarray,
        ranked_features: np.ndarray,
        k: int,
        seed: int,
    ) -> float:
        """Train a logistic regression on the top-k features, return F1."""
        top_k_idx = ranked_features[:k]
        X_k = X[:, top_k_idx]

        clf = LogisticRegression(solver="lbfgs", max_iter=500, C=1.0,
                                 random_state=seed)
        clf.fit(X_k, y)
        preds = clf.predict(X_k)
        return float(f1_score(y, preds, average="binary", zero_division=0.0))

    def _encode_shards(
        self,
        sae: torch.nn.Module,
        shard_paths: list[str],
        mean: torch.Tensor,
        std: float,
        device: str,
        dict_size: int,
    ) -> np.ndarray:
        """Encode shards → max-pooled image codes via memmap (same as M3/M4)."""
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
            for shard_path in tqdm(shard_paths, desc="M7 encoding shards"):
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
