"""M5: Cross-Domain Generalization.

Tests whether SAE features learned on ImageNet transfer to out-of-distribution
(OOD) datasets.  Good SAE features should capture general visual concepts that
generalize beyond the training domain.

Algorithm:
  1. Load OOD dataset images and pass through the same backbone -> activations.
  2. Normalize with ImageNet stats (the SAE was trained on ImageNet-normalized data).
  3. Encode through SAE -> codes; max-pool across patches -> per-image vectors.
  4. Also max-pool raw activations across patches (baseline).
  5. Train sparse logistic-regression probes (f_classif feature selection at each k)
     on SAE codes, and a dense probe on raw activations.
  6. Report accuracies and preservation ratio (SAE / raw).
"""

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.evaluation.base import MetricBase

DEFAULT_K_VALUES = [32, 128, 512]

# Map dataset name -> loader function name in ood_datasets module
_DATASET_LOADERS = {
    "eurosat": "load_eurosat",
    "inaturalist": "load_inaturalist",
}


class CrossDomainGeneralization(MetricBase):
    """M5: Cross-domain generalization metric.

    Evaluates whether SAE features preserve enough information for
    classification on out-of-distribution datasets, compared to raw
    backbone activations.

    Results dict keys (per dataset):
        raw_accuracy           -- baseline probe accuracy on raw activations
        sae_k{k}_accuracy      -- SAE probe accuracy at sparsity level k
        preservation_k{k}      -- ratio SAE_accuracy / raw_accuracy
        num_images             -- number of images evaluated
        num_classes            -- number of classes in the OOD dataset
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        test_size: float = 0.2,
        max_images: int = 10000,
        encode_batch_size: int = 512,
        backbone_batch_size: int = 32,
    ):
        """
        Args:
            k_values: Sparsity levels for SAE probes.
            test_size: Fraction for test split.
            max_images: Max images to use from each OOD dataset.
            encode_batch_size: Batch size for SAE encoding.
            backbone_batch_size: Batch size for backbone feature extraction.
        """
        self.k_values = k_values or DEFAULT_K_VALUES
        self.test_size = test_size
        self.max_images = max_images
        self.encode_batch_size = encode_batch_size
        self.backbone_batch_size = backbone_batch_size

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
        """Run M5 cross-domain generalization.

        Note: shard_paths is unused here (OOD activations are extracted live),
        but kept for MetricBase interface compatibility.

        Required kwargs:
            dict_size (int): SAE dictionary size.
            backbone_name (str): Which backbone to load for OOD feature extraction.
            datasets (list[str]): Which OOD datasets to evaluate,
                e.g. ["eurosat"] or ["eurosat", "inaturalist"].
        Optional kwargs:
            seed (int): Random seed (default 42).
        """
        dict_size: int = kwargs["dict_size"]
        backbone_name: str = kwargs["backbone_name"]
        dataset_names: list[str] = kwargs["datasets"]
        seed: int = kwargs.get("seed", 42)

        results = {}
        for ds_name in dataset_names:
            print(f"\n[M5] Evaluating cross-domain on: {ds_name}")
            results[ds_name] = self._evaluate_dataset(
                sae=sae,
                mean=mean,
                std=std,
                device=device,
                dict_size=dict_size,
                backbone_name=backbone_name,
                dataset_name=ds_name,
                seed=seed,
            )
        return results

    # -- Per-dataset evaluation ------------------------------------------------

    def _evaluate_dataset(
        self,
        sae: torch.nn.Module,
        mean: torch.Tensor,
        std: float,
        device: str,
        dict_size: int,
        backbone_name: str,
        dataset_name: str,
        seed: int,
    ) -> dict:
        """Full cross-domain pipeline for one OOD dataset."""
        from src.evaluation.concept_detection.ood_datasets import (
            load_eurosat,
            load_inaturalist,
        )

        loaders = {
            "eurosat": load_eurosat,
            "inaturalist": load_inaturalist,
        }
        if dataset_name not in loaders:
            raise ValueError(
                f"Unknown OOD dataset '{dataset_name}'. "
                f"Valid options: {list(loaders.keys())}"
            )

        # Step 1: Load OOD images and labels
        print(f"[M5] Loading {dataset_name} (max {self.max_images} images)...")
        images, labels = loaders[dataset_name](max_images=self.max_images)
        num_images = len(images)
        num_classes = len(np.unique(labels))
        print(f"[M5] Loaded {num_images} images, {num_classes} classes")

        # Step 2: Extract activations through backbone
        print(f"[M5] Extracting activations via {backbone_name}...")
        raw_activations = self._extract_ood_activations(
            images, backbone_name, device,
        )
        # raw_activations: [num_images, patch_count, 768]

        # Free images from memory now that we have activations
        del images

        # Step 3: Normalize with ImageNet stats and encode through SAE
        print(f"[M5] Encoding through SAE...")
        sae_codes, raw_pooled = self._encode_ood_activations(
            sae=sae,
            activations=raw_activations,
            mean=mean,
            std=std,
            device=device,
            dict_size=dict_size,
        )
        # sae_codes: [num_images, dict_size]
        # raw_pooled: [num_images, 768]

        del raw_activations
        if device == "cuda":
            torch.cuda.empty_cache()

        # Step 4: Train probes
        print(f"[M5] Training probes...")
        result = self._probe(
            sae_codes=sae_codes,
            raw_pooled=raw_pooled,
            labels=labels,
            seed=seed,
        )
        result["num_images"] = num_images
        result["num_classes"] = num_classes

        del sae_codes, raw_pooled
        return result

    # -- Feature extraction ----------------------------------------------------

    @torch.no_grad()
    def _extract_ood_activations(
        self,
        images: list,
        backbone_name: str,
        device: str,
    ) -> torch.Tensor:
        """Extract patch activations from OOD images via backbone.

        Loads the backbone on-demand, processes images in batches,
        then frees the backbone to reclaim GPU memory.

        Returns:
            Tensor of shape [num_images, patch_count, 768] on CPU.
        """
        from src.backbones import load_backbone

        adapter = load_backbone(backbone_name, device=device)

        all_activations = []

        for batch_start in tqdm(
            range(0, len(images), self.backbone_batch_size),
            desc=f"M5 backbone ({backbone_name})",
        ):
            batch_images = images[batch_start : batch_start + self.backbone_batch_size]
            # extract_patch_activations handles PIL -> tensor preprocessing
            patch_acts = adapter.extract_patch_activations(batch_images, layer=11)
            # patch_acts: [B, patch_count, 768] on device
            all_activations.append(patch_acts.cpu())
            del patch_acts

        # Free backbone
        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()

        return torch.cat(all_activations, dim=0)  # [N, P, 768]

    # -- SAE encoding ----------------------------------------------------------

    @torch.no_grad()
    def _encode_ood_activations(
        self,
        sae: torch.nn.Module,
        activations: torch.Tensor,
        mean: torch.Tensor,
        std: float,
        device: str,
        dict_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Normalize OOD activations with ImageNet stats, encode through SAE,
        and max-pool.

        Also produces max-pooled raw activations for the baseline probe.

        Args:
            activations: [num_images, patch_count, 768] on CPU.

        Returns:
            sae_codes: np.ndarray [num_images, dict_size]
            raw_pooled: np.ndarray [num_images, 768]
        """
        sae.eval()
        num_images, P, D = activations.shape

        sae_codes = np.zeros((num_images, dict_size), dtype=np.float32)
        raw_pooled = np.zeros((num_images, D), dtype=np.float32)

        group_size = 16

        for g_start in tqdm(range(0, num_images, group_size), desc="M5 SAE encoding"):
            g_end = min(g_start + group_size, num_images)
            group = activations[g_start:g_end]  # [G, P, D]
            G = group.shape[0]

            tokens = group.reshape(G * P, D).float()
            tokens_norm = (tokens - mean) / std

            # Raw baseline: max-pool normalized activations
            raw_reshaped = tokens_norm.reshape(G, P, D)
            raw_pooled[g_start:g_end] = raw_reshaped.max(dim=1).values.numpy()

            # SAE encoding
            code_chunks: list[torch.Tensor] = []
            for start in range(0, tokens_norm.shape[0], self.encode_batch_size):
                batch = tokens_norm[start : start + self.encode_batch_size].to(device)
                _pre, codes, _xhat = sae(batch)
                code_chunks.append(codes.cpu())
                del batch, _pre, codes, _xhat

            group_codes = torch.cat(code_chunks, dim=0)
            group_codes = group_codes.reshape(G, P, -1)
            sae_codes[g_start:g_end] = group_codes.max(dim=1).values.numpy()

            del group, tokens, tokens_norm, raw_reshaped, code_chunks, group_codes

        if device == "cuda":
            torch.cuda.empty_cache()

        return sae_codes, raw_pooled

    # -- Probing ---------------------------------------------------------------

    def _probe(
        self,
        sae_codes: np.ndarray,
        raw_pooled: np.ndarray,
        labels: np.ndarray,
        seed: int,
    ) -> dict:
        """Train sparse probes on SAE codes and a dense probe on raw activations.

        Returns dict with raw_accuracy, sae_k{k}_accuracy, preservation_k{k}.
        """
        # Stratified train/test split (same split for all probes)
        (
            X_sae_train, X_sae_test,
            X_raw_train, X_raw_test,
            y_train, y_test,
        ) = train_test_split(
            sae_codes, raw_pooled, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=seed,
        )

        results: dict = {}

        # Baseline: dense probe on raw activations
        print("[M5] Training raw baseline probe...")
        clf_raw = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            random_state=seed,
        )
        clf_raw.fit(X_raw_train, y_train)
        raw_acc = float(clf_raw.score(X_raw_test, y_test))
        results["raw_accuracy"] = raw_acc
        print(f"[M5] Raw baseline accuracy: {raw_acc:.4f}")
        del clf_raw

        # Feature ranking for SAE codes (f_classif, computed once)
        f_scores, _ = f_classif(X_sae_train, y_train)
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        ranked_indices = np.argsort(f_scores)[::-1]

        # Sparse probes at each k
        for k in tqdm(self.k_values, desc="M5 sparse probes"):
            top_k_idx = ranked_indices[:k]
            X_tr_k = X_sae_train[:, top_k_idx]
            X_te_k = X_sae_test[:, top_k_idx]

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                C=1.0,
                random_state=seed,
            )
            clf.fit(X_tr_k, y_train)
            sae_acc = float(clf.score(X_te_k, y_test))
            results[f"sae_k{k}_accuracy"] = sae_acc

            # Preservation ratio
            if raw_acc > 0:
                results[f"preservation_k{k}"] = sae_acc / raw_acc
            else:
                results[f"preservation_k{k}"] = 0.0

            print(f"[M5] SAE k={k}: {sae_acc:.4f} "
                  f"(preservation: {results[f'preservation_k{k}']:.3f})")
            del clf

        return results
