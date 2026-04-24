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

import sys

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.evaluation.base import MetricBase

DEFAULT_K_VALUES = [32, 128, 512]

# When stderr is not a TTY (piped to a log file / captured by the dispatcher),
# tqdm emits a full new line on every update instead of rewriting in place.
# Throttling `mininterval` turns that from thousands of lines per extraction
# into one line every 30s. In a real terminal we keep tqdm's default fast
# refresh so the live bar still animates.
_TQDM_KW: dict = (
    {"mininterval": 30.0} if not sys.stderr.isatty() else {}
)

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
        backbone_batch_size: int = 64,
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

        # Steps 2+3: Extract backbone activations and SAE-encode them.
        #
        # For small OOD sets (EuroSAT, 10K × 64 patches × 768 = ~2 GB) the
        # two-phase path is fine. For iNaturalist (60K × 196 × 768 = ~35 GB)
        # the raw activation tensor does not fit in RAM on smaller nodes, so
        # we fuse extraction and encoding into one batched pass that only
        # holds the current batch of patch activations on device at a time.
        use_cache = dataset_name == "inaturalist"
        if use_cache:
            # Per-backbone patch-activation cache: first worker per backbone
            # builds it (~15 min at batch=64); all later SAEs for the same
            # backbone skip extraction entirely (~3 min to read + SAE-encode).
            print(f"[M5] Using iNat activation cache for {backbone_name}...")
            patch_acts_mmap = self._get_or_build_inat_cache(
                images=images,
                backbone_name=backbone_name,
                num_images=num_images,
                device=device,
            )
            del images  # cache owns its own image reads; we're done with the subset
            print(f"[M5] SAE-encoding from cache...")
            sae_codes, raw_pooled, sae_recon = self._encode_cached_activations(
                patch_acts=patch_acts_mmap,
                sae=sae,
                mean=mean,
                std=std,
                device=device,
                dict_size=dict_size,
            )
            del patch_acts_mmap  # release memmap handle before probing
        else:
            print(f"[M5] Extracting activations via {backbone_name}...")
            raw_activations = self._extract_ood_activations(
                images, backbone_name, device,
            )
            del images
            print(f"[M5] Encoding through SAE...")
            sae_codes, raw_pooled, sae_recon = self._encode_ood_activations(
                sae=sae,
                activations=raw_activations,
                mean=mean,
                std=std,
                device=device,
                dict_size=dict_size,
            )
            del raw_activations

        if device == "cuda":
            torch.cuda.empty_cache()

        # Step 4: Train probes.
        #
        # sklearn's lbfgs takes 20-40 min per probe at 10K iNat classes
        # (7.7M-parameter multinomial softmax, CPU-only). The torch-on-GPU
        # path solves the same optimisation problem in ~30-60 s per probe,
        # but via Adam + fixed-epoch minibatch SGD instead of deterministic
        # lbfgs-on-C-regularized-MLE. The two estimators converge to the
        # same solution in theory; in finite-iteration practice their
        # absolute accuracies differ by ~0.5-2 points. Cross-backbone
        # ranking and preservation ratios stay valid since all 60 SAEs use
        # one probe. EuroSAT is tiny (10 classes) and stays on sklearn so
        # its legacy result JSONs remain bitwise-reproducible.
        if dataset_name == "inaturalist":
            print(f"[M5] Training probes on GPU...")
            result = self._probe_gpu(
                sae_codes=sae_codes,
                raw_pooled=raw_pooled,
                sae_recon=sae_recon,
                labels=labels,
                device=device,
                seed=seed,
            )
        else:
            print(f"[M5] Training probes...")
            result = self._probe(
                sae_codes=sae_codes,
                raw_pooled=raw_pooled,
                sae_recon=sae_recon,
                labels=labels,
                seed=seed,
            )
        result["num_images"] = num_images
        result["num_classes"] = num_classes

        del sae_codes, raw_pooled, sae_recon
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
            **_TQDM_KW,
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Normalize OOD activations with ImageNet stats, encode through SAE,
        and max-pool codes / raw / reconstructions across patches.

        Args:
            activations: [num_images, patch_count, 768] on CPU.

        Returns:
            sae_codes:  np.ndarray [num_images, dict_size]
            raw_pooled: np.ndarray [num_images, 768]
            sae_recon:  np.ndarray [num_images, 768]  (max-pooled SAE reconstructions)
        """
        sae.eval()
        num_images, P, D = activations.shape

        sae_codes = np.zeros((num_images, dict_size), dtype=np.float32)
        raw_pooled = np.zeros((num_images, D), dtype=np.float32)
        sae_recon = np.zeros((num_images, D), dtype=np.float32)

        group_size = 16

        for g_start in tqdm(
            range(0, num_images, group_size),
            desc="M5 SAE encoding",
            **_TQDM_KW,
        ):
            g_end = min(g_start + group_size, num_images)
            group = activations[g_start:g_end]  # [G, P, D]
            G = group.shape[0]

            tokens = group.reshape(G * P, D).float()
            tokens_norm = (tokens - mean) / std

            # Raw baseline: max-pool normalized activations
            raw_reshaped = tokens_norm.reshape(G, P, D)
            raw_pooled[g_start:g_end] = raw_reshaped.max(dim=1).values.numpy()

            # SAE encoding — keep both codes AND reconstructions
            code_chunks: list[torch.Tensor] = []
            recon_chunks: list[torch.Tensor] = []
            for start in range(0, tokens_norm.shape[0], self.encode_batch_size):
                batch = tokens_norm[start : start + self.encode_batch_size].to(device)
                _pre, codes, xhat = sae(batch)
                code_chunks.append(codes.cpu())
                recon_chunks.append(xhat.cpu())
                del batch, _pre, codes, xhat

            group_codes = torch.cat(code_chunks, dim=0)
            group_codes = group_codes.reshape(G, P, -1)
            sae_codes[g_start:g_end] = group_codes.max(dim=1).values.numpy()

            group_recon = torch.cat(recon_chunks, dim=0).reshape(G, P, D)
            sae_recon[g_start:g_end] = group_recon.max(dim=1).values.numpy()

            del group, tokens, tokens_norm, raw_reshaped
            del code_chunks, recon_chunks, group_codes, group_recon

        if device == "cuda":
            torch.cuda.empty_cache()

        return sae_codes, raw_pooled, sae_recon

    # -- Per-backbone iNat patch-activation cache -----------------------------
    #
    # At M5 time the same 100K iNat images are forwarded through one of five
    # ViT backbones for each of 60 SAE checkpoints — meaning every backbone's
    # patch activations are recomputed 12× on identical inputs. This cache
    # writes each backbone's patch activations to disk once; subsequent SAEs
    # for the same backbone skip the backbone entirely and stream from the
    # memmap. Cache is versioned by (backbone, max_images, sampler seed) so
    # an incompatible subset can't silently be reused. A simple O_CREAT|O_EXCL
    # lock coordinates concurrent workers on the same backbone.

    def _inat_cache_dir(self, backbone_name: str, num_images: int) -> str:
        """Directory under DATA_ROOT that stores this backbone's iNat cache.

        Versioned by num_images + loader seed so a changed subsample never
        reuses a stale cache.
        """
        import os
        import src.utils.paths as paths

        # Loader default seed; mirrors load_inaturalist's default.
        sampler_seed = 42
        return os.path.join(
            paths.DATA_ROOT,
            "activations_inat_val",
            backbone_name,
            "layer_11",
            f"n{num_images}_seed{sampler_seed}",
        )

    def _get_or_build_inat_cache(
        self,
        images,
        backbone_name: str,
        num_images: int,
        device: str,
    ) -> np.memmap:
        """Return a read-only memmap of per-image patch activations.

        First caller for this (backbone, num_images, seed) triple builds the
        cache; concurrent callers block on an O_CREAT|O_EXCL lock file and
        poll until the `done.marker` appears, then mmap the completed cache.
        """
        import json
        import os
        import time

        cache_dir = self._inat_cache_dir(backbone_name, num_images)
        os.makedirs(cache_dir, exist_ok=True)

        data_path = os.path.join(cache_dir, "patch_acts.f16")
        shape_path = os.path.join(cache_dir, "shape.json")
        done_path = os.path.join(cache_dir, "done.marker")
        lock_path = os.path.join(cache_dir, "build.lock")

        def _load_ready() -> np.memmap:
            with open(shape_path) as f:
                shape = tuple(json.load(f)["shape"])
            return np.memmap(
                data_path, dtype=np.float16, mode="r", shape=shape,
            )

        # Fast path: cache already built
        if os.path.exists(done_path) and os.path.exists(shape_path):
            print(f"[M5 cache] HIT {backbone_name}: {data_path}")
            return _load_ready()

        # Try to take the build lock. O_EXCL is atomic-ish on NFS and
        # bulletproof on local/POSIX filesystems.
        while True:
            try:
                fd = os.open(
                    lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
                try:
                    os.write(fd, f"pid={os.getpid()} t={time.time():.0f}\n".encode())
                finally:
                    os.close(fd)
                break  # acquired
            except FileExistsError:
                # Another worker holds the lock. If it already finished, we
                # may see done.marker appear while we wait.
                if os.path.exists(done_path) and os.path.exists(shape_path):
                    print(f"[M5 cache] HIT (just finished) {backbone_name}")
                    return _load_ready()
                print(
                    f"[M5 cache] WAIT {backbone_name}: another worker is "
                    f"building {data_path}; polling every 30 s"
                )
                time.sleep(30)

        # We have the lock. Double-check in case another process finished
        # building between our last fast-path check and lock acquisition.
        try:
            if os.path.exists(done_path) and os.path.exists(shape_path):
                print(f"[M5 cache] HIT (race-lost) {backbone_name}")
                return _load_ready()

            print(f"[M5 cache] BUILD {backbone_name}: {data_path}")
            self._build_inat_cache(
                images=images,
                backbone_name=backbone_name,
                num_images=num_images,
                device=device,
                cache_dir=cache_dir,
                data_path=data_path,
                shape_path=shape_path,
                done_path=done_path,
            )
            return _load_ready()
        finally:
            try:
                os.unlink(lock_path)
            except FileNotFoundError:
                pass

    def _build_inat_cache(
        self,
        images,
        backbone_name: str,
        num_images: int,
        device: str,
        cache_dir: str,
        data_path: str,
        shape_path: str,
        done_path: str,
    ) -> None:
        """Run the backbone over all iNat images and write patch activations
        to a disk memmap. Called only by the worker that holds the lock.
        """
        import json
        import os

        from src.backbones import load_backbone

        adapter = load_backbone(backbone_name, device=device)

        # Probe shape on the first image
        first = adapter.extract_patch_activations(images[:1], layer=11)
        _, patch_count, d_model = first.shape
        del first

        shape = (num_images, patch_count, d_model)
        mmap = np.memmap(
            data_path, dtype=np.float16, mode="w+", shape=shape,
        )

        try:
            for batch_start in tqdm(
                range(0, num_images, self.backbone_batch_size),
                desc=f"iNat cache build ({backbone_name})",
                **_TQDM_KW,
            ):
                batch_end = min(
                    batch_start + self.backbone_batch_size, num_images,
                )
                batch_images = images[batch_start:batch_end]
                patch_acts = adapter.extract_patch_activations(
                    batch_images, layer=11,
                )
                # Cast on device (fp16) then one H2D→H2H copy per batch.
                mmap[batch_start:batch_end] = (
                    patch_acts.to(torch.float16).cpu().numpy()
                )
                del patch_acts

            mmap.flush()
        finally:
            # Release memmap handle whether or not we finished cleanly
            del mmap
            del adapter
            if device == "cuda":
                torch.cuda.empty_cache()

        # Shape metadata before the done marker — readers check both exist
        with open(shape_path, "w") as f:
            json.dump(
                {"shape": list(shape), "dtype": "float16"}, f,
            )
        # Create the done marker atomically via rename so partial writes
        # can't leave it visible before shape.json lands on disk.
        tmp_done = done_path + ".tmp"
        with open(tmp_done, "w") as f:
            f.write("ok\n")
        os.rename(tmp_done, done_path)
        size_gb = os.path.getsize(data_path) / (1024 ** 3)
        print(
            f"[M5 cache] DONE {backbone_name}: {data_path} "
            f"shape={shape} size={size_gb:.1f}GB"
        )

    # -- SAE encode from cached patch activations -----------------------------

    @torch.no_grad()
    def _encode_cached_activations(
        self,
        patch_acts: np.memmap,
        sae: torch.nn.Module,
        mean: torch.Tensor,
        std: float,
        device: str,
        dict_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stream from the cache memmap; normalize, SAE-encode, max-pool.

        Produces three outputs per image:
          - sae_codes:  max-pooled SAE codes         [N, dict_size] fp16
          - raw_pooled: max-pooled raw activations   [N, d_model]   fp16
          - sae_recon:  max-pooled SAE reconstructions [N, d_model] fp16

        Memory footprint: only one `backbone_batch_size` batch lives on GPU
        at a time. CPU-resident cost is num_images × (dict_size + 2·d_model)
        × 2 B (≈1 GB for 100K images and dict_size=6144).
        """
        sae.eval()
        num_images, patch_count, d_model = patch_acts.shape

        sae_codes = np.zeros((num_images, dict_size), dtype=np.float16)
        raw_pooled = np.zeros((num_images, d_model), dtype=np.float16)
        sae_recon = np.zeros((num_images, d_model), dtype=np.float16)

        mean_d = mean.to(device)

        for batch_start in tqdm(
            range(0, num_images, self.backbone_batch_size),
            desc="M5 SAE encode (from iNat cache)",
            **_TQDM_KW,
        ):
            batch_end = min(
                batch_start + self.backbone_batch_size, num_images,
            )
            # Cache is fp16 on disk; cast up to fp32 on device for numerics
            batch = torch.from_numpy(
                np.ascontiguousarray(
                    patch_acts[batch_start:batch_end], dtype=np.float32,
                )
            ).to(device)
            B, P, D = batch.shape

            normed = (batch - mean_d) / std
            raw_pooled[batch_start:batch_end] = (
                normed.max(dim=1).values.to(torch.float16).cpu().numpy()
            )

            tokens = normed.reshape(B * P, D)
            code_chunks: list[torch.Tensor] = []
            recon_chunks: list[torch.Tensor] = []
            for start in range(0, tokens.shape[0], self.encode_batch_size):
                chunk = tokens[start : start + self.encode_batch_size]
                _pre, codes, xhat = sae(chunk)
                code_chunks.append(codes)
                recon_chunks.append(xhat)
                del _pre
            all_codes = torch.cat(code_chunks, dim=0).reshape(B, P, -1)
            sae_codes[batch_start:batch_end] = (
                all_codes.max(dim=1).values.to(torch.float16).cpu().numpy()
            )
            all_recon = torch.cat(recon_chunks, dim=0).reshape(B, P, D)
            sae_recon[batch_start:batch_end] = (
                all_recon.max(dim=1).values.to(torch.float16).cpu().numpy()
            )
            del batch, normed, tokens, code_chunks, recon_chunks
            del all_codes, all_recon

        if device == "cuda":
            torch.cuda.empty_cache()

        return sae_codes, raw_pooled, sae_recon

    # -- Fused extract + encode (LEGACY — superseded by cache path above) ------

    @torch.no_grad()
    def _extract_and_encode_streamed(
        self,
        images,
        backbone_name: str,
        sae: torch.nn.Module,
        mean: torch.Tensor,
        std: float,
        device: str,
        dict_size: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Single-pass fused extract + SAE-encode + max-pool.

        Avoids ever materializing the full [num_images, patch_count, 768]
        activation tensor in RAM — only one backbone batch lives on device
        at a time. Outputs stored as float16 (halved RAM vs float32).

        Returns:
            sae_codes:  np.ndarray [num_images, dict_size] (float16)
            raw_pooled: np.ndarray [num_images, d_model]   (float16)
            sae_recon:  np.ndarray [num_images, d_model]   (float16)
        """
        from src.backbones import load_backbone

        sae.eval()
        num_images = len(images)
        d_model = mean.shape[0]

        sae_codes = np.zeros((num_images, dict_size), dtype=np.float16)
        raw_pooled = np.zeros((num_images, d_model), dtype=np.float16)
        sae_recon = np.zeros((num_images, d_model), dtype=np.float16)

        adapter = load_backbone(backbone_name, device=device)

        for batch_start in tqdm(
            range(0, num_images, self.backbone_batch_size),
            desc=f"M5 streamed ({backbone_name})",
            **_TQDM_KW,
        ):
            batch_end = min(batch_start + self.backbone_batch_size, num_images)
            batch_images = images[batch_start:batch_end]

            # [B, P, D] on device
            patch_acts = adapter.extract_patch_activations(batch_images, layer=11)
            B, P, D = patch_acts.shape

            # Normalize with ImageNet stats. `mean` / `std` come from CPU;
            # promote once to the activations' device for the subtract.
            mean_d = mean.to(patch_acts.device)
            normed = (patch_acts - mean_d) / std  # [B, P, D] on device

            # Raw max-pool (baseline features) — cast to fp16 on device first
            # so the host-side numpy allocation stays at the declared dtype.
            raw_pooled[batch_start:batch_end] = (
                normed.max(dim=1).values.to(torch.float16).cpu().numpy()
            )

            # SAE encode token-by-token then max-pool codes AND reconstructions.
            tokens = normed.reshape(B * P, D)
            code_chunks: list[torch.Tensor] = []
            recon_chunks: list[torch.Tensor] = []
            for start in range(0, tokens.shape[0], self.encode_batch_size):
                chunk = tokens[start : start + self.encode_batch_size]
                _pre, codes, xhat = sae(chunk)
                code_chunks.append(codes)
                recon_chunks.append(xhat)
                del _pre
            all_codes = torch.cat(code_chunks, dim=0).reshape(B, P, -1)
            sae_codes[batch_start:batch_end] = (
                all_codes.max(dim=1).values.to(torch.float16).cpu().numpy()
            )
            all_recon = torch.cat(recon_chunks, dim=0).reshape(B, P, D)
            sae_recon[batch_start:batch_end] = (
                all_recon.max(dim=1).values.to(torch.float16).cpu().numpy()
            )

            del patch_acts, normed, tokens, code_chunks, recon_chunks
            del all_codes, all_recon

        del adapter
        if device == "cuda":
            torch.cuda.empty_cache()

        return sae_codes, raw_pooled, sae_recon

    # -- Probing ---------------------------------------------------------------

    def _probe(
        self,
        sae_codes: np.ndarray,
        raw_pooled: np.ndarray,
        sae_recon: np.ndarray,
        labels: np.ndarray,
        seed: int,
    ) -> dict:
        """Train dense probes on raw + reconstructed activations and sparse
        probes on SAE codes.

        Returns dict with raw_accuracy, recon_accuracy, preservation_recon,
        and sae_k{k}_accuracy / preservation_k{k} for each k in self.k_values.
        """
        # Stratified train/test split — same split rows for raw, recon, codes.
        (
            X_sae_train, X_sae_test,
            X_raw_train, X_raw_test,
            X_recon_train, X_recon_test,
            y_train, y_test,
        ) = train_test_split(
            sae_codes, raw_pooled, sae_recon, labels,
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

        # Reconstruction baseline: dense probe on SAE x̂ (same hyperparameters
        # as raw, so preservation_recon is bounded in [0, 1]).
        print("[M5] Training reconstruction probe...")
        clf_recon = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            random_state=seed,
        )
        clf_recon.fit(X_recon_train, y_train)
        recon_acc = float(clf_recon.score(X_recon_test, y_test))
        results["recon_accuracy"] = recon_acc
        results["preservation_recon"] = (
            recon_acc / raw_acc if raw_acc > 0 else 0.0
        )
        print(f"[M5] Recon accuracy: {recon_acc:.4f} "
              f"(preservation_recon: {results['preservation_recon']:.3f})")
        del clf_recon

        # Feature ranking for SAE codes (f_classif, computed once)
        f_scores, _ = f_classif(X_sae_train, y_train)
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        ranked_indices = np.argsort(f_scores)[::-1]

        # Sparse probes at each k
        for k in tqdm(self.k_values, desc="M5 sparse probes", **_TQDM_KW):
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

    # -- GPU probing (torch.nn.Linear + Adam, for large-class OOD sets) --------

    def _probe_gpu(
        self,
        sae_codes: np.ndarray,
        raw_pooled: np.ndarray,
        sae_recon: np.ndarray,
        labels: np.ndarray,
        device: str,
        seed: int,
    ) -> dict:
        """GPU multinomial logistic regression — drop-in for _probe at 10K
        classes. Returns raw_accuracy, recon_accuracy, preservation_recon,
        plus sae_k{k}_accuracy / preservation_k{k} for each k in self.k_values.

        Train/test split, feature ranking, and the output schema match the
        sklearn path. Only the estimator itself differs: torch.nn.Linear
        trained with Adam (lr=1e-2, weight_decay=1e-4, 50 fixed epochs, batch
        2048) instead of sklearn lbfgs on a C=1.0-regularized MLE. All
        probes (raw, recon, and each sparse-k SAE probe) share these
        hyperparameters for fair comparability.

        Each train matrix is materialized on-device once per probe, before
        the epoch loop. The training loop only indexes into that resident
        tensor — no per-epoch host→device transfers.
        """
        import time

        # Split raw, recon, sae codes, and labels in lockstep — identical
        # stratification and seed => identical train/test rows across all
        # three probes. This is the property that makes preservation ratios
        # comparable across the dense and sparse probes.
        (
            X_sae_train, X_sae_test,
            X_raw_train, X_raw_test,
            X_recon_train, X_recon_test,
            y_train, y_test,
        ) = train_test_split(
            sae_codes, raw_pooled, sae_recon, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=seed,
        )
        num_classes = int(labels.max()) + 1
        print(f"[M5 gpu] split: {X_raw_train.shape[0]} train / "
              f"{X_raw_test.shape[0]} test, {num_classes} classes")

        results: dict = {}

        # ---- Raw baseline (dense probe on raw max-pool) ------------------
        t0 = time.time()
        raw_acc = self._fit_linear_gpu(
            X_tr_np=X_raw_train, X_te_np=X_raw_test,
            y_tr_np=y_train, y_te_np=y_test,
            num_classes=num_classes, device=device, seed=seed,
        )
        results["raw_accuracy"] = raw_acc
        print(f"[M5 gpu] Raw baseline accuracy: {raw_acc:.4f} "
              f"({time.time() - t0:.1f}s)")

        # ---- Reconstruction baseline (dense probe on SAE x̂) -------------
        # Mirrors M2's reconstruction-preservation semantics: 768-d features,
        # same probe hyperparameters as raw, so preservation_recon is
        # bounded in [0, 1] and is directly interpretable as "fraction of
        # raw probe accuracy retained after passing through the SAE".
        t0 = time.time()
        recon_acc = self._fit_linear_gpu(
            X_tr_np=X_recon_train, X_te_np=X_recon_test,
            y_tr_np=y_train, y_te_np=y_test,
            num_classes=num_classes, device=device, seed=seed,
        )
        results["recon_accuracy"] = recon_acc
        results["preservation_recon"] = (
            recon_acc / raw_acc if raw_acc > 0 else 0.0
        )
        print(f"[M5 gpu] Recon accuracy:        {recon_acc:.4f} "
              f"(preservation_recon: {results['preservation_recon']:.3f}, "
              f"{time.time() - t0:.1f}s)")

        # ---- Feature ranking for SAE codes -------------------------------
        # f_classif wants fp32+; SAE code arrays come from the streamed path
        # as fp16. Single cast here, released after ranking.
        X_sae_train_f32 = X_sae_train.astype(np.float32, copy=False)
        f_scores, _ = f_classif(X_sae_train_f32, y_train)
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        ranked_indices = np.argsort(f_scores)[::-1]
        del X_sae_train_f32

        # ---- Sparse probes ----------------------------------------------
        for k in self.k_values:
            top_k_idx = ranked_indices[:k]
            X_tr_k = X_sae_train[:, top_k_idx]
            X_te_k = X_sae_test[:, top_k_idx]
            t0 = time.time()
            sae_acc = self._fit_linear_gpu(
                X_tr_np=X_tr_k, X_te_np=X_te_k,
                y_tr_np=y_train, y_te_np=y_test,
                num_classes=num_classes, device=device, seed=seed,
            )
            results[f"sae_k{k}_accuracy"] = sae_acc
            results[f"preservation_k{k}"] = (
                sae_acc / raw_acc if raw_acc > 0 else 0.0
            )
            print(f"[M5 gpu] SAE k={k}: {sae_acc:.4f} "
                  f"(preservation: {results[f'preservation_k{k}']:.3f}, "
                  f"{time.time() - t0:.1f}s)")

        return results

    def _fit_linear_gpu(
        self,
        X_tr_np: np.ndarray,
        X_te_np: np.ndarray,
        y_tr_np: np.ndarray,
        y_te_np: np.ndarray,
        num_classes: int,
        device: str,
        seed: int,
        epochs: int = 50,
        batch_size: int = 2048,
        lr: float = 1e-2,
        weight_decay: float = 1e-4,
    ) -> float:
        """Fit one torch.nn.Linear probe on GPU; return top-1 test accuracy.

        The full train/test matrices are moved to `device` once up front
        (one cudaMemcpy per tensor). The inner loop is pure device-resident
        indexing + Adam steps — no host→device traffic per epoch, per the
        user's implementation note.

        Features are per-dim standardized (fit-on-train, apply-to-both) on
        the GPU before training. sklearn's LogisticRegression+lbfgs tolerates
        unscaled features via its regularization coupling; Adam with fixed
        lr does not — raw CLIP max-pool features were stuck at ~chance until
        this step was added. Confirmed by scripts/debug_inat_m5.py.
        """
        from torch import nn

        torch.manual_seed(seed)
        if device == "cuda":
            torch.cuda.manual_seed_all(seed)

        # One-shot H2D transfer. fp16 → fp32 cast happens during .astype,
        # which numpy does lazily-ish; the tensor landing on-device is fp32.
        X_tr = torch.from_numpy(
            np.ascontiguousarray(X_tr_np, dtype=np.float32)
        ).to(device)
        X_te = torch.from_numpy(
            np.ascontiguousarray(X_te_np, dtype=np.float32)
        ).to(device)
        y_tr = torch.from_numpy(y_tr_np.astype(np.int64, copy=False)).to(device)
        y_te = torch.from_numpy(y_te_np.astype(np.int64, copy=False)).to(device)

        # Per-dim StandardScaler fit on train, applied to train+test.
        # eps guards constant (dead) dims from a divide-by-zero.
        mu = X_tr.mean(dim=0, keepdim=True)
        sd = X_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd
        del mu, sd

        n, d = X_tr.shape
        clf = nn.Linear(d, num_classes).to(device)
        opt = torch.optim.Adam(
            clf.parameters(), lr=lr, weight_decay=weight_decay,
        )
        loss_fn = nn.CrossEntropyLoss()

        clf.train()
        for _ in range(epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, batch_size):
                idx = perm[start : start + batch_size]
                logits = clf(X_tr[idx])
                loss = loss_fn(logits, y_tr[idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        clf.eval()
        correct = 0
        with torch.no_grad():
            eval_batch = 4096
            for start in range(0, X_te.shape[0], eval_batch):
                logits = clf(X_te[start : start + eval_batch])
                preds = logits.argmax(dim=1)
                correct += int((preds == y_te[start : start + eval_batch]).sum())
        acc = correct / X_te.shape[0]

        del clf, opt, X_tr, X_te, y_tr, y_te
        if device == "cuda":
            torch.cuda.empty_cache()
        return float(acc)
