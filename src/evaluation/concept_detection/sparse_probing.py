"""M3: Sparse Probing.

For each sparsity level k in {1, 2, 4, 8, 16, 32}, train a linear probe using
only the top-k most informative SAE features to predict ImageNet class labels.
Higher accuracy at low k indicates more concept-aligned, interpretable features.

Algorithm:
  1. Encode val activations through the SAE (shard-by-shard, no full load).
  2. Max-pool codes across patches to get one vector per image.
  3. Select top-k features by F-statistic (f_classif) w.r.t. class labels.
  4. Train logistic regression on the k-sparse features (80/20 stratified split).
  5. Report top-1 accuracy per k and area under the k-accuracy curve.
"""

import glob
import json
import os

import numpy as np
import torch
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.evaluation.base import MetricBase

DEFAULT_K_VALUES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


class SparseProbing(MetricBase):
    """M3: Sparse probing metric.

    Results dict keys:
        k_{k}_accuracy  -- top-1 accuracy using only the k most informative features
        auc             -- area under the k-accuracy curve, normalized to [0, 1]
        num_images      -- number of images evaluated
        num_features    -- SAE dictionary size
    """

    def __init__(
        self,
        k_values: list[int] | None = None,
        encode_batch_size: int = 512,
        test_size: float = 0.2,
    ):
        self.k_values = k_values or DEFAULT_K_VALUES
        self.encode_batch_size = encode_batch_size
        self.test_size = test_size

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
        """Run M3 sparse probing.

        Required kwargs:
            dict_size (int): SAE dictionary size.
            labels (list[int] | np.ndarray): Class labels aligned with shards.
        Optional kwargs:
            seed (int): Random seed for train/test split (default 42).
        """
        dict_size: int = kwargs["dict_size"]
        labels = kwargs["labels"]
        seed: int = kwargs.get("seed", 42)

        if isinstance(labels, list):
            labels = np.array(labels, dtype=np.int64)

        # Step 1: encode all images → [num_images, dict_size] via max-pool
        image_codes = self._encode_shards(sae, shard_paths, mean, std, device, dict_size)
        num_images, num_features = image_codes.shape

        if num_images > len(labels):
            raise ValueError(
                f"More images ({num_images}) than labels ({len(labels)}). "
                f"Cannot align shards to class labels."
            )
        if num_images < len(labels):
            print(f"[sparse_probing] Val shards contain {num_images} of {len(labels)} "
                  f"images — using first {num_images} labels.")
            labels = labels[:num_images]

        # Step 2: feature ranking (F-statistic, computed once)
        f_scores, _ = f_classif(image_codes, labels)
        # Replace NaN scores (constant features) with -inf so they rank last
        f_scores = np.nan_to_num(f_scores, nan=-np.inf)
        ranked_indices = np.argsort(f_scores)[::-1]  # descending

        # Step 3: train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            image_codes, labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=seed,
        )

        # Step 4: k-sparse probes
        # Use lbfgs solver — much faster than saga for small feature counts,
        # and converges reliably for k <= 32 features × 1000 classes.
        results: dict = {}
        for k in tqdm(self.k_values, desc="M3 sparse probes"):
            top_k_idx = ranked_indices[:k]
            X_tr_k = X_train[:, top_k_idx]
            X_te_k = X_test[:, top_k_idx]

            clf = LogisticRegression(
                solver="lbfgs",
                max_iter=500,
                C=1.0,
                random_state=seed,
            )
            clf.fit(X_tr_k, y_train)
            acc = float(clf.score(X_te_k, y_test))
            results[f"k_{k}_accuracy"] = acc

        # Step 5: AUC (trapezoidal, normalized by max possible area)
        ks = sorted(self.k_values)
        accs = [results[f"k_{k}_accuracy"] for k in ks]
        auc = float(np.trapezoid(accs, x=ks) / (ks[-1] - ks[0])) if len(ks) > 1 else accs[0]

        results["auc"] = auc
        results["num_images"] = num_images
        results["num_features"] = num_features
        return results

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
        """Encode activation shards through SAE and max-pool across patches.

        Processes one shard at a time.  Within each shard, batches of patches
        are sent through the SAE encoder; codes are moved back to CPU
        immediately.

        Returns:
            np.ndarray of shape [num_images, dict_size], float32.
        """
        sae.eval()

        # Count total images across shards to pre-allocate a memmap file,
        # avoiding accumulation of a multi-GB list in RAM.
        shard_sizes = []
        for sp in shard_paths:
            s = torch.load(sp, map_location="cpu", weights_only=True)
            shard_sizes.append(s.shape[0])
            del s
        total_images = sum(shard_sizes)

        import tempfile
        tmp = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_path = tmp.name
        tmp.close()

        codes_mmap = np.memmap(
            tmp_path, dtype=np.float32, mode="w+",
            shape=(total_images, dict_size),
        )

        # Process images in small groups to limit peak RAM.
        group_size = 16
        write_idx = 0

        with torch.no_grad():
            for shard_path in tqdm(shard_paths, desc="M3 encoding shards"):
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                N, P, D = shard.shape

                for g_start in range(0, N, group_size):
                    g_end = min(g_start + group_size, N)
                    group = shard[g_start:g_end]  # [G, P, D]
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
                    pooled = group_codes.max(dim=1).values.numpy()  # [G, dict_size]

                    codes_mmap[write_idx : write_idx + G] = pooled
                    write_idx += G

                    del group, tokens, code_chunks, group_codes, pooled

                del shard

        if device == "cuda":
            torch.cuda.empty_cache()

        # Flush and re-open as read-only memmap so it stays on disk
        codes_mmap.flush()
        result = np.memmap(tmp_path, dtype=np.float32, mode="r",
                           shape=(total_images, dict_size))

        # Clean up temp file when the array is garbage collected
        import atexit
        atexit.register(lambda: os.unlink(tmp_path))

        return result


def load_imagenet_val_labels() -> np.ndarray:
    """Load ImageNet-1K validation labels from cached arrow files.

    Reads directly from the HuggingFace datasets cache on NAS, avoiding
    the slow ``load_dataset`` call that rebuilds all splits.

    Returns:
        np.ndarray of int64 labels, shape [50000].
    """
    import pyarrow as pa
    from src.utils.paths import HF_CACHE_DIR

    # Find the arrow files in the HF datasets cache
    cache_root = os.path.join(HF_CACHE_DIR, "datasets", "imagenet-1k")
    val_pattern = os.path.join(cache_root, "**", "imagenet-1k-validation-*.arrow")
    val_files = sorted(glob.glob(val_pattern, recursive=True))

    if not val_files:
        raise FileNotFoundError(
            f"No cached imagenet-1k validation arrow files found under {cache_root}. "
            f"Run `load_dataset('imagenet-1k', split='validation')` once to populate the cache."
        )

    labels = []
    for f in val_files:
        reader = pa.ipc.open_stream(f)
        table = reader.read_all()
        labels.extend(table.column("label").to_pylist())

    return np.array(labels, dtype=np.int64)


def load_stats(activation_dir: str) -> tuple[torch.Tensor, float]:
    """Load mean/std from stats.json in an activation directory."""
    stats_path = os.path.join(activation_dir, "stats.json")
    with open(stats_path) as f:
        stats = json.load(f)
    mean = torch.tensor(stats["mean"], dtype=torch.float32)
    std = float(stats["std"])
    return mean, std


def discover_shards(activation_dir: str) -> list[str]:
    """Return sorted shard_*.pt paths."""
    paths = sorted(glob.glob(os.path.join(activation_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard files found in {activation_dir}")
    return paths
