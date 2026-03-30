import glob
import math
import os
from typing import Dict, List

import torch


def save_shard(activations: torch.Tensor, output_dir: str, shard_idx: int) -> str:
    """Save activation tensor as shard_XXX.pt inside output_dir.

    Args:
        activations: Tensor of shape [N, num_patches, d_model].
        output_dir:  Directory to write the shard into.
        shard_idx:   Zero-based shard index used for the filename.

    Returns:
        Absolute path of the written file.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"shard_{shard_idx:03d}.pt")
    torch.save(activations.float(), path)
    return path


def load_shard(output_dir: str, shard_idx: int) -> torch.Tensor:
    """Load a single shard by index.

    Args:
        output_dir: Directory containing shard files.
        shard_idx:  Zero-based shard index.

    Returns:
        Tensor loaded from shard_XXX.pt.
    """
    path = os.path.join(output_dir, f"shard_{shard_idx:03d}.pt")
    return torch.load(path, map_location="cpu")


def load_all_shards(output_dir: str) -> torch.Tensor:
    """Load and concatenate all shards in lexicographic order.

    Args:
        output_dir: Directory containing shard_XXX.pt files.

    Returns:
        Concatenated tensor along dim 0.
    """
    paths = sorted(glob.glob(os.path.join(output_dir, "shard_*.pt")))
    if not paths:
        raise FileNotFoundError(f"No shard files found in {output_dir}")
    shards = [torch.load(p, map_location="cpu") for p in paths]
    return torch.cat(shards, dim=0)


class WelfordAccumulator:
    """Online mean and variance via Welford's algorithm.

    Each call to update() accepts a batch of vectors [N, D] and updates
    running statistics treating every row as an independent sample.
    """

    def __init__(self) -> None:
        self._count: int = 0
        self._mean: torch.Tensor | None = None
        self._M2: torch.Tensor | None = None

    def update(self, batch: torch.Tensor) -> None:
        """Update running stats with a batch of shape [N, D].

        Args:
            batch: Float tensor of shape [N, D].
        """
        batch = batch.float()
        n = batch.shape[0]
        if self._mean is None:
            d = batch.shape[1]
            self._mean = torch.zeros(d, dtype=torch.float64)
            self._M2 = torch.zeros(d, dtype=torch.float64)

        batch_f64 = batch.double()
        for i in range(n):
            self._count += 1
            delta = batch_f64[i] - self._mean
            self._mean += delta / self._count
            delta2 = batch_f64[i] - self._mean
            self._M2 += delta * delta2

    @property
    def mean(self) -> torch.Tensor:
        """Per-dimension mean as a [D] float32 tensor."""
        if self._mean is None:
            raise RuntimeError("No data has been accumulated yet.")
        return self._mean.float()

    @property
    def std(self) -> float:
        """Scalar std: sqrt of the average per-dimension variance."""
        if self._M2 is None or self._count < 2:
            raise RuntimeError("Need at least 2 samples to compute std.")
        variance_per_dim = self._M2 / (self._count - 1)
        return float(variance_per_dim.mean().sqrt().item())

    def to_dict(self) -> Dict:
        """Return stats as a plain dict suitable for merging into stats.json."""
        return {
            "mean": self.mean.tolist(),
            "std": self.std,
        }

