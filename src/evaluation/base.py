"""Base class for all VisaeBench evaluation metrics."""

from abc import ABC, abstractmethod

import torch


class MetricBase(ABC):
    """Abstract base for evaluation metrics (M1–M7).

    Each metric operates on a trained SAE and a set of activation shards.
    Implementations must be memory-safe: process shards one at a time and
    use running accumulators rather than collecting tensors.
    """

    @abstractmethod
    def evaluate(
        self,
        sae: torch.nn.Module,
        shard_paths: list[str],
        mean: torch.Tensor,
        std: float,
        device: str,
        **kwargs,
    ) -> dict:
        """Run the metric and return a results dict.

        Args:
            sae: Trained SAE (already on ``device``, set to eval mode by caller).
            shard_paths: Paths to shard_*.pt files.
            mean: Per-dimension mean from stats.json (shape ``[d_model]``).
            std: Scalar std from stats.json.
            device: torch device string.

        Returns:
            Dictionary of metric name → value.
        """
        ...
