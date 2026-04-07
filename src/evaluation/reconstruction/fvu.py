"""M1: Fraction of Variance Unexplained (FVU).

Measures reconstruction quality of a sparse autoencoder by computing
the ratio of residual variance to input variance over held-out activations.
Also reports average L0 sparsity and dead-feature counts.

All statistics are computed incrementally (running sums) to stay within
the ~8 GB RAM budget — no tensor accumulation across shards.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.evaluation.base import MetricBase


class FVUMetric(MetricBase):
    """Fraction of Variance Unexplained (+ L0, dead features).

    Results dict keys:
        fvu         – Var(residual) / Var(input); lower is better, target < 0.10
        l0          – average number of active (non-zero) features per token
        dead_features – number of dictionary entries never activated
        dead_pct    – dead_features as a percentage of dict_size
    """

    def __init__(self, batch_size: int = 512):
        self.batch_size = batch_size

    def evaluate(
        self,
        sae: torch.nn.Module,
        shard_paths: list[str],
        mean: torch.Tensor,
        std: float,
        device: str,
        **kwargs,
    ) -> dict:
        """Compute FVU, L0, and dead-feature statistics.

        Args:
            sae: Trained SAE in eval mode on ``device``.
            shard_paths: Held-out shard files (``shard_*.pt``).
            mean: Per-dimension mean tensor, shape ``[d_model]``.
            std: Scalar standard deviation.
            device: Torch device string.
            **kwargs: Must include ``dict_size`` (int) — total dictionary size.

        Returns:
            Dict with keys ``fvu``, ``l0``, ``dead_features``, ``dead_pct``.
        """
        dict_size: int = kwargs["dict_size"]

        sae.eval()

        n_tok = 0
        sum_x2 = 0.0
        sum_x = 0.0
        sum_res2 = 0.0
        sum_res = 0.0
        sum_l0 = 0.0
        ever_active = torch.zeros(dict_size, dtype=torch.bool)

        with torch.no_grad():
            for shard_path in tqdm(shard_paths, desc="M1 FVU eval"):
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                N, P, D = shard.shape
                tokens = shard.reshape(N * P, D).float()
                tokens = (tokens - mean) / std

                loader = DataLoader(
                    TensorDataset(tokens),
                    batch_size=self.batch_size,
                    shuffle=False,
                    drop_last=False,
                    num_workers=0,
                )

                for (batch,) in loader:
                    batch = batch.to(device)
                    _pre, codes, x_hat = sae(batch)
                    res = batch - x_hat

                    sum_x2 += float(batch.pow(2).sum())
                    sum_x += float(batch.sum())
                    sum_res2 += float(res.pow(2).sum())
                    sum_res += float(res.sum())
                    sum_l0 += float((codes != 0).float().sum(dim=1).sum())
                    ever_active |= (codes != 0).any(dim=0).cpu()
                    n_tok += batch.shape[0]

                    del batch, _pre, codes, x_hat, res

                del shard, tokens, loader

        if device == "cuda":
            torch.cuda.empty_cache()

        var_x = sum_x2 / n_tok - (sum_x / n_tok) ** 2
        var_res = sum_res2 / n_tok - (sum_res / n_tok) ** 2
        fvu = float(var_res / var_x)
        l0 = float(sum_l0 / n_tok)
        dead_count = int((~ever_active).sum())
        dead_pct = 100.0 * dead_count / dict_size

        return {
            "fvu": fvu,
            "l0": l0,
            "dead_features": dead_count,
            "dead_pct": dead_pct,
        }
