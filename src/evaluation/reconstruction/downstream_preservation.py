"""M2: Downstream Preservation.

Measures how much task-relevant information the SAE preserves by comparing
linear probe accuracy on original (normalised) activations vs SAE-reconstructed
activations.  A preservation ratio near 1.0 means the SAE retains nearly all
information needed for ImageNet classification.
"""

import os
import tempfile

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from src.evaluation.base import MetricBase


class DownstreamPreservation(MetricBase):
    """M2: Downstream preservation via linear probe comparison.

    Results dict keys:
        accuracy_original        -- probe accuracy on original activations
        accuracy_reconstructed   -- probe accuracy on SAE reconstructions
        preservation_ratio       -- recon / orig (1.0 = perfect)
        accuracy_gap             -- orig - recon (0.0 = perfect)
        num_images               -- total images evaluated
    """

    def __init__(self, test_size: float = 0.2, encode_batch_size: int = 512):
        self.test_size = test_size
        self.encode_batch_size = encode_batch_size

    def evaluate(
        self,
        sae: torch.nn.Module,
        shard_paths: list[str],
        mean: torch.Tensor,
        std: float,
        device: str,
        **kwargs,
    ) -> dict:
        """Compute M2 downstream preservation.

        Required kwargs:
            dict_size (int): SAE dictionary size (unused but kept for interface).
            labels (np.ndarray): Class labels aligned with val shards.
        Optional kwargs:
            seed (int): Random seed (default 42).
        """
        labels = np.asarray(kwargs["labels"], dtype=np.int64)
        seed: int = kwargs.get("seed", 42)

        sae.eval()
        d_model = mean.shape[0]

        # Count total images for memmap pre-allocation
        shard_sizes = []
        for sp in shard_paths:
            s = torch.load(sp, map_location="cpu", weights_only=True)
            shard_sizes.append(s.shape[0])
            del s
        total_images = sum(shard_sizes)

        # Two memmaps: original pooled and reconstructed pooled
        tmp_orig = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_recon = tempfile.NamedTemporaryFile(suffix=".dat", delete=False)
        tmp_orig_path, tmp_recon_path = tmp_orig.name, tmp_recon.name
        tmp_orig.close(); tmp_recon.close()

        orig_mmap = np.memmap(tmp_orig_path, dtype=np.float32, mode="w+",
                              shape=(total_images, d_model))
        recon_mmap = np.memmap(tmp_recon_path, dtype=np.float32, mode="w+",
                               shape=(total_images, d_model))

        # Process shards: encode, reconstruct, max-pool, store
        group_size = 16
        write_idx = 0

        with torch.no_grad():
            for shard_path in tqdm(shard_paths, desc="M2 encoding"):
                shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                N, P, D = shard.shape

                for g_start in range(0, N, group_size):
                    g_end = min(g_start + group_size, N)
                    group = shard[g_start:g_end].float()  # [G, P, D]
                    G = group.shape[0]

                    # Normalise
                    normed = (group - mean) / std  # [G, P, D]

                    # Max-pool original normalised activations
                    orig_pooled = normed.max(dim=1).values  # [G, D]

                    # Reconstruct through SAE in sub-batches
                    tokens = normed.reshape(G * P, D)
                    recon_chunks: list[torch.Tensor] = []
                    for s in range(0, tokens.shape[0], self.encode_batch_size):
                        batch = tokens[s : s + self.encode_batch_size].to(device)
                        _pre, _codes, x_hat = sae(batch)
                        recon_chunks.append(x_hat.cpu())
                        del batch, _pre, _codes, x_hat

                    recon_all = torch.cat(recon_chunks, dim=0)  # [G*P, D]
                    recon_all = recon_all.reshape(G, P, D)
                    recon_pooled = recon_all.max(dim=1).values  # [G, D]

                    orig_mmap[write_idx : write_idx + G] = orig_pooled.numpy()
                    recon_mmap[write_idx : write_idx + G] = recon_pooled.numpy()
                    write_idx += G

                    del group, normed, tokens, recon_chunks, recon_all
                    del orig_pooled, recon_pooled

                del shard

        if device == "cuda":
            torch.cuda.empty_cache()

        orig_mmap.flush(); recon_mmap.flush()

        # Re-open read-only
        orig_data = np.memmap(tmp_orig_path, dtype=np.float32, mode="r",
                              shape=(total_images, d_model))
        recon_data = np.memmap(tmp_recon_path, dtype=np.float32, mode="r",
                               shape=(total_images, d_model))

        # Align labels
        if total_images < len(labels):
            print(f"[M2] Using first {total_images} of {len(labels)} labels")
            labels = labels[:total_images]

        # Stratified train/test split (identical for both probes)
        idx_train, idx_test, y_train, y_test = train_test_split(
            np.arange(total_images), labels,
            test_size=self.test_size,
            stratify=labels,
            random_state=seed,
        )

        # Probe on original activations
        print("[M2] Training probe on original activations...")
        clf_orig = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                                      random_state=seed)
        clf_orig.fit(orig_data[idx_train], y_train)
        acc_orig = float(clf_orig.score(orig_data[idx_test], y_test))
        del clf_orig

        # Probe on reconstructed activations
        print("[M2] Training probe on reconstructed activations...")
        clf_recon = LogisticRegression(solver="lbfgs", max_iter=1000, C=1.0,
                                       random_state=seed)
        clf_recon.fit(recon_data[idx_train], y_train)
        acc_recon = float(clf_recon.score(recon_data[idx_test], y_test))
        del clf_recon

        # Cleanup memmaps
        del orig_data, recon_data
        import atexit
        atexit.register(lambda: os.unlink(tmp_orig_path))
        atexit.register(lambda: os.unlink(tmp_recon_path))

        preservation = acc_recon / acc_orig if acc_orig > 0 else 0.0

        return {
            "accuracy_original": acc_orig,
            "accuracy_reconstructed": acc_recon,
            "preservation_ratio": preservation,
            "accuracy_gap": acc_orig - acc_recon,
            "num_images": total_images,
        }
