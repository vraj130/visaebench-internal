"""Utilities for configuring overcomplete SAEs and loading cached activations."""

import glob
import json
import os
from typing import Any

import torch


def make_sae_config(
    d_model: int = 768,
    expansion_factor: int = 16,
    k: int = 192,
    architecture: str = "batchtopk",
) -> dict[str, Any]:
    """Return the config dict needed to instantiate an overcomplete SAE.

    Args:
        d_model:          Input dimension (ViT hidden dim).
        expansion_factor: Dictionary size multiplier (dict_size = d_model * expansion_factor).
        k:                TopK sparsity — number of active features kept per batch step.
        architecture:     One of "batchtopk", "topk", "jumprelu".

    Returns:
        Dict with keys understood by the corresponding overcomplete constructor.
    """
    dict_size = d_model * expansion_factor
    config = {
        "architecture": architecture,
        "d_model": d_model,
        "expansion_factor": expansion_factor,
        "dict_size": dict_size,
        "k": k,
    }
    if architecture == "batchtopk":
        config["constructor_kwargs"] = {
            "input_shape": d_model,
            "nb_concepts": dict_size,
            "top_k": k,
            "threshold_momentum": 0.9,
        }
    elif architecture == "topk":
        config["constructor_kwargs"] = {
            "input_shape": d_model,
            "nb_concepts": dict_size,
            "top_k": k,
        }
    elif architecture == "jumprelu":
        config["constructor_kwargs"] = {
            "input_shape": d_model,
            "nb_concepts": dict_size,
        }
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}. Choose from batchtopk, topk, jumprelu.")
    return config


def load_training_data(activation_dir: str, normalize: bool = True) -> torch.Tensor:
    """Load all activation shards, reshape to patch-token rows, and optionally normalize.

    Shards are expected to be .pt files of shape [N, num_patches, d_model].
    They are concatenated along dim 0, then reshaped to [N*num_patches, d_model].
    Normalization uses mean (per-dim) and std (scalar) from stats.json in the same dir.

    Args:
        activation_dir: Directory containing shard_*.pt files and stats.json.
        normalize:      If True, subtract mean and divide by std from stats.json.

    Returns:
        Float32 tensor of shape [total_patches, d_model].
    """
    shard_paths = sorted(glob.glob(os.path.join(activation_dir, "shard_*.pt")))
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found in {activation_dir}")

    shards = [torch.load(p, map_location="cpu") for p in shard_paths]
    data = torch.cat(shards, dim=0)  # [num_images, num_patches, d_model]

    num_images, num_patches, d_model = data.shape
    data = data.reshape(num_images * num_patches, d_model).float()

    if normalize:
        stats_path = os.path.join(activation_dir, "stats.json")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"stats.json not found in {activation_dir}")
        with open(stats_path) as f:
            stats = json.load(f)
        mean = torch.tensor(stats["mean"], dtype=torch.float32)
        std = float(stats["std"])
        data = (data - mean) / std

    return data
