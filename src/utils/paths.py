"""Centralised path constants for data artifacts.

All heavy data (cached activations, model checkpoints, datasets, and
HuggingFace caches) lives on the NAS *data* volume rather than the home
volume so that the repo directory stays lightweight and code-only.

Override DATA_ROOT by setting the VISAEBENCH_DATA_ROOT environment variable.
"""

import os

DATA_ROOT: str = os.environ.get(
    "VISAEBENCH_DATA_ROOT",
    "/mnt/NAS/data/ds5725/visaebench",
)

ACTIVATION_ROOT: str = os.path.join(DATA_ROOT, "activations_1M")
CHECKPOINT_ROOT: str = os.path.join(DATA_ROOT, "checkpoints")
DATASET_ROOT: str = os.path.join(DATA_ROOT, "datasets")

# ── Redirect HuggingFace caches to the data volume ──────────────────────────
# This ensures model weights, tokenizers, and streamed datasets all land on
# /mnt/NAS/data instead of ~/.cache/huggingface on the home volume.
HF_CACHE_DIR: str = os.path.join(DATA_ROOT, "huggingface")

# Set env vars so that transformers / datasets / huggingface_hub all respect
# the redirect.  os.environ.setdefault keeps any user-level override intact.
os.environ.setdefault("HF_HOME", HF_CACHE_DIR)
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(HF_CACHE_DIR, "datasets"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(HF_CACHE_DIR, "hub"))
