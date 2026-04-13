"""OOD dataset loaders for M5: Cross-Domain Generalization.

Provides uniform loading of out-of-distribution datasets as
(list[PIL.Image], np.ndarray of labels) pairs.
"""

import numpy as np
from PIL import Image


def load_eurosat(max_images: int = 10000) -> tuple[list[Image.Image], np.ndarray]:
    """Load EuroSAT satellite imagery dataset.

    Tries torchvision first, falls back to HuggingFace if download fails.

    Returns:
        images: list of PIL.Image (RGB)
        labels: np.ndarray of int class labels (0-9, 10 classes)
    """
    images, labels = _load_eurosat_torchvision(max_images)
    if images is not None:
        return images, labels
    # Fallback to HuggingFace
    return _load_eurosat_huggingface(max_images)


def _load_eurosat_torchvision(max_images: int) -> tuple[list | None, np.ndarray | None]:
    """Try loading EuroSAT via torchvision.datasets.EuroSAT."""
    try:
        import torchvision.datasets as tvd
        import src.utils.paths as paths  # noqa: F401 — sets HF_HOME

        root = paths.DATASET_ROOT
        ds = tvd.EuroSAT(root=root, download=True)

        n = min(len(ds), max_images)
        images = []
        labels = []
        for i in range(n):
            img, label = ds[i]
            images.append(img.convert("RGB"))
            labels.append(label)

        return images, np.array(labels, dtype=np.int64)
    except Exception as e:
        print(f"[ood_datasets] torchvision EuroSAT failed ({e}), trying HuggingFace...")
        return None, None


def _load_eurosat_huggingface(max_images: int) -> tuple[list[Image.Image], np.ndarray]:
    """Load EuroSAT from HuggingFace datasets."""
    import src.utils.paths  # noqa: F401 — sets HF_HOME
    from datasets import load_dataset

    ds = load_dataset("tanganke/eurosat", split="train")

    n = min(len(ds), max_images)
    images = []
    labels = []
    for i in range(n):
        row = ds[i]
        images.append(row["image"].convert("RGB"))
        labels.append(row["label"])

    return images, np.array(labels, dtype=np.int64)


def load_inaturalist(
    max_images: int = 10000,
    group_by: str = "kingdom",
) -> tuple[list[Image.Image], np.ndarray]:
    """Load iNaturalist 2021 Mini dataset grouped by top-level taxonomy.

    Not yet implemented — requires large download and complex taxonomy
    mapping. Use EuroSAT for cross-domain evaluation.

    Raises:
        NotImplementedError: Always. Use load_eurosat() instead.
    """
    raise NotImplementedError(
        "iNaturalist loader not yet implemented. "
        "Use load_eurosat() for cross-domain evaluation. "
        "iNaturalist requires ~10K fine-grained classes to be grouped by "
        f"'{group_by}' taxonomy level, which needs additional metadata parsing."
    )
