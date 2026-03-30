import logging
import os
from typing import Any

import torch
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from PIL import Image

logger = logging.getLogger(__name__)


class ImageFolderForCaching(Dataset):
    """Dataset that wraps torchvision.datasets.ImageFolder and applies a
    HuggingFace image processor, returning preprocessed pixel-value tensors.

    Supports both ImageNet-style layouts (class subfolders) and flat
    directories containing images directly (images are placed under a single
    synthetic subfolder named "images").

    Args:
        root_dir:  Path to the dataset root.
        processor: HuggingFace image processor (e.g. CLIPImageProcessor).
    """

    def __init__(self, root_dir: str, processor: Any) -> None:
        self.root_dir = root_dir
        self.processor = processor
        self._dataset = self._build_dataset(root_dir)

    def _build_dataset(self, root_dir: str) -> ImageFolder:
        """Return an ImageFolder, creating a one-level wrapper for flat dirs."""
        try:
            ds = ImageFolder(root=root_dir)
            if len(ds) > 0:
                return ds
        except FileNotFoundError:
            pass

        flat_images = [
            f for f in os.listdir(root_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"))
        ]
        if flat_images:
            import tempfile, shutil, pathlib
            tmp = tempfile.mkdtemp(prefix="visaebench_flat_")
            cls_dir = os.path.join(tmp, "images")
            os.makedirs(cls_dir)
            for fname in flat_images:
                src = os.path.join(root_dir, fname)
                dst = os.path.join(cls_dir, fname)
                os.symlink(os.path.abspath(src), dst)
            ds = ImageFolder(root=tmp)
            return ds

        return ImageFolder(root=root_dir)

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Return preprocessed pixel values as a float32 tensor.

        Skips and logs a warning for corrupted images, returning the next
        valid item instead.

        Args:
            idx: Dataset index.

        Returns:
            Pixel values tensor of shape [C, H, W].
        """
        for attempt in range(len(self._dataset)):
            actual_idx = (idx + attempt) % len(self._dataset)
            try:
                path, _ = self._dataset.samples[actual_idx]
                img = Image.open(path).convert("RGB")
                processed = self.processor(images=img, return_tensors="pt")
                pixel_values = processed["pixel_values"].squeeze(0)
                return pixel_values.float()
            except Exception as exc:
                logger.warning(
                    "Skipping corrupted image at index %d (%s): %s",
                    actual_idx,
                    self._dataset.samples[actual_idx][0],
                    exc,
                )
        raise RuntimeError(
            f"Could not load any image from dataset (all {len(self._dataset)} items failed)."
        )
