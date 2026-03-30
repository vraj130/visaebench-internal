from abc import ABC, abstractmethod

import torch


class BackboneAdapter(ABC):
    """Abstract base class for all ViT backbone adapters."""

    def __init__(self, model_id: str, device: str = "cuda") -> None:
        self.model_id = model_id
        self.device = device
        self._load_model()

    @abstractmethod
    def _load_model(self) -> None:
        """Load HuggingFace model and processor into self.model and self.processor."""
        ...

    @property
    @abstractmethod
    def patch_count(self) -> int:
        """Expected number of patch tokens after dropping special tokens."""
        ...

    @property
    def d_model(self) -> int:
        """Hidden dimension — 768 for all ViT-B models."""
        return 768

    @torch.no_grad()
    @abstractmethod
    def extract_patch_activations(self, images: torch.Tensor, layer: int = 11) -> torch.Tensor:
        """Run forward pass, extract hidden state at `layer`, drop special tokens.

        Args:
            images: Raw image tensor [B, C, H, W] or PIL images — preprocessing is
                    handled internally via self.processor.
            layer:  0-indexed transformer block index (default 11 for ViT-B layer 11).

        Returns:
            Patch activations of shape [B, patch_count, d_model].
        """
        ...
