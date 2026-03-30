import torch
from transformers import Dinov2Model, AutoImageProcessor

from .base import BackboneAdapter
from .token_utils import drop_cls, validate_shape

MODEL_ID = "facebook/dinov2-base"


class DINOv2Adapter(BackboneAdapter):
    """Adapter for DINOv2 ViT-B/14. Drops CLS token at position 0.
    facebook/dinov2-base has no register tokens (seq_len = 1 + 256 = 257).
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(model_id=MODEL_ID, device=device)

    def _load_model(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = Dinov2Model.from_pretrained(self.model_id)
        self.model.eval()
        self.model.to(self.device)

    @property
    def patch_count(self) -> int:
        return 256

    @torch.no_grad()
    def extract_patch_activations(self, images, layer: int = 11) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[layer + 1]
        patch_tokens = drop_cls(hidden_state)
        validate_shape(patch_tokens, self.patch_count, self.d_model)
        return patch_tokens
