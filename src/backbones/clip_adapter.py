import torch
from transformers import CLIPVisionModel, CLIPImageProcessor

from .base import BackboneAdapter
from .token_utils import drop_cls, validate_shape

MODEL_ID = "openai/clip-vit-base-patch16"


class CLIPAdapter(BackboneAdapter):
    """Adapter for CLIP ViT-B/16. Drops CLS token at position 0."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(model_id=MODEL_ID, device=device)

    def _load_model(self) -> None:
        self.processor = CLIPImageProcessor.from_pretrained(self.model_id)
        self.model = CLIPVisionModel.from_pretrained(self.model_id)
        self.model.eval()
        self.model.to(self.device)

    @property
    def patch_count(self) -> int:
        return 196

    @torch.no_grad()
    def extract_patch_activations(self, images, layer: int = 11) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_state = outputs.hidden_states[layer + 1]
        patch_tokens = drop_cls(hidden_state)
        validate_shape(patch_tokens, self.patch_count, self.d_model)
        return patch_tokens
