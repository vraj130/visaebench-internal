import torch
from transformers import SiglipVisionModel, AutoProcessor

from .base import BackboneAdapter
from .token_utils import validate_shape

MODEL_ID = "google/siglip-base-patch16-224"


class SigLIPAdapter(BackboneAdapter):
    """Adapter for SigLIP ViT-B/16. No CLS token — all tokens are patch tokens."""

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(model_id=MODEL_ID, device=device)

    def _load_model(self) -> None:
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = SiglipVisionModel.from_pretrained(self.model_id)
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
        patch_tokens = outputs.hidden_states[layer + 1]
        validate_shape(patch_tokens, self.patch_count, self.d_model)
        return patch_tokens
