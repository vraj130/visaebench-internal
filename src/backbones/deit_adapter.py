import torch
from transformers import AutoModel, AutoImageProcessor

from .base import BackboneAdapter
from .token_utils import drop_cls, validate_shape

MODEL_ID = "facebook/deit-base-patch16-224"


class DeiTAdapter(BackboneAdapter):
    """Adapter for DeiT ViT-B/16.

    Uses AutoModel which loads the checkpoint as ViTModel, correctly
    mapping the ``vit.``-prefixed weights. The resulting model has
    CLS + 196 patch tokens (no distillation token), so we drop only CLS.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(model_id=MODEL_ID, device=device)

    def _load_model(self) -> None:
        self.processor = AutoImageProcessor.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
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
