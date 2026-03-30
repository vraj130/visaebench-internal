from .base import BackboneAdapter
from .clip_adapter import CLIPAdapter
from .dinov2_adapter import DINOv2Adapter
from .siglip_adapter import SigLIPAdapter
from .mae_adapter import MAEAdapter
from .deit_adapter import DeiTAdapter

_BACKBONE_MAP = {
    "clip_vitb16": CLIPAdapter,
    "dinov2_vitb14": DINOv2Adapter,
    "siglip_vitb16": SigLIPAdapter,
    "mae_vitb16": MAEAdapter,
    "deit_vitb16": DeiTAdapter,
}


def load_backbone(backbone_name: str, device: str = "cuda") -> BackboneAdapter:
    """Instantiate a BackboneAdapter by name.

    Args:
        backbone_name: One of "clip_vitb16", "dinov2_vitb14", "siglip_vitb16",
                       "mae_vitb16", "deit_vitb16".
        device: "cuda" or "cpu".

    Returns:
        An initialized BackboneAdapter with the model loaded on `device`.
    """
    if backbone_name not in _BACKBONE_MAP:
        raise ValueError(
            f"Unknown backbone '{backbone_name}'. "
            f"Valid options: {list(_BACKBONE_MAP.keys())}"
        )
    return _BACKBONE_MAP[backbone_name](device=device)


__all__ = [
    "load_backbone",
    "BackboneAdapter",
    "CLIPAdapter",
    "DINOv2Adapter",
    "SigLIPAdapter",
    "MAEAdapter",
    "DeiTAdapter",
]
