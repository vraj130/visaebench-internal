import torch


def drop_cls(hidden_state: torch.Tensor) -> torch.Tensor:
    """Remove CLS token at position 0. Input: [B, 1+N, D] → Output: [B, N, D]"""
    return hidden_state[:, 1:, :]


def drop_cls_and_distillation(hidden_state: torch.Tensor) -> torch.Tensor:
    """Remove CLS at position 0 and distillation token at position 1. Input: [B, 2+N, D] → Output: [B, N, D]"""
    return hidden_state[:, 2:, :]


def drop_cls_and_registers(hidden_state: torch.Tensor, num_registers: int) -> torch.Tensor:
    """Remove CLS at position 0 and num_registers register tokens at positions 1..num_registers.
    Input: [B, 1+num_registers+N, D] → Output: [B, N, D]"""
    return hidden_state[:, 1 + num_registers:, :]


def validate_shape(tensor: torch.Tensor, expected_patches: int, expected_dim: int) -> None:
    """Assert that tensor has shape [..., expected_patches, expected_dim]."""
    actual_patches = tensor.shape[-2]
    actual_dim = tensor.shape[-1]
    assert actual_patches == expected_patches, (
        f"Expected {expected_patches} patch tokens, got {actual_patches}"
    )
    assert actual_dim == expected_dim, (
        f"Expected hidden dim {expected_dim}, got {actual_dim}"
    )
