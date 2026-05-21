"""
Variant A: No spatial embedding.

Returns None — no spatial signal is added to channel tokens.
Serves as the true ablation floor.
"""

from typing import Optional
import torch

from .base import SpatialEmbeddingBase


class NoSpatialEmbedding(SpatialEmbeddingBase):
    """
    No spatial signal.

    The model must rely entirely on temporal features and token order.
    This is the most conservative baseline.
    """

    def __init__(self, embed_dim: int):
        super().__init__(embed_dim=embed_dim, name="none")

    def forward(
        self,
        channel_names: list[str],
        coords_2d: Optional[torch.Tensor],
        coords_3d: Optional[torch.Tensor],
        reference_meta: Optional[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        return None
