"""
Variant D: 3D coordinate embedding.

Maps 3D electrode positions (x, y, z) through a small MLP to the embedding space.
Analogous to coords2d but uses the full 3D unit-sphere coordinates.

Can handle novel electrode positions not in any fixed vocabulary.
"""

from typing import Optional
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase, CoordsMLP

_COORD_SCALE = 1.0


class Coords3DEmbedding(SpatialEmbeddingBase):
    """
    3D coordinate embedding via MLP.

    Takes 3D electrode positions (x, y, z) on the unit sphere and maps them
    to the embedding space through a small MLP.

    This is the primary geometry-aware variant in the ablation study.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__(embed_dim=embed_dim, name="coords3d")
        self.normalize = normalize
        self.mlp = CoordsMLP(in_dim=3, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        channel_names: list[str],
        coords_2d: Optional[torch.Tensor],
        coords_3d: Optional[torch.Tensor],
        reference_meta: Optional[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        C = len(channel_names)

        if coords_3d is None:
            coords_3d = self._lookup_coords_3d(channel_names, device)

        coords_3d = self._ensure_batch_dim(coords_3d, batch_size, expected_last_dim=3)

        if coords_3d is None:
            return torch.zeros(batch_size, C, self.embed_dim, device=device)

        coords_3d = coords_3d.to(device)

        if self.normalize:
            coords_3d = coords_3d / _COORD_SCALE
            coords_3d = coords_3d.clamp(-1, 1)

        emb = self.mlp(coords_3d)
        return emb

    def _lookup_coords_3d(
        self,
        channel_names: list[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        try:
            from src.data.coordinate_lookup import get_coords_3d
        except ImportError:
            return None

        coords = []
        for name in channel_names:
            c = get_coords_3d(name)
            if c is not None:
                coords.append(list(c))
            else:
                coords.append([0.0, 0.0, 0.0])

        return torch.tensor(coords, dtype=torch.float32, device=device)  # (C, 3)
