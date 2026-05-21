"""
Variant C: 2D coordinate embedding.

Maps 2D electrode positions (x, y) through a small MLP to the embedding space.
Coordinates are normalized to [-1, 1] before MLP.

Can handle novel electrode positions that are not in any fixed vocabulary.
"""

from typing import Optional
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase, CoordsMLP

# Expected coordinate range for normalization (unit sphere projection)
_COORD_SCALE = 1.0


class Coords2DEmbedding(SpatialEmbeddingBase):
    """
    2D coordinate embedding via MLP.

    Takes 2D topographic electrode positions (x_2d, y_2d) and maps them
    to the embedding space through a small MLP.

    Handles missing coordinates by returning a zero embedding for those channels.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        """
        Args:
            embed_dim: Output embedding dimension.
            hidden_dim: Hidden dimension in MLP. Defaults to embed_dim.
            normalize: Whether to normalize input coords to [-1, 1].
        """
        super().__init__(embed_dim=embed_dim, name="coords2d")
        self.normalize = normalize
        self.mlp = CoordsMLP(in_dim=2, embed_dim=embed_dim, hidden_dim=hidden_dim)

    def forward(
        self,
        channel_names: list[str],
        coords_2d: Optional[torch.Tensor],
        coords_3d: Optional[torch.Tensor],
        reference_meta: Optional[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Returns:
            Tensor of shape (B, C, embed_dim).
        """
        C = len(channel_names)

        if coords_2d is None:
            # Fall back to coordinate lookup from the internal table
            coords_2d = self._lookup_coords_2d(channel_names, device)

        # Ensure batch dimension: (B, C, 2)
        coords_2d = self._ensure_batch_dim(coords_2d, batch_size, expected_last_dim=2)

        if coords_2d is None:
            # Still None — no coords available; return zeros
            return torch.zeros(batch_size, C, self.embed_dim, device=device)

        coords_2d = coords_2d.to(device)

        # Normalize to [-1, 1]
        if self.normalize:
            coords_2d = coords_2d / _COORD_SCALE
            coords_2d = coords_2d.clamp(-1, 1)

        # (B, C, 2) → MLP → (B, C, embed_dim)
        emb = self.mlp(coords_2d)
        return emb

    def _lookup_coords_2d(
        self,
        channel_names: list[str],
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Look up 2D coordinates from the internal table."""
        try:
            from src.data.coordinate_lookup import get_coords_2d
        except ImportError:
            return None

        coords = []
        for name in channel_names:
            c = get_coords_2d(name)
            if c is not None:
                coords.append(list(c))
            else:
                coords.append([0.0, 0.0])  # zero-fill unknown

        return torch.tensor(coords, dtype=torch.float32, device=device)  # (C, 2)
