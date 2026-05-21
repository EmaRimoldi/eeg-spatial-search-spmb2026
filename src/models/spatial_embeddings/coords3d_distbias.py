"""
Variant E: 3D coordinates + pairwise distance attention bias.

Extends coords3d with a pairwise-distance bias that can be injected
into attention logits.

Design:
  1. Compute 3D embedding (same as Variant D).
  2. Compute pairwise Euclidean distances between electrodes.
  3. Map distances to attention bias using a learned RBF or linear scale.

The distance bias is intended to be added to attention scores:
    attn_logits = Q·K^T / sqrt(d) + distance_bias

If the backbone architecture does not expose attention logits,
the distance bias is instead added as an extra spatial token or ignored.
"""

from typing import Optional
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase, CoordsMLP


class Coords3DDistBiasEmbedding(SpatialEmbeddingBase):
    """
    3D coordinates + pairwise distance attention bias.

    Returns:
    - A (B, C, embed_dim) spatial token embedding (same as coords3d).
    - Via get_attention_bias(): a (B, C, C) attention bias tensor.

    The attention bias is stored as self.last_attn_bias after each forward call.
    Model wrappers that support attention bias injection should use get_attention_bias().
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        n_rbf: int = 8,
        normalize: bool = True,
    ):
        """
        Args:
            embed_dim: Output embedding dimension.
            hidden_dim: Hidden dim for coordinate MLP.
            n_rbf: Number of RBF basis functions for distance → bias mapping.
            normalize: Whether to normalize coordinates to [-1, 1].
        """
        super().__init__(embed_dim=embed_dim, name="coords3d_distbias")
        self.normalize = normalize
        self.n_rbf = n_rbf

        # Coordinate → token embedding
        self.coord_mlp = CoordsMLP(in_dim=3, embed_dim=embed_dim, hidden_dim=hidden_dim)

        # RBF centers: evenly spaced in [0, 2] (max distance on unit sphere)
        self.register_buffer(
            "rbf_centers",
            torch.linspace(0.0, 2.0, n_rbf)
        )
        # Learned scale for RBF widths
        self.rbf_log_scale = nn.Parameter(torch.zeros(n_rbf))

        # Map RBF features to scalar attention bias
        self.bias_proj = nn.Linear(n_rbf, 1, bias=False)

        # Store last attention bias for wrapper access
        self.last_attn_bias: Optional[torch.Tensor] = None

    def _compute_distance_bias(
        self,
        coords_3d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute pairwise distance attention bias.

        Args:
            coords_3d: (B, C, 3)

        Returns:
            (B, C, C) attention bias tensor.
        """
        B, C, _ = coords_3d.shape

        # Pairwise Euclidean distances: (B, C, C)
        diff = coords_3d.unsqueeze(2) - coords_3d.unsqueeze(1)  # (B, C, C, 3)
        dists = diff.norm(dim=-1)  # (B, C, C)

        # RBF features: (B, C, C, n_rbf)
        centers = self.rbf_centers.view(1, 1, 1, -1)
        scales = self.rbf_log_scale.exp().view(1, 1, 1, -1)
        rbf = torch.exp(-scales * (dists.unsqueeze(-1) - centers) ** 2)

        # Project to scalar bias: (B, C, C)
        bias = self.bias_proj(rbf).squeeze(-1)
        return bias

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
            (B, C, embed_dim) token embeddings.
            Side effect: sets self.last_attn_bias to (B, C, C).
        """
        C = len(channel_names)

        if coords_3d is None:
            coords_3d = self._lookup_coords_3d(channel_names, device)

        coords_3d = self._ensure_batch_dim(coords_3d, batch_size, expected_last_dim=3)

        if coords_3d is None:
            self.last_attn_bias = None
            return torch.zeros(batch_size, C, self.embed_dim, device=device)

        coords_3d = coords_3d.to(device)

        if self.normalize:
            coords_3d = coords_3d.clamp(-1, 1)

        # Token embedding
        emb = self.coord_mlp(coords_3d)  # (B, C, embed_dim)

        # Attention bias (stored for wrapper injection)
        self.last_attn_bias = self._compute_distance_bias(coords_3d)  # (B, C, C)

        return emb

    def get_attention_bias(self) -> Optional[torch.Tensor]:
        """Return the last-computed attention bias, or None."""
        return self.last_attn_bias

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
            coords.append(list(c) if c is not None else [0.0, 0.0, 0.0])
        return torch.tensor(coords, dtype=torch.float32, device=device)
