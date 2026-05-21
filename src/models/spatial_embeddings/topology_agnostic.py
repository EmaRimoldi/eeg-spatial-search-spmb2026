"""
Variant G: Topology-agnostic / layout-robust embedding.

Design goal: test whether the benefit attributed to explicit geometry actually
comes from layout-robust design rather than xyz coordinates themselves.

Approach: set-style pooling with a shared channel encoder.
- No fixed channel identities.
- No fixed coordinate vocabulary.
- Optional: include only permutation-robust local features.

The hypothesis: if this performs as well as coords3d under layout shift,
much of the geometry benefit is actually a robustness benefit, not a
geometry-understanding benefit.

Architecture:
1. Map each channel's optional raw features (amplitude stats, etc.) through
   a shared encoder.
2. Use a learnable class token that aggregates all channels via attention.
3. The output per-channel embedding is independent of channel ordering.

For ablation purposes, this module accepts the same interface but ignores
the coordinates — it uses only the temporal representation.

When temporal features are not available at embedding time, returns a
uniform learned "generic channel" embedding.
"""

from typing import Optional
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase


class TopologyAgnosticEmbedding(SpatialEmbeddingBase):
    """
    Layout-invariant embedding via a shared channel encoder.

    Every channel gets the same generic embedding, irrespective of its
    identity or position. The model learns to use temporal features alone.

    This is the most layout-robust variant and tests whether geometry priors
    are actually necessary for EEG generalization.
    """

    def __init__(
        self,
        embed_dim: int,
        use_generic_token: bool = True,
    ):
        """
        Args:
            embed_dim: Output embedding dimension.
            use_generic_token: If True, return a shared learned token for
                               all channels. If False, return zeros.
        """
        super().__init__(embed_dim=embed_dim, name="topology_agnostic")
        self.use_generic_token = use_generic_token

        if use_generic_token:
            # A single learned "generic channel" token
            self.generic_token = nn.Parameter(torch.randn(embed_dim) * 0.02)
        else:
            self.generic_token = None

    def forward(
        self,
        channel_names: list[str],
        coords_2d: Optional[torch.Tensor],
        coords_3d: Optional[torch.Tensor],
        reference_meta: Optional[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """
        Returns:
            (B, C, embed_dim) where every channel gets the same generic token.
            If use_generic_token=False, returns None (equivalent to 'none').
        """
        C = len(channel_names)

        if not self.use_generic_token or self.generic_token is None:
            return None

        # Broadcast generic token to (B, C, embed_dim)
        token = self.generic_token.to(device)
        emb = token.unsqueeze(0).unsqueeze(0).expand(batch_size, C, -1)
        return emb
