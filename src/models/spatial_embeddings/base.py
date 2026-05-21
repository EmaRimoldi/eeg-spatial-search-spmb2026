"""
Base class for all spatial embedding modules.

All variants must implement the same interface so they can be
swapped transparently in model wrappers.

Interface:
    forward(channel_names, coords_2d, coords_3d, reference_meta, batch_size, device)
    -> Tensor of shape (B, C, embed_dim) or None (for the 'none' variant)
"""

from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn


class SpatialEmbeddingBase(ABC, nn.Module):
    """
    Abstract base class for all spatial embedding variants.

    Every variant must:
    1. Accept the full metadata dict (even if it ignores some fields).
    2. Return a tensor of shape (B, C, embed_dim) or None.
    3. Handle missing coordinates gracefully (zero-fill or mask).
    4. Be deterministic in eval mode.
    """

    def __init__(self, embed_dim: int, name: str):
        super().__init__()
        self.embed_dim = embed_dim
        self.name = name

    @abstractmethod
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
        Compute spatial embedding for a batch of EEG recordings.

        Args:
            channel_names: List of canonical channel names, length C.
            coords_2d: 2D coordinates tensor, shape (B, C, 2) or (C, 2).
                       May be None if not available.
            coords_3d: 3D coordinates tensor, shape (B, C, 3) or (C, 3).
                       May be None if not available.
            reference_meta: List of reference scheme strings, length C.
                            E.g. ['referential', 'referential', 'average', ...].
                            May be None.
            batch_size: B — number of recordings in the batch.
            device: Target device.

        Returns:
            Tensor of shape (B, C, embed_dim) to be added to channel tokens,
            or None if this variant provides no spatial signal.
        """
        ...

    def _ensure_batch_dim(
        self,
        coords: Optional[torch.Tensor],
        batch_size: int,
        expected_last_dim: int,
    ) -> Optional[torch.Tensor]:
        """
        Ensure coords has a batch dimension.

        Handles (C, D) → (B, C, D) expansion.
        """
        if coords is None:
            return None
        if coords.dim() == 2:
            # (C, D) → (B, C, D)
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        assert coords.dim() == 3, f"Expected 3D tensor, got shape {coords.shape}"
        assert coords.shape[-1] == expected_last_dim, (
            f"Expected last dim {expected_last_dim}, got {coords.shape[-1]}"
        )
        return coords.float()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r}, embed_dim={self.embed_dim})"


class CoordsMLP(nn.Module):
    """
    Small MLP that maps raw coordinates to the embedding space.

    Used by coords2d, coords3d, and reference variants.
    Architecture: Linear → GELU → Linear
    """

    def __init__(self, in_dim: int, embed_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
