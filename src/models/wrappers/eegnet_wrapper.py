"""
EEGNet wrapper — non-FM classical baseline.

EEGNet (Lawhern et al. 2018) is a lightweight CNN designed for EEG BCIs.
It serves as a sanity-check baseline to verify that foundation models
actually help in the chosen regimes.

Since EEGNet does not have pretrained weights, it is trained from scratch.
Spatial embedding variants are injected as additive channel-level embeddings
before the depthwise convolution.

Reference: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
"""

import logging
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class EEGNet(nn.Module):
    """
    EEGNet: compact EEG classification CNN.

    Architecture:
    - Temporal conv (1D conv across time)
    - Depthwise spatial conv (across channels)
    - Separable conv
    - Classifier
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        sfreq: int = 250,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        T: int = 256,
        dropout_rate: float = 0.5,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Block 1: Temporal convolution
        self.block1_temporal = nn.Sequential(
            nn.Conv2d(1, F1, (1, sfreq // 2), padding=(0, sfreq // 4), bias=False),
            nn.BatchNorm2d(F1),
        )

        # Depthwise spatial convolution
        self.block1_depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout_rate),
        )

        # Block 2: Separable convolution
        self.block2 = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout_rate),
        )

        # Compute flattened size
        self.flatten_size = self._get_flatten_size(n_channels, T, sfreq, F1, D, F2)

        # Classifier
        self.classifier = nn.Linear(self.flatten_size, n_classes)

    def _get_flatten_size(self, n_channels, T, sfreq, F1, D, F2):
        """Compute flattened feature size with dummy forward pass."""
        dummy = torch.zeros(1, 1, n_channels, T)
        x = self.block1_temporal(dummy)
        x = self.block1_depthwise(x)
        x = self.block2(x)
        return x.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)

        Returns:
            logits: (B, n_classes)
        """
        B, C, T = x.shape
        x = x.unsqueeze(1)  # (B, 1, C, T) — add filter dim
        x = self.block1_temporal(x)
        x = self.block1_depthwise(x)
        x = self.block2(x)
        x = x.view(B, -1)
        return self.classifier(x)


class EEGNetWrapper(nn.Module):
    """
    EEGNet wrapper supporting spatial embedding injection.

    For EEGNet, spatial embeddings are added as per-channel offsets
    to the input signal before the temporal convolution:
        x_augmented = x + spatial_emb_projected
    """

    def __init__(
        self,
        spatial_variant: str = "channel_id",
        n_channels: int = 22,
        num_classes: int = 4,
        sfreq: int = 250,
        T: int = 1000,
        dropout: float = 0.5,
        freeze_policy: str = "full",  # EEGNet trains from scratch; always full
        checkpoint_path=None,  # Not used; EEGNet has no pretrained weights
    ):
        super().__init__()
        self.spatial_variant = spatial_variant
        self.n_channels = n_channels
        self.num_classes = num_classes

        # Spatial embedding (projects to 1 for additive injection)
        embed_dim = 64
        from src.models.spatial_embeddings.utils import build_spatial_embedding
        self.spatial_embedding = build_spatial_embedding(
            variant=spatial_variant,
            embed_dim=embed_dim,
        )
        # Project spatial embedding to signal space
        self.spatial_proj = nn.Linear(embed_dim, 1)

        # EEGNet backbone
        self.backbone = EEGNet(
            n_channels=n_channels,
            n_classes=num_classes,
            sfreq=sfreq,
            T=T,
            dropout_rate=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
            metadata: dict with channel_names, coords_2d, coords_3d, etc.

        Returns:
            logits: (B, num_classes)
        """
        B, C, T = x.shape
        device = x.device
        channel_names = metadata.get("channel_names", [f"ch{i}" for i in range(C)])

        # Compute spatial embedding
        spatial_emb = self.spatial_embedding(
            channel_names=channel_names,
            coords_2d=metadata.get("coords_2d"),
            coords_3d=metadata.get("coords_3d"),
            reference_meta=metadata.get("reference_meta"),
            batch_size=B,
            device=device,
        )  # (B, C, embed_dim) or None

        if spatial_emb is not None:
            # Project to (B, C, 1) and add to signal as channel offset
            offset = self.spatial_proj(spatial_emb)  # (B, C, 1)
            x = x + offset.expand(-1, -1, T)

        return self.backbone(x)
