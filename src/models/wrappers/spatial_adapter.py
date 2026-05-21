"""
Shared spatial-coordinate adapters for foundation-model wrappers.

The wrappers have different native insertion points, but they all need the
same metadata plumbing:
- build the requested spatial variant from channel metadata
- mask and normalize per-channel spatial states
- optionally convert channel states into pairwise attention bias
"""

from __future__ import annotations

import logging
from typing import Optional, Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def channel_names_from_metadata(metadata: dict, n_channels: int) -> list[str]:
    channel_names = metadata.get("channel_names")
    if channel_names is None or len(channel_names) != n_channels:
        return [f"ch{i}" for i in range(n_channels)]
    return list(channel_names)


def reference_meta_from_metadata(metadata: dict):
    reference_meta = metadata.get("reference_meta")
    if (
        isinstance(reference_meta, list)
        and reference_meta
        and isinstance(reference_meta[0], list)
    ):
        return reference_meta[0]
    return reference_meta


def normalize_channel_mask(
    channel_mask: Optional[torch.Tensor],
    batch_size: int,
    n_channels: int,
    device: torch.device,
) -> torch.Tensor:
    if channel_mask is None:
        return torch.ones(batch_size, n_channels, dtype=torch.bool, device=device)

    channel_mask = channel_mask.to(device)
    if channel_mask.dim() == 1:
        channel_mask = channel_mask.unsqueeze(0).expand(batch_size, -1)
    elif channel_mask.dim() == 2 and channel_mask.shape[0] == 1 and batch_size > 1:
        channel_mask = channel_mask.expand(batch_size, -1)
    if channel_mask.shape != (batch_size, n_channels):
        raise ValueError(
            f"channel_mask must have shape {(batch_size, n_channels)}, "
            f"got {tuple(channel_mask.shape)}"
        )
    return channel_mask.bool()


class SpatialBackboneAdapter(nn.Module):
    """
    Build and condition spatial states for backbone-native insertion points.

    The adapter intentionally does not decide where to inject the result. REVE
    consumes raw coordinate tensors, BIOT consumes channel-token residuals,
    LaBraM consumes positional residual/replacement states, and CBraMod consumes
    spatial-attention bias. This class handles the shared coordinate metadata
    and masking behavior.
    """

    def __init__(
        self,
        variant: str,
        embed_dim: int,
        normalize: bool = True,
        pairwise_bias: bool = False,
        pair_dim: int = 64,
        dist_bias_scale: float = 0.05,
        spatial_embedding_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.variant = variant
        self.embed_dim = int(embed_dim)
        self.normalize = bool(normalize)
        self.pairwise_bias = bool(pairwise_bias)
        self.spatial_embedding_kwargs = dict(spatial_embedding_kwargs or {})

        from src.models.spatial_embeddings.utils import build_spatial_embedding

        self.spatial_embedding = build_spatial_embedding(
            variant=variant,
            embed_dim=embed_dim,
            **self.spatial_embedding_kwargs,
        )
        self.spatial_norm: Optional[nn.LayerNorm] = None
        if variant != "none" and normalize:
            self.spatial_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)

        self.pair_norm: Optional[nn.LayerNorm] = None
        self.pair_q: Optional[nn.Linear] = None
        self.pair_k: Optional[nn.Linear] = None
        self.dist_bias_gain: Optional[nn.Parameter] = None
        if pairwise_bias and variant != "none":
            hidden = max(8, min(int(pair_dim), int(embed_dim)))
            self.pair_norm = nn.LayerNorm(embed_dim, elementwise_affine=False)
            self.pair_q = nn.Linear(embed_dim, hidden, bias=False)
            self.pair_k = nn.Linear(embed_dim, hidden, bias=False)
            if variant == "coords3d_distbias":
                self.dist_bias_gain = nn.Parameter(torch.tensor(float(dist_bias_scale)))

    def build_embedding(
        self,
        metadata: dict,
        batch_size: int,
        n_channels: int,
        device: torch.device,
        channel_names: Optional[list[str]] = None,
        coords_2d: Optional[torch.Tensor] = None,
        coords_3d: Optional[torch.Tensor] = None,
        reference_meta=None,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if channel_names is None:
            channel_names = channel_names_from_metadata(metadata, n_channels)
        if reference_meta is None:
            reference_meta = reference_meta_from_metadata(metadata)
        if isinstance(reference_meta, list) and len(reference_meta) != len(channel_names):
            logger.debug(
                "Ignoring reference_meta with length %d for %d channels",
                len(reference_meta),
                len(channel_names),
            )
            reference_meta = None

        spatial_emb = self.spatial_embedding(
            channel_names=channel_names,
            coords_2d=metadata.get("coords_2d") if coords_2d is None else coords_2d,
            coords_3d=metadata.get("coords_3d") if coords_3d is None else coords_3d,
            reference_meta=reference_meta,
            batch_size=batch_size,
            device=device,
        )

        attention_bias = None
        if hasattr(self.spatial_embedding, "get_attention_bias"):
            try:
                attention_bias = self.spatial_embedding.get_attention_bias()
            except Exception as exc:
                logger.debug("Spatial embedding attention-bias lookup failed: %s", exc)

        return spatial_emb, attention_bias

    def prepare(
        self,
        spatial_emb: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if spatial_emb is None:
            return None

        mask = channel_mask.to(spatial_emb.dtype).unsqueeze(-1)
        spatial_state = spatial_emb * mask

        if self.spatial_norm is not None:
            spatial_state = self.spatial_norm(spatial_state)

        denom = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        center = (spatial_state * mask).sum(dim=1, keepdim=True) / denom
        return (spatial_state - center) * mask

    def build_pairwise_attention_bias(
        self,
        spatial_state: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        if (
            spatial_state is None
            or not self.pairwise_bias
            or self.pair_q is None
            or self.pair_k is None
        ):
            return None

        if self.pair_norm is not None:
            pair_state = self.pair_norm(spatial_state)
        else:
            pair_state = spatial_state

        q = self.pair_q(pair_state)
        k = self.pair_k(pair_state)
        scale = q.shape[-1] ** -0.5 if q.shape[-1] > 0 else 1.0
        pair_bias = torch.matmul(q, k.transpose(-1, -2)) * scale
        pair_bias = torch.tanh(pair_bias)
        pair_bias = pair_bias - pair_bias.mean(dim=-1, keepdim=True)

        if attention_bias is not None:
            dist_bias = attention_bias.to(device=pair_bias.device, dtype=pair_bias.dtype)
            if dist_bias.dim() == 2:
                dist_bias = dist_bias.unsqueeze(0)
            dist_bias = dist_bias - dist_bias.mean(dim=-1, keepdim=True)
            if self.dist_bias_gain is not None:
                dist_bias = self.dist_bias_gain * dist_bias
            pair_bias = pair_bias + dist_bias

        invalid = ~channel_mask.bool()
        pair_bias = pair_bias.masked_fill(invalid.unsqueeze(-1), 0.0)
        pair_bias = pair_bias.masked_fill(invalid.unsqueeze(1), -1e4)
        return pair_bias
