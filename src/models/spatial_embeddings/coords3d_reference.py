"""
Variant F: 3D coordinates + reference-scheme metadata embedding.

Extends coords3d with explicit reference-scheme information.
Reference type is encoded as a learned categorical embedding and concatenated
with the coordinate features before the projection MLP.

Reference types supported:
- referential       (linked mastoids, A1/A2)
- average           (average reference / CAR)
- linked_ears       (A1+A2 linked ears reference)
- bipolar           (local bipolar derivation)
- a1_reference      (single A1)
- a2_reference      (single A2)
- m1_reference      (mastoid M1)
- m2_reference      (mastoid M2)
- unknown

Scientific motivation:
The reference scheme changes the relative amplitude structure of EEG channels.
A model unaware of the reference cannot correctly interpret spatial patterns.
Hypothesis H3: Reference-awareness explains additional variance beyond coords alone.
"""

from typing import Optional
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase

REFERENCE_TYPES = [
    "referential",
    "average",
    "linked_ears",
    "bipolar",
    "a1_reference",
    "a2_reference",
    "m1_reference",
    "m2_reference",
    "unknown",
]

REFERENCE_TO_IDX = {r: i for i, r in enumerate(REFERENCE_TYPES)}
UNK_REF_IDX = REFERENCE_TO_IDX["unknown"]


class Coords3DReferenceEmbedding(SpatialEmbeddingBase):
    """
    3D coordinates + reference-scheme metadata.

    Architecture:
    1. Embed 3D coordinates via MLP → coord_feat (embed_dim // 2).
    2. Embed reference type via learned lookup → ref_feat (embed_dim // 2).
    3. Concatenate and project to embed_dim.

    Channels with unknown reference type use the 'unknown' embedding.
    """

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: Optional[int] = None,
        normalize: bool = True,
    ):
        super().__init__(embed_dim=embed_dim, name="coords3d_reference")
        self.normalize = normalize

        coord_out = embed_dim // 2
        ref_out = embed_dim - coord_out  # handles odd embed_dims

        # Coordinate pathway
        _hidden = hidden_dim or embed_dim
        self.coord_proj = nn.Sequential(
            nn.Linear(3, _hidden),
            nn.GELU(),
            nn.Linear(_hidden, coord_out),
        )

        # Reference-type pathway
        self.ref_embedding = nn.Embedding(len(REFERENCE_TYPES), ref_out)
        nn.init.trunc_normal_(self.ref_embedding.weight, std=0.02)

        # Final projection
        self.output_proj = nn.Linear(coord_out + ref_out, embed_dim)

    def _encode_reference(
        self,
        reference_meta: Optional[list[str]],
        C: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Map reference strings to indices."""
        if reference_meta is None:
            indices = [UNK_REF_IDX] * C
        else:
            indices = [
                REFERENCE_TO_IDX.get(r.lower() if r else "unknown", UNK_REF_IDX)
                for r in reference_meta
            ]
        return torch.tensor(indices, dtype=torch.long, device=device)

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
            coords_3d = torch.zeros(batch_size, C, 3, device=device)

        coords_3d = coords_3d.to(device)

        if self.normalize:
            coords_3d = coords_3d.clamp(-1, 1)

        # (B, C, embed_dim//2)
        coord_feat = self.coord_proj(coords_3d)

        # (C,) → embedding → (C, embed_dim//2)
        ref_indices = self._encode_reference(reference_meta, C, device)
        ref_feat = self.ref_embedding(ref_indices)  # (C, ref_out)

        # Expand reference to batch: (B, C, ref_out)
        ref_feat = ref_feat.unsqueeze(0).expand(batch_size, -1, -1)

        # Concatenate and project: (B, C, embed_dim)
        combined = torch.cat([coord_feat, ref_feat], dim=-1)
        emb = self.output_proj(combined)

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
            coords.append(list(c) if c is not None else [0.0, 0.0, 0.0])
        return torch.tensor(coords, dtype=torch.float32, device=device)
