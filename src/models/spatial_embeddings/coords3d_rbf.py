"""
3D anchor-RBF spatial embeddings.

These variants encode each electrode by its distances to a small fixed set of
anchors on the unit sphere. The Euclidean version is a compact distance-feature
baseline; the geodesic version uses spherical arc length and is a simple
manifold-aware alternative for scalp coordinates.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase


def _anchor_table(anchor_mode: str, device: torch.device) -> torch.Tensor:
    axes = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
        device=device,
    )
    if anchor_mode == "axes":
        return axes
    if anchor_mode == "axes_diagonal":
        diag = torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, 1.0, -1.0],
                [1.0, -1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, 1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
                [-1.0, -1.0, -1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        diag = diag / diag.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        return torch.cat([axes, diag], dim=0)
    raise ValueError("anchor_mode must be 'axes' or 'axes_diagonal'")


class _Coords3DAnchorRBFBase(SpatialEmbeddingBase):
    def __init__(
        self,
        embed_dim: int,
        name: str,
        hidden_dim: Optional[int] = None,
        n_rbf: int = 8,
        rbf_sigma: Optional[float] = None,
        anchor_mode: str = "axes",
        include_coords: bool = True,
        normalize: bool = True,
        geodesic: bool = False,
    ):
        super().__init__(embed_dim=embed_dim, name=name)
        if n_rbf < 1:
            raise ValueError("n_rbf must be >= 1")
        self.n_rbf = int(n_rbf)
        self.rbf_sigma = rbf_sigma
        self.anchor_mode = anchor_mode
        self.include_coords = bool(include_coords)
        self.normalize = bool(normalize)
        self.geodesic = bool(geodesic)

        max_dist = torch.pi if self.geodesic else 2.0
        self.register_buffer("rbf_centers", torch.linspace(0.0, float(max_dist), self.n_rbf))

        n_anchors = 14 if anchor_mode == "axes_diagonal" else 6
        in_dim = n_anchors * self.n_rbf + (3 if self.include_coords else 0)
        _hidden = hidden_dim or embed_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, _hidden),
            nn.GELU(),
            nn.Linear(_hidden, embed_dim),
        )

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
            coord = get_coords_3d(name)
            coords.append(list(coord) if coord is not None else [0.0, 0.0, 0.0])
        return torch.tensor(coords, dtype=torch.float32, device=device)

    def _prepare_coords(
        self,
        coords_3d: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        coords_3d = coords_3d.to(device).float()
        if self.normalize:
            coords_3d = coords_3d.clamp(-1.0, 1.0)
        if self.geodesic:
            norms = coords_3d.norm(dim=-1, keepdim=True)
            coords_3d = coords_3d / norms.clamp_min(1e-6)
            coords_3d = torch.where(norms > 0, coords_3d, torch.zeros_like(coords_3d))
        return coords_3d

    def _anchor_distances(self, coords_3d: torch.Tensor, device: torch.device) -> torch.Tensor:
        anchors = _anchor_table(self.anchor_mode, device=device)
        if self.geodesic:
            dots = torch.matmul(coords_3d, anchors.transpose(0, 1)).clamp(-1.0, 1.0)
            return torch.acos(dots)
        return torch.cdist(coords_3d, anchors.unsqueeze(0).expand(coords_3d.shape[0], -1, -1))

    def forward(
        self,
        channel_names: list[str],
        coords_2d: Optional[torch.Tensor],
        coords_3d: Optional[torch.Tensor],
        reference_meta: Optional[list[str]],
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        channels = len(channel_names)
        if coords_3d is None:
            coords_3d = self._lookup_coords_3d(channel_names, device)

        coords_3d = self._ensure_batch_dim(coords_3d, batch_size, expected_last_dim=3)
        if coords_3d is None:
            coords_3d = torch.zeros(batch_size, channels, 3, device=device)

        coords_3d = self._prepare_coords(coords_3d, device)
        dists = self._anchor_distances(coords_3d, device)

        centers = self.rbf_centers.to(device=device, dtype=dists.dtype).view(1, 1, 1, -1)
        if self.rbf_sigma is None:
            sigma = (centers[..., 1] - centers[..., 0]).abs() if self.n_rbf > 1 else torch.tensor(1.0, device=device)
            sigma = sigma.clamp_min(1e-3)
        else:
            sigma = torch.tensor(float(self.rbf_sigma), device=device, dtype=dists.dtype).clamp_min(1e-3)
        rbf = torch.exp(-0.5 * torch.square((dists.unsqueeze(-1) - centers) / sigma))
        features = rbf.flatten(start_dim=2)
        if self.include_coords:
            features = torch.cat([coords_3d, features], dim=-1)
        return self.proj(features)


class Coords3DRBFEmbedding(_Coords3DAnchorRBFBase):
    """Euclidean distance-to-anchor RBF embedding."""

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(embed_dim=embed_dim, name="coords3d_rbf", geodesic=False, **kwargs)


class Coords3DGeodesicRBFEmbedding(_Coords3DAnchorRBFBase):
    """Spherical geodesic distance-to-anchor RBF embedding."""

    def __init__(self, embed_dim: int, **kwargs):
        super().__init__(embed_dim=embed_dim, name="coords3d_geodesic_rbf", geodesic=True, **kwargs)
