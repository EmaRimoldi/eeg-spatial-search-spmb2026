"""
Shared utilities for spatial embedding modules.
"""

from typing import Optional, Union
import torch
import torch.nn as nn

from .base import SpatialEmbeddingBase


def build_spatial_embedding(
    variant: str,
    embed_dim: int,
    **kwargs,
) -> SpatialEmbeddingBase:
    """
    Factory function: build a spatial embedding by variant name.

    Args:
        variant: One of 'none', 'channel_id', 'coords2d', 'coords3d',
                 'coords3d_distbias', 'coords3d_reference', 'coords3d_rbf',
                 'coords3d_geodesic_rbf', 'topology_agnostic'.
        embed_dim: Embedding output dimension.
        **kwargs: Additional arguments passed to the constructor.

    Returns:
        Initialized SpatialEmbeddingBase subclass.
    """
    from .none import NoSpatialEmbedding
    from .channel_id import ChannelIDEmbedding
    from .coords2d import Coords2DEmbedding
    from .coords3d import Coords3DEmbedding
    from .coords3d_distbias import Coords3DDistBiasEmbedding
    from .coords3d_reference import Coords3DReferenceEmbedding
    from .coords3d_rbf import Coords3DGeodesicRBFEmbedding, Coords3DRBFEmbedding
    from .topology_agnostic import TopologyAgnosticEmbedding

    registry = {
        "none": NoSpatialEmbedding,
        "channel_id": ChannelIDEmbedding,
        "coords2d": Coords2DEmbedding,
        "coords3d": Coords3DEmbedding,
        "coords3d_distbias": Coords3DDistBiasEmbedding,
        "coords3d_reference": Coords3DReferenceEmbedding,
        "coords3d_rbf": Coords3DRBFEmbedding,
        "coords3d_geodesic_rbf": Coords3DGeodesicRBFEmbedding,
        "topology_agnostic": TopologyAgnosticEmbedding,
    }

    if variant not in registry:
        raise ValueError(
            f"Unknown spatial variant: {variant!r}. "
            f"Choose from: {sorted(registry.keys())}"
        )

    cls = registry[variant]
    return cls(embed_dim=embed_dim, **kwargs)


def prepare_metadata_batch(
    channel_names: list[str],
    batch_size: int,
    reference_type: str = "unknown",
    device: Union[str, torch.device] = "cpu",
) -> dict:
    """
    Build a standard metadata dictionary for a batch.

    Looks up coordinates from the canonical table.
    Returns a dict suitable for passing to model wrappers.
    """
    from src.data.coordinate_lookup import lookup_coordinates

    coords_2d_list = []
    coords_3d_list = []
    ref_list = []

    for name in channel_names:
        c = lookup_coordinates(name)
        if c is not None:
            coords_2d_list.append([c.x_2d, c.y_2d])
            coords_3d_list.append([c.x_3d, c.y_3d, c.z_3d])
        else:
            coords_2d_list.append([0.0, 0.0])
            coords_3d_list.append([0.0, 0.0, 0.0])
        ref_list.append(reference_type)

    coords_2d = torch.tensor(coords_2d_list, dtype=torch.float32, device=device)
    coords_3d = torch.tensor(coords_3d_list, dtype=torch.float32, device=device)

    return {
        "channel_names": channel_names,
        "coords_2d": coords_2d,
        "coords_3d": coords_3d,
        "reference_meta": ref_list,
        "channel_mask": torch.ones(batch_size, len(channel_names), dtype=torch.bool, device=device),
    }
