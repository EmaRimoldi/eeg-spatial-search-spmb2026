"""
Model wrapper registry.

Central registry for all backbone wrappers.
Provides a uniform interface for experiment scripts to instantiate models
by name without importing specific wrappers.

Usage:
    from src.models.wrappers.registry import build_model

    model = build_model(
        backbone='reve',
        spatial_variant='coords3d',
        num_classes=4,
        freeze_policy='head_only',
    )
"""

import logging
from typing import Optional, Any

logger = logging.getLogger(__name__)

# Registry of backbone name → wrapper class
_REGISTRY: dict[str, str] = {
    "reve": "src.models.wrappers.reve_wrapper.REVEWrapper",
    "biot": "src.models.wrappers.biot_wrapper.BIOTWrapper",
    "labram": "src.models.wrappers.labram_wrapper.LaBraMWrapper",
    "cbramod": "src.models.wrappers.cbramod_wrapper.CBraModWrapper",
    "eegnet": "src.models.wrappers.eegnet_wrapper.EEGNetWrapper",
}


def build_model(
    backbone: str,
    spatial_variant: str,
    num_classes: int,
    freeze_policy: str = "head_only",
    checkpoint_path: Optional[str] = None,
    **kwargs: Any,
):
    """
    Build a model wrapper by backbone name.

    Args:
        backbone: One of 'reve', 'biot', 'labram', 'cbramod', 'eegnet'.
        spatial_variant: Spatial embedding variant name.
        num_classes: Number of downstream classes.
        freeze_policy: Freeze/unfreeze policy for backbone.
        checkpoint_path: Path to pretrained checkpoint.
        **kwargs: Additional arguments passed to the wrapper constructor.

    Returns:
        Initialized model wrapper (nn.Module subclass).
    """
    if backbone not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone: {backbone!r}. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )

    # Dynamic import to avoid mandatory dependencies at import time
    module_path, cls_name = _REGISTRY[backbone].rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, cls_name)

    common_kwargs = {
        "spatial_variant": spatial_variant,
        "num_classes": num_classes,
        "freeze_policy": freeze_policy,
    }
    if checkpoint_path is not None:
        common_kwargs["checkpoint_path"] = checkpoint_path

    # Merge extra kwargs, but let each wrapper accept what it needs
    # by using its own __init__ signature
    import inspect
    sig = inspect.signature(cls.__init__)
    valid_params = set(sig.parameters.keys()) - {"self"}
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid_params}
    common_filtered = {k: v for k, v in common_kwargs.items() if k in valid_params}
    all_kwargs = {**common_filtered, **filtered_kwargs}

    model = cls(**all_kwargs)
    logger.info(
        f"Built model: backbone={backbone}, spatial={spatial_variant}, "
        f"freeze={freeze_policy}, classes={num_classes}"
    )
    return model


def list_backbones() -> list[str]:
    """Return sorted list of registered backbone names."""
    return sorted(_REGISTRY.keys())


def list_spatial_variants() -> list[str]:
    """Return sorted list of available spatial variants."""
    from src.models.spatial_embeddings.utils import build_spatial_embedding
    return [
        "none", "channel_id", "coords2d", "coords3d",
        "coords3d_distbias", "coords3d_reference", "coords3d_rbf",
        "coords3d_geodesic_rbf", "topology_agnostic",
    ]
