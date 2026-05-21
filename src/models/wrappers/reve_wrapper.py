"""
REVE model wrapper.

Wraps the REVE pretrained EEG foundation model with spatial variant control.

REVE architecture:
- Input: EEG signal (B, C, T) + position tensor (B, C, 3) (x, y, z coordinates)
- Positional encoding: FourierEmb4D — computes Fourier embeddings of (x,y,z,t)
- Backbone: Transformer (depth=22, embed_dim=512)
- Pooling: attention_pooling → (B, embed_dim)

Spatial variant → pos tensor mapping:
- 'none':                pos = zeros                    (ablate all spatial info)
- 'channel_id':          pos = canonical 3D coords via name lookup
- 'coords2d':            pos = (x, y, 0)
- 'coords3d':            pos = (x, y, z)            ← natural REVE input
- 'coords3d_distbias':   pos = (x, y, z)            ← same as coords3d for REVE
- 'coords3d_reference':  pos = (x, y, z)            ← coords + reference embedding added
- 'coords3d_rbf':        pos = (x, y, z)            ← same native coordinates for REVE
- 'coords3d_geodesic_rbf': pos = (x, y, z)          ← same native coordinates for REVE
- 'topology_agnostic':   pos = zeros                    (same as none, robustness probe)
- 'reve_default':        pos = coords_3d if available, else zeros

Reference encoding (coords3d_reference only): a reference-type embedding is added to
the classifier head input, not the backbone input (cannot inject into REVE's FourierEmb4D
without modifying the backbone).

Freeze policies:
- 'frozen' / 'head_only': backbone frozen, only head + extra spatial module train
- 'partial': unfreeze last N transformer blocks
- 'full': unfreeze all
"""

import sys
import logging
from typing import Optional, Union, Literal
from pathlib import Path

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_REVE_HF_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "external" / "reve_eeg" / "hf" / "reve-base"
)

# Variants that use zero position (ablate spatial information)
_ZERO_POS_VARIANTS = {"none", "topology_agnostic"}

# Variants that use 3D coordinates
_COORDS3D_VARIANTS = {
    "coords3d",
    "coords3d_distbias",
    "coords3d_reference",
    "coords3d_rbf",
    "coords3d_geodesic_rbf",
    "reve_default",
}

# Variants that use 2D coordinates (z=0)
_COORDS2D_VARIANTS = {"coords2d"}

# Variants that look up canonical coords from channel name
_CHANNEL_ID_VARIANTS = {"channel_id"}


class REVEWrapper(nn.Module):
    """
    Wrapper around the REVE pretrained EEG foundation model.

    Controls how spatial information (pos tensor) is constructed and fed
    to REVE's FourierEmb4D positional encoding as part of the ablation study.

    Usage:
        wrapper = REVEWrapper(
            spatial_variant='coords3d',
            embed_dim=512,
            num_classes=4,
            freeze_policy='head_only',
        )
        logits = wrapper(x, metadata)  # x: (B, C, T), metadata: dict

    Args:
        spatial_variant: Spatial embedding variant controlling the pos tensor.
        embed_dim: REVE embedding dimension (512 for reve-base).
        num_classes: Downstream classification classes.
        freeze_policy: Backbone freeze strategy.
        checkpoint_path: Local safetensors path, or None to use HuggingFace.
        n_partial_layers: For 'partial' freeze: number of layers to keep unfrozen.
        dropout: Dropout rate in the classification head.
    """

    def __init__(
        self,
        spatial_variant: str = "coords3d",
        embed_dim: int = 512,
        num_classes: int = 4,
        freeze_policy: Literal["frozen", "head_only", "partial", "full"] = "head_only",
        checkpoint_path: Optional[Union[str, Path]] = None,
        n_partial_layers: int = 4,
        dropout: float = 0.1,
        smoke_test: bool = False,
        allow_stub_backbone: bool = False,
    ):
        super().__init__()
        self.spatial_variant = spatial_variant
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.freeze_policy = freeze_policy
        self.smoke_test = bool(smoke_test)
        self.allow_stub_backbone = bool(allow_stub_backbone or smoke_test)

        # Optional reference-type embedding for coords3d_reference variant
        self._ref_type_vocab = ["referential", "average", "linked", "unknown"]
        self._ref_embedding: Optional[nn.Embedding] = None
        if spatial_variant == "coords3d_reference":
            self._ref_embedding = nn.Embedding(len(self._ref_type_vocab), embed_dim)
            nn.init.normal_(self._ref_embedding.weight, std=0.02)

        # Load REVE backbone
        self.backbone = self._load_reve_backbone(checkpoint_path)

        # Classification head (defined before freeze policy)
        head_input_dim = embed_dim
        self.classifier_head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, num_classes),
        )

        # Apply freeze policy last (needs classifier_head to exist)
        self._apply_freeze_policy(freeze_policy, n_partial_layers)

        logger.info(
            f"REVEWrapper: variant={spatial_variant}, freeze={freeze_policy}, "
            f"classes={num_classes}, backbone={'real' if not isinstance(self.backbone, _REVEStub) else 'stub'}"
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_reve_backbone(
        self,
        checkpoint_path: Optional[Union[str, Path]],
    ) -> nn.Module:
        """
        Load REVE backbone.

        Priority:
        1. Local safetensors checkpoint  (if checkpoint_path points to .safetensors)
        2. HuggingFace download          (brain-bzh/reve-base)
        3. Architecture-only (random weights) — for testing
        4. Stub                           — if all else fails
        """
        if self.smoke_test:
            logger.warning("REVE smoke_test=True; using stub backbone.")
            return _REVEStub(self.embed_dim)

        if not _REVE_HF_PATH.exists():
            return self._stub_or_raise(f"REVE HF files not found at {_REVE_HF_PATH}")

        # Use transformers AutoModel to load from local HF directory.
        # This handles relative imports inside modeling_reve.py correctly.
        try:
            from transformers import AutoModel, AutoConfig  # type: ignore
            config = AutoConfig.from_pretrained(
                str(_REVE_HF_PATH), trust_remote_code=True
            )
            model = AutoModel.from_config(config, trust_remote_code=True)
        except Exception as e:
            return self._stub_or_raise(f"Cannot load REVE architecture: {e}")

        # Try local safetensors first
        if checkpoint_path is not None:
            ckpt = Path(checkpoint_path)
            if ckpt.suffix == ".safetensors" and ckpt.exists():
                return self._load_safetensors(model, ckpt)
            elif ckpt.is_dir():
                sf = ckpt / "model.safetensors"
                if sf.exists():
                    return self._load_safetensors(model, sf)
            else:
                logger.warning("Checkpoint not found at %s", ckpt)

        # Try local checkpoints/reve/ directory
        local_ckpt = (
            Path(__file__).parent.parent.parent.parent
            / "checkpoints" / "reve" / "model.safetensors"
        )
        if local_ckpt.exists():
            return self._load_safetensors(model, local_ckpt)

        # Try HuggingFace — use AutoModel.from_pretrained directly (handles gated repo auth)
        try:
            from transformers import AutoModel as _AutoModel
            hf_model = _AutoModel.from_pretrained(
                "brain-bzh/reve-base",
                trust_remote_code=True,
            )
            # Copy weights into our already-constructed model instance
            model.load_state_dict(hf_model.state_dict(), strict=True)
            logger.info("Loaded REVE weights from HuggingFace (brain-bzh/reve-base).")
            return model
        except Exception as e:
            logger.warning(
                "Could not load REVE weights from HuggingFace (%s). "
                "Using random-init architecture.", e
            )

        logger.info("Using REVE with random (untrained) weights.")
        return model

    def _stub_or_raise(self, reason: str) -> nn.Module:
        if not self.allow_stub_backbone:
            raise RuntimeError(
                f"REVE real backbone is unavailable: {reason}. "
                "Set smoke_test=True only for lightweight wrapper tests."
            )
        logger.warning("%s. Using REVE stub because smoke-test stubs are enabled.", reason)
        return _REVEStub(self.embed_dim)

    @staticmethod
    def _load_safetensors(model: nn.Module, path: Path) -> nn.Module:
        """Load model weights from a .safetensors file."""
        try:
            from safetensors.torch import load_file
            state = load_file(str(path))
        except ImportError:
            # Fallback: torch.load (for .bin or .pt)
            state = torch.load(str(path), map_location="cpu", weights_only=True)

        if isinstance(state, dict) and "model" in state:
            state = state["model"]

        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            logger.warning("REVE checkpoint missing %d keys", len(missing))
        if unexpected:
            logger.warning("REVE checkpoint unexpected %d keys", len(unexpected))
        logger.info("Loaded REVE weights from %s", path)
        return model

    # ------------------------------------------------------------------
    # Freeze policy
    # ------------------------------------------------------------------

    def _apply_freeze_policy(self, policy: str, n_partial_layers: int):
        if isinstance(self.backbone, _REVEStub):
            return

        if policy in ("frozen", "head_only"):
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("REVE backbone frozen.")

        elif policy == "partial":
            for p in self.backbone.parameters():
                p.requires_grad = False
            if hasattr(self.backbone, "transformer"):
                layers = list(self.backbone.transformer.layers)
                for layer in layers[-n_partial_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True
                logger.info("REVE partial: unfreezing last %d layers", n_partial_layers)

        elif policy == "full":
            for p in self.backbone.parameters():
                p.requires_grad = True

        # Head and reference embedding always trainable
        for p in self.classifier_head.parameters():
            p.requires_grad = True
        if self._ref_embedding is not None:
            for p in self._ref_embedding.parameters():
                p.requires_grad = True

    # ------------------------------------------------------------------
    # pos tensor construction
    # ------------------------------------------------------------------

    def _build_pos_tensor(
        self,
        metadata: dict,
        B: int,
        C: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Build the (B, C, 3) position tensor for REVE's FourierEmb4D.

        Maps each spatial_variant to the appropriate coordinate representation.
        """
        variant = self.spatial_variant

        if variant in _ZERO_POS_VARIANTS:
            return torch.zeros(B, C, 3, device=device)

        if variant in _COORDS3D_VARIANTS:
            coords = metadata.get("coords_3d")
            if coords is not None:
                pos = coords.float().to(device)
                if pos.dim() == 2:
                    pos = pos.unsqueeze(0).expand(B, -1, -1)
                return pos.contiguous()
            # Fall back to zeros
            logger.debug("coords_3d not in metadata; using zeros for %s", variant)
            return torch.zeros(B, C, 3, device=device)

        if variant in _COORDS2D_VARIANTS:
            coords = metadata.get("coords_2d")
            if coords is not None:
                pos2d = coords.float().to(device)
                if pos2d.dim() == 2:
                    pos2d = pos2d.unsqueeze(0).expand(B, -1, -1)
                pos3d = torch.cat(
                    [pos2d, torch.zeros(B, C, 1, device=device)], dim=-1
                )
                return pos3d.contiguous()
            return torch.zeros(B, C, 3, device=device)

        if variant in _CHANNEL_ID_VARIANTS:
            # Map channel names to canonical 3D coordinates
            from src.data.coordinate_lookup import get_coords_3d
            channel_names = metadata.get("channel_names", [f"ch{i}" for i in range(C)])
            coords_list = []
            for ch in channel_names:
                xyz = get_coords_3d(ch)
                coords_list.append(xyz if xyz is not None else (0.0, 0.0, 0.0))
            pos = torch.tensor(coords_list, dtype=torch.float32, device=device)  # (C, 3)
            return pos.unsqueeze(0).expand(B, -1, -1).contiguous()

        # Unknown variant — use zeros
        logger.warning("Unknown spatial_variant=%s; using zeros.", variant)
        return torch.zeros(B, C, 3, device=device)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: EEG signal, shape (B, C, T).
            metadata: Dict with optional keys:
                - channel_names: list[str] of length C
                - coords_2d:     Tensor (C, 2) or (B, C, 2)
                - coords_3d:     Tensor (C, 3) or (B, C, 3)
                - reference_meta: list[str] of length C (reference scheme per channel)
                - channel_mask:  BoolTensor (B, C)

        Returns:
            logits: (B, num_classes)
        """
        B, C, T = x.shape
        device = x.device

        pos = self._build_pos_tensor(metadata, B, C, device)

        # --- Run backbone ---
        if isinstance(self.backbone, _REVEStub):
            features = self.backbone(x, pos)
        else:
            features = self._run_real_backbone(x, pos, metadata, B, C, device)

        # --- Optional reference embedding (coords3d_reference only) ---
        if self._ref_embedding is not None and "reference_meta" in metadata:
            ref_idx = self._encode_reference(metadata["reference_meta"], device)
            ref_emb = self._ref_embedding(ref_idx)  # (B, embed_dim) or (B, C, embed_dim)
            if ref_emb.dim() == 3:
                ref_emb = ref_emb.mean(dim=1)  # aggregate over channels → (B, embed_dim)
            features = features + ref_emb

        logits = self.classifier_head(features)
        return logits

    def _run_real_backbone(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        metadata: dict,
        B: int,
        C: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Run real REVE backbone → pooled features (B, embed_dim)."""
        try:
            out = self.backbone(x, pos)  # (B, C, H, embed_dim)
            if out.dim() == 4:
                features = self.backbone.attention_pooling(out)  # (B, embed_dim)
            elif out.dim() == 3:
                features = out.mean(dim=1)  # (B, embed_dim)
            else:
                features = out
            return features
        except Exception as e:
            if not self.allow_stub_backbone:
                raise RuntimeError(
                    "REVE real backbone forward failed. "
                    "Set smoke_test=True only for lightweight wrapper tests."
                ) from e
            logger.warning("REVE forward error: %s. Returning zeros for smoke test.", e)
            return torch.zeros(B, self.embed_dim, device=device)

    def _encode_reference(
        self,
        reference_meta,
        device: torch.device,
    ) -> torch.Tensor:
        """Encode reference type names to indices."""
        vocab = {r: i for i, r in enumerate(self._ref_type_vocab)}
        if isinstance(reference_meta, list):
            if isinstance(reference_meta[0], list):
                # batch of lists
                idx = torch.tensor(
                    [[vocab.get(r, len(self._ref_type_vocab) - 1) for r in row]
                     for row in reference_meta],
                    dtype=torch.long, device=device,
                )  # (B, C)
            else:
                idx = torch.tensor(
                    [vocab.get(r, len(self._ref_type_vocab) - 1) for r in reference_meta],
                    dtype=torch.long, device=device,
                ).unsqueeze(0)  # (1, C)
        else:
            idx = torch.zeros(1, 1, dtype=torch.long, device=device)
        return idx

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        head = sum(p.numel() for p in self.classifier_head.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "classifier_head": head,
        }


class _REVEStub(nn.Module):
    """
    Minimal REVE stub for testing without real weights.

    Accepts the same (x, pos) signature as the real backbone but returns
    random features of the correct shape.
    """

    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.embed_dim = embed_dim
        self._proj = nn.Linear(1, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        # Produce small random features that depend on x (maintains gradient)
        features = x.mean(dim=-1).mean(dim=-1, keepdim=True)  # (B, 1)
        return self._proj(features)  # (B, embed_dim)
