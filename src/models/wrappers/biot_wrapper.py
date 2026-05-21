"""
BIOT model wrapper.

BIOT builds channel-wise spectral token sequences, adds learned channel tokens,
then runs a linear-attention transformer. For the EEG-FM-Bench-style path we
first normalize each input channel, route the dataset's raw electrodes into the
BIOT checkpoint's fixed channel space, and only then inject spatial variants at
the channel-token stage.

Key fidelity notes versus the benchmark replication path:
- use per-channel percentile normalization before the encoder
- use a 1x1 channel router into the pretrained BIOT channel count (18 by default)
- keep a montage-prior fallback/init so routing remains usable outside the
  benchmark configs and still exposes meaningful spatial coordinates
- use the benchmark-style MLP classifier head instead of a single linear layer
"""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Optional, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.wrappers.spatial_adapter import (
    SpatialBackboneAdapter,
    normalize_channel_mask,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_BIOT_MODEL = _PROJECT_ROOT / "external" / "BIOT" / "model" / "biot.py"
_BIOT_DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT / "external" / "BIOT" / "pretrained-models" / "EEG-six-datasets-18-channels.ckpt"
)

# Fixed BIOT checkpoint channel order from the vendored README.
_BIOT_TARGET_CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "C3-A2", "C4-A1",
]


def _normalize_channel_name(name: str) -> str:
    return str(name).upper().replace(" ", "").replace(".", "")


def _ensure_batch_coords(
    coords: Optional[torch.Tensor],
    batch_size: int,
    expected_last_dim: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if coords is None:
        return None
    coords = coords.to(device).float()
    if coords.dim() == 2:
        coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
    if coords.dim() != 3 or coords.shape[-1] != expected_last_dim:
        raise ValueError(
            f"Expected coords with shape (B, C, {expected_last_dim}) or (C, {expected_last_dim}), "
            f"got {tuple(coords.shape)}"
        )
    return coords


class _Conv1dWithConstraint(nn.Conv1d):
    """Minimal local copy of EEG-FM-Bench's constrained 1x1 channel router."""

    def __init__(self, *args, max_norm: float = 1.0, do_weight_norm: bool = True, **kwargs):
        self.max_norm = float(max_norm)
        self.do_weight_norm = bool(do_weight_norm)
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.do_weight_norm:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class _MLPClassifierHead(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (),
        dropout: float = 0.3,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = int(embed_dim)
        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(float(dropout)),
            ])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BIOTWrapper(nn.Module):
    """Wrapper around the vendored BIOT encoder."""

    def __init__(
        self,
        spatial_variant: str = "coords3d",
        embed_dim: int = 256,
        num_classes: int = 4,
        freeze_policy: Literal["frozen", "head_only", "partial", "full"] = "head_only",
        checkpoint_path: Optional[Union[str, Path]] = None,
        dropout: float = 0.3,
        heads: int = 8,
        depth: int = 4,
        max_channels: int = 18,
        input_channels: Optional[int] = None,
        use_channel_conv: bool = True,
        router_init_mode: Literal["montage_prior", "random"] = "montage_prior",
        input_norm_percentile: float = 0.95,
        classifier_hidden_dims: Optional[Sequence[int]] = None,
        n_fft: int = 200,
        hop_length: int = 100,
        spatial_init_gain: float = 0.1,
        spatial_embedding_kwargs: Optional[dict] = None,
        n_partial_layers: int = 1,
        smoke_test: bool = False,
        allow_stub_backbone: bool = False,
    ):
        super().__init__()
        self.spatial_variant = spatial_variant
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.freeze_policy = freeze_policy
        self.heads = int(heads)
        self.depth = int(depth)
        self.max_channels = int(max_channels)
        self.input_channels = None if input_channels is None else int(input_channels)
        self.use_channel_conv = bool(use_channel_conv)
        self.router_init_mode = router_init_mode
        self.input_norm_percentile = float(input_norm_percentile)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.smoke_test = bool(smoke_test)
        self.allow_stub_backbone = bool(allow_stub_backbone or smoke_test)
        self.target_channel_names = self._target_channel_names(self.max_channels)
        self._router_initialized_from_metadata = False

        self.spatial_adapter = SpatialBackboneAdapter(
            variant=spatial_variant,
            embed_dim=self.embed_dim,
            normalize=True,
            pairwise_bias=False,
            spatial_embedding_kwargs=spatial_embedding_kwargs,
        )
        self.spatial_token_gain: Optional[nn.Parameter] = None
        if spatial_variant != "none":
            self.spatial_token_gain = nn.Parameter(torch.tensor(float(spatial_init_gain)))

        self.channel_router: Optional[_Conv1dWithConstraint] = None
        if self.input_channels is not None and self.use_channel_conv:
            self.channel_router = _Conv1dWithConstraint(
                self.input_channels,
                self.max_channels,
                kernel_size=1,
                max_norm=1.0,
            )
            self._initialize_router_default()

        self.backbone = self._load_backbone(checkpoint_path)
        self.classifier_head = _MLPClassifierHead(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            hidden_dims=tuple(classifier_hidden_dims or (128,)),
            dropout=dropout,
        )
        self._apply_freeze_policy(freeze_policy, n_partial_layers)

        logger.info(
            "BIOTWrapper: variant=%s, freeze=%s, classes=%s, backbone=%s, max_channels=%s, input_channels=%s, channel_router=%s",
            spatial_variant,
            freeze_policy,
            num_classes,
            "stub" if isinstance(self.backbone, _BIOTStub) else "real",
            self.max_channels,
            self.input_channels,
            self.channel_router is not None,
        )

    @staticmethod
    def _target_channel_names(max_channels: int) -> list[str]:
        names = list(_BIOT_TARGET_CHANNELS[:max_channels])
        if len(names) < max_channels:
            names.extend([f"biot_latent_{i}" for i in range(len(names), max_channels)])
        return names

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_backbone(self, checkpoint_path: Optional[Union[str, Path]]) -> nn.Module:
        if self.smoke_test:
            logger.warning("BIOT smoke_test=True; using stub backbone.")
            return _BIOTStub(self.embed_dim, self.max_channels)

        encoder_cls = self._import_biot_encoder()
        if encoder_cls is None:
            return self._stub_or_raise("BIOT reference implementation import failed")

        try:
            model = encoder_cls(
                emb_size=self.embed_dim,
                heads=self.heads,
                depth=self.depth,
                n_channels=self.max_channels,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
            )
        except Exception as exc:
            return self._stub_or_raise(f"Could not instantiate BIOT encoder: {exc}")

        ckpt = self._resolve_checkpoint_path(checkpoint_path)
        if ckpt is None:
            logger.warning("BIOT checkpoint not found; using random-init BIOT architecture.")
            return model

        try:
            self._load_checkpoint(model, ckpt)
        except Exception as exc:
            logger.warning("Could not load BIOT checkpoint from %s: %s", ckpt, exc)
        return model

    @staticmethod
    def _import_biot_encoder():
        if not _BIOT_MODEL.exists():
            logger.warning("BIOT model file not found at %s", _BIOT_MODEL)
            return None
        try:
            spec = importlib.util.spec_from_file_location("external_biot_model", str(_BIOT_MODEL))
            if spec is None or spec.loader is None:
                raise ImportError("Could not create import spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "BIOTEncoder")
        except Exception as exc:
            logger.warning("Failed to import BIOT model: %s", exc)
            return None

    def _stub_or_raise(self, reason: str) -> nn.Module:
        if not self.allow_stub_backbone:
            raise RuntimeError(
                f"BIOT real backbone is unavailable: {reason}. "
                "Set smoke_test=True only for lightweight wrapper tests."
            )
        logger.warning("%s. Using BIOT stub because smoke-test stubs are enabled.", reason)
        return _BIOTStub(self.embed_dim, self.max_channels)

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path: Optional[Union[str, Path]]) -> Optional[Path]:
        candidates: list[Path] = []
        if checkpoint_path is not None:
            ckpt = Path(str(checkpoint_path))
            if not ckpt.is_absolute():
                ckpt = _PROJECT_ROOT / ckpt
            candidates.append(ckpt)
        candidates.append(_BIOT_DEFAULT_CHECKPOINT)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                if candidate.stat().st_size < 10_000:
                    logger.warning(
                        "BIOT checkpoint at %s looks too small (%d bytes); ignoring.",
                        candidate,
                        candidate.stat().st_size,
                    )
                    continue
                return candidate
        return None

    @staticmethod
    def _load_checkpoint(model: nn.Module, ckpt_path: Path) -> None:
        state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        if isinstance(state, dict):
            state = state.get("state_dict", state.get("model", state))

        model_state = model.state_dict()
        filtered = {}
        with torch.no_grad():
            for key, value in state.items():
                if key == "channel_tokens.weight" and key in model_state:
                    rows = min(model_state[key].shape[0], value.shape[0])
                    model.channel_tokens.weight[:rows].copy_(value[:rows])
                    continue
                if key == "index":
                    continue
                if key in model_state and model_state[key].shape == value.shape:
                    filtered[key] = value

        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if missing:
            logger.warning("BIOT checkpoint missing %d keys", len(missing))
        if unexpected:
            logger.warning("BIOT checkpoint unexpected %d keys", len(unexpected))
        logger.info("Loaded BIOT weights from %s", ckpt_path)

    # ------------------------------------------------------------------
    # Freeze policy
    # ------------------------------------------------------------------

    def _apply_freeze_policy(self, policy: str, n_partial_layers: int):
        if not isinstance(self.backbone, _BIOTStub):
            if policy in ("frozen", "head_only"):
                for p in self.backbone.parameters():
                    p.requires_grad = False
            elif policy == "partial":
                for p in self.backbone.parameters():
                    p.requires_grad = False
                layers = getattr(getattr(self.backbone.transformer, "layers", None), "layers", None)
                if layers is not None:
                    for layer in list(layers)[-n_partial_layers:]:
                        for p in layer.parameters():
                            p.requires_grad = True
            elif policy == "full":
                for p in self.backbone.parameters():
                    if p.is_floating_point() or p.is_complex():
                        p.requires_grad = True

        if self.channel_router is not None:
            for p in self.channel_router.parameters():
                p.requires_grad = True
        for p in self.classifier_head.parameters():
            p.requires_grad = True
        for p in self.spatial_adapter.parameters():
            p.requires_grad = True
        if self.spatial_token_gain is not None:
            self.spatial_token_gain.requires_grad = True

    # ------------------------------------------------------------------
    # Input routing + spatial state
    # ------------------------------------------------------------------

    def _initialize_router_default(self) -> None:
        if self.channel_router is None:
            return
        with torch.no_grad():
            nn.init.xavier_uniform_(self.channel_router.weight)
            if self.channel_router.bias is not None:
                self.channel_router.bias.zero_()
            if self.channel_router.in_channels == self.channel_router.out_channels:
                self.channel_router.weight.zero_()
                eye = torch.eye(self.channel_router.out_channels, device=self.channel_router.weight.device)
                self.channel_router.weight[:, :, 0].copy_(eye)
        self._router_initialized_from_metadata = False

    def _projection_prior_from_metadata(
        self,
        channel_names: Sequence[str],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        c_in = len(channel_names)
        prior = torch.zeros(self.max_channels, c_in, device=device, dtype=dtype)
        input_lookup = {_normalize_channel_name(name): i for i, name in enumerate(channel_names)}

        for out_idx, target_name in enumerate(self.target_channel_names):
            row = prior[out_idx]
            if "-" in target_name:
                left, right = target_name.split("-", 1)
                left_idx = input_lookup.get(_normalize_channel_name(left))
                right_idx = input_lookup.get(_normalize_channel_name(right))
                if left_idx is not None and right_idx is not None:
                    row[left_idx] = 1.0
                    row[right_idx] = -1.0
                elif left_idx is not None:
                    row[left_idx] = 1.0
                elif right_idx is not None:
                    row[right_idx] = 1.0
            else:
                target_idx = input_lookup.get(_normalize_channel_name(target_name))
                if target_idx is not None:
                    row[target_idx] = 1.0

            if torch.count_nonzero(row) == 0 and c_in > 0:
                row[out_idx % c_in] = 1.0

            norm = row.norm(p=2)
            if norm > 1.0:
                row.div_(norm)

        return prior

    def _maybe_initialize_router_from_metadata(self, metadata: dict) -> None:
        if self.channel_router is None or self._router_initialized_from_metadata:
            return
        if self.router_init_mode != "montage_prior":
            self._router_initialized_from_metadata = True
            return

        channel_names = metadata.get("channel_names")
        if not isinstance(channel_names, Sequence) or len(channel_names) != self.channel_router.in_channels:
            return

        prior = self._projection_prior_from_metadata(
            channel_names,
            device=self.channel_router.weight.device,
            dtype=self.channel_router.weight.dtype,
        )
        with torch.no_grad():
            self.channel_router.weight.zero_()
            self.channel_router.weight[:, :, 0].copy_(prior)
            if self.channel_router.bias is not None:
                self.channel_router.bias.zero_()
        self._router_initialized_from_metadata = True

    def _effective_projection_matrix(
        self,
        metadata: dict,
        n_input_channels: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.channel_router is not None:
            if n_input_channels != self.channel_router.in_channels:
                raise ValueError(
                    f"BIOTWrapper expected {self.channel_router.in_channels} input channels, got {n_input_channels}. "
                    "Pass wrapper_kwargs.input_channels that matches the dataset channel count."
                )
            self._maybe_initialize_router_from_metadata(metadata)
            return self.channel_router.weight[:, :, 0].to(device=device, dtype=dtype)

        channel_names = metadata.get("channel_names")
        if isinstance(channel_names, Sequence) and len(channel_names) == n_input_channels:
            return self._projection_prior_from_metadata(channel_names, device=device, dtype=dtype)

        projection = torch.zeros(self.max_channels, n_input_channels, device=device, dtype=dtype)
        if n_input_channels == 0:
            return projection
        for out_idx in range(self.max_channels):
            projection[out_idx, out_idx % n_input_channels] = 1.0
        return projection

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.quantile(x.abs(), self.input_norm_percentile, dim=-1, keepdim=True)
        scale = scale.clamp_min(1.0e-6)
        return x / scale

    def _trim_time(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] < self.n_fft:
            return F.interpolate(x, size=self.n_fft, mode="linear", align_corners=False)

        n_patches = max(1, x.shape[-1] // self.n_fft)
        target_len = n_patches * self.n_fft
        if target_len == x.shape[-1]:
            return x
        return x[:, :, :target_len]

    def _project_input(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, _ = x.shape
        x = x.float()
        input_mask = normalize_channel_mask(metadata.get("channel_mask"), b, c, x.device)
        x = x * input_mask.to(x.dtype).unsqueeze(-1)
        x = self._normalize_input(x)
        x = self._trim_time(x)

        projection = self._effective_projection_matrix(metadata, c, x.device, x.dtype)
        if self.channel_router is not None:
            mapped_x = self.channel_router(x)
        else:
            mapped_x = torch.einsum("oc,bct->bot", projection, x)

        mapped_mask = torch.ones(b, self.max_channels, dtype=torch.bool, device=x.device)
        return mapped_x.contiguous(), input_mask, mapped_mask, projection

    def _project_coords(
        self,
        coords: Optional[torch.Tensor],
        input_channel_mask: torch.Tensor,
        projection: torch.Tensor,
        *,
        batch_size: int,
        last_dim: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        coords = _ensure_batch_coords(coords, batch_size, last_dim, device)
        if coords is None:
            return None

        weights = projection.to(device=device, dtype=coords.dtype).abs()
        effective = weights.unsqueeze(0) * input_channel_mask.to(coords.dtype).unsqueeze(1)
        denom = effective.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        effective = effective / denom
        return torch.einsum("boc,bcd->bod", effective, coords)

    def _build_spatial_state(
        self,
        metadata: dict,
        batch_size: int,
        device: torch.device,
        input_channel_mask: torch.Tensor,
        projection: torch.Tensor,
        mapped_channel_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        coords_2d = self._project_coords(
            metadata.get("coords_2d"),
            input_channel_mask,
            projection,
            batch_size=batch_size,
            last_dim=2,
            device=device,
        )
        coords_3d = self._project_coords(
            metadata.get("coords_3d"),
            input_channel_mask,
            projection,
            batch_size=batch_size,
            last_dim=3,
            device=device,
        )
        reference_meta = ["average"] * self.max_channels

        spatial_emb, _ = self.spatial_adapter.build_embedding(
            metadata=metadata,
            batch_size=batch_size,
            n_channels=self.max_channels,
            device=device,
            channel_names=self.target_channel_names,
            coords_2d=coords_2d,
            coords_3d=coords_3d,
            reference_meta=reference_meta,
        )
        return self.spatial_adapter.prepare(spatial_emb, mapped_channel_mask)

    # ------------------------------------------------------------------
    # Spatial BIOT execution
    # ------------------------------------------------------------------

    def _run_real_backbone(
        self,
        x: torch.Tensor,
        spatial_state: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        emb_seq = []
        for i in range(x.shape[1]):
            channel_spec_emb = self.backbone.stft(x[:, i : i + 1, :])
            channel_spec_emb = self.backbone.patch_embedding(channel_spec_emb)
            batch_size, steps, _ = channel_spec_emb.shape

            channel_token_emb = (
                self.backbone.channel_tokens(self.backbone.index[i])
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(batch_size, steps, -1)
            )
            channel_emb = channel_spec_emb + channel_token_emb
            if spatial_state is not None and self.spatial_token_gain is not None:
                residual = spatial_state[:, i : i + 1, :].expand(-1, steps, -1)
                channel_emb = channel_emb + self.spatial_token_gain.to(channel_emb.dtype) * residual

            channel_emb = self.backbone.positional_encoding(channel_emb)
            valid = channel_mask[:, i].to(channel_emb.dtype).view(batch_size, 1, 1)
            emb_seq.append(channel_emb * valid)

        emb = torch.cat(emb_seq, dim=1)
        return self.backbone.transformer(emb).mean(dim=1)

    def forward(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        b = x.shape[0]
        mapped_x, input_mask, mapped_mask, projection = self._project_input(x, metadata)
        spatial_state = self._build_spatial_state(
            metadata=metadata,
            batch_size=b,
            device=mapped_x.device,
            input_channel_mask=input_mask,
            projection=projection,
            mapped_channel_mask=mapped_mask,
        )

        if isinstance(self.backbone, _BIOTStub):
            features = self.backbone(mapped_x, spatial_state=spatial_state, channel_mask=mapped_mask)
        else:
            features = self._run_real_backbone(mapped_x, spatial_state, mapped_mask)
        return self.classifier_head(features)

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        head = sum(p.numel() for p in self.classifier_head.parameters())
        spatial = sum(p.numel() for p in self.spatial_adapter.parameters())
        router = sum(p.numel() for p in self.channel_router.parameters()) if self.channel_router is not None else 0
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "classifier_head": head,
            "spatial_adapter": spatial,
            "channel_router": router,
        }


class _BIOTStub(nn.Module):
    """Lightweight BIOT-shaped encoder for explicit smoke tests."""

    def __init__(self, embed_dim: int, max_channels: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.max_channels = int(max_channels)
        self.proj = nn.Linear(1, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        spatial_state: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        channel_feats = self.proj(x.mean(dim=-1, keepdim=True))
        if spatial_state is not None:
            channel_feats = channel_feats + 0.1 * spatial_state
        if channel_mask is not None:
            channel_feats = channel_feats * channel_mask.to(channel_feats.dtype).unsqueeze(-1)
            denom = channel_mask.to(channel_feats.dtype).sum(dim=1).clamp_min(1.0)
        else:
            denom = torch.tensor(float(x.shape[1]), device=x.device, dtype=channel_feats.dtype)
        return channel_feats.sum(dim=1) / denom.unsqueeze(-1)
