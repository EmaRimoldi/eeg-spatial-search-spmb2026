"""
LaBraM model wrapper.

The benchmark path does *not* require a tiny fixed 10-20 channel subset. It
accepts arbitrary EEG layouts by selecting whichever channels exist in
LaBraM's large positional vocabulary, preserving their benchmark indices, and
scaling the raw amplitudes before patchification. Spatial variants are then
injected into the positional-token pathway.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Sequence
from functools import partial
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
_LABRAM_PATH = _PROJECT_ROOT / "external" / "LaBraM"
_LABRAM_CHECKPOINT = _LABRAM_PATH / "checkpoints" / "labram-base.pth"

# Full benchmark-supported LaBraM vocabulary from EEG-FM-Bench's adapter.
_LABRAM_SUPPORTED_CHANNELS = [
    "FP1", "FPZ", "FP2",
    "AF9", "AF7", "AF5", "AF3", "AF1", "AFZ", "AF2", "AF4", "AF6", "AF8", "AF10",
    "F9", "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8", "F10",
    "FT9", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8", "FT10",
    "T9", "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8", "T10",
    "TP9", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8", "TP10",
    "P9", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8", "P10",
    "PO9", "PO7", "PO5", "PO3", "PO1", "POZ", "PO2", "PO4", "PO6", "PO8", "PO10",
    "O1", "OZ", "O2", "O9", "CB1", "CB2",
    "IZ", "O10", "T3", "T5", "T4", "T6", "M1", "M2", "A1", "A2",
    "CFC1", "CFC2", "CFC3", "CFC4", "CFC5", "CFC6", "CFC7", "CFC8",
    "CCP1", "CCP2", "CCP3", "CCP4", "CCP5", "CCP6", "CCP7", "CCP8",
    "T1", "T2", "FTT9H", "TTP7H", "TPP9H", "FTT10H", "TPP8H", "TPP10H",
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1", "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1", "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
]


def _normalize_channel_name(name: str) -> str:
    return str(name).upper().replace(" ", "").replace(".", "")


class _MLPClassifierHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, hidden_dims: Sequence[int], dropout: float):
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


class LaBraMWrapper(nn.Module):
    """Wrapper around the vendored LaBraM EEG foundation model."""

    def __init__(
        self,
        spatial_variant: str = "coords3d",
        embed_dim: int = 200,
        num_classes: int = 4,
        freeze_policy: Literal["frozen", "head_only", "partial", "full"] = "head_only",
        checkpoint_path: Optional[Union[str, Path]] = None,
        dropout: float = 0.3,
        classifier_hidden_dims: Optional[Sequence[int]] = None,
        eeg_size: int = 800,
        patch_size: int = 200,
        in_chans: int = 1,
        out_chans: int = 8,
        depth: int = 12,
        num_heads: int = 10,
        mlp_ratio: float = 4.0,
        dropout_rate: float = 0.1,
        attn_dropout_rate: float = 0.1,
        drop_path_rate: float = 0.1,
        qkv_bias: bool = False,
        init_values: float = 0.1,
        use_abs_pos_emb: bool = True,
        use_rel_pos_bias: bool = False,
        use_shared_rel_pos_bias: bool = False,
        use_mean_pooling: bool = True,
        init_scale: float = 0.001,
        input_scale: float = 0.01,
        spatial_pos_mode: Literal["residual", "replace"] = "residual",
        spatial_embedding_kwargs: Optional[dict] = None,
        n_partial_layers: int = 2,
        smoke_test: bool = False,
        allow_stub_backbone: bool = False,
    ):
        super().__init__()
        self.spatial_variant = spatial_variant
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)
        self.freeze_policy = freeze_policy
        self.eeg_size = int(eeg_size)
        self.patch_size = int(patch_size)
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.depth = int(depth)
        self.num_heads = int(num_heads)
        self.mlp_ratio = float(mlp_ratio)
        self.dropout_rate = float(dropout_rate)
        self.attn_dropout_rate = float(attn_dropout_rate)
        self.drop_path_rate = float(drop_path_rate)
        self.qkv_bias = bool(qkv_bias)
        self.init_values = float(init_values)
        self.use_abs_pos_emb = bool(use_abs_pos_emb)
        self.use_rel_pos_bias = bool(use_rel_pos_bias)
        self.use_shared_rel_pos_bias = bool(use_shared_rel_pos_bias)
        self.use_mean_pooling = bool(use_mean_pooling)
        self.init_scale = float(init_scale)
        self.input_scale = float(input_scale)
        self.smoke_test = bool(smoke_test)
        self.allow_stub_backbone = bool(allow_stub_backbone or smoke_test)
        self.spatial_pos_mode = spatial_pos_mode
        self.labram_channels = list(_LABRAM_SUPPORTED_CHANNELS)
        self.n_labram_channels = len(self.labram_channels)
        self._labram_lookup = {_normalize_channel_name(ch): i for i, ch in enumerate(self.labram_channels)}

        self.spatial_adapter = SpatialBackboneAdapter(
            variant=spatial_variant,
            embed_dim=self.embed_dim,
            normalize=True,
            pairwise_bias=False,
            spatial_embedding_kwargs=spatial_embedding_kwargs,
        )
        self.spatial_pos_gain: Optional[nn.Parameter] = None
        if spatial_variant != "none":
            self.spatial_pos_gain = nn.Parameter(torch.tensor(0.1))

        self.backbone = self._load_backbone(checkpoint_path)
        self.classifier_head = _MLPClassifierHead(
            embed_dim=self.embed_dim,
            num_classes=self.num_classes,
            hidden_dims=tuple(classifier_hidden_dims or (128,)),
            dropout=dropout,
        )
        self._apply_freeze_policy(freeze_policy, n_partial_layers)

        logger.info(
            "LaBraMWrapper: variant=%s, freeze=%s, classes=%s, backbone=%s, vocab_channels=%s",
            spatial_variant,
            freeze_policy,
            num_classes,
            "stub" if isinstance(self.backbone, _LaBraMStub) else "real",
            self.n_labram_channels,
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_backbone(self, checkpoint_path) -> nn.Module:
        if self.smoke_test:
            logger.warning("LaBraM smoke_test=True; using stub backbone.")
            return _LaBraMStub(self.embed_dim, self.n_labram_channels)

        labram_path_str = str(_LABRAM_PATH)
        if _LABRAM_PATH.exists() and labram_path_str not in sys.path:
            sys.path.insert(0, labram_path_str)

        try:
            from modeling_finetune import NeuralTransformer  # type: ignore

            model = NeuralTransformer(
                EEG_size=self.eeg_size,
                patch_size=self.patch_size,
                in_chans=self.in_chans,
                out_chans=self.out_chans,
                num_classes=0,
                embed_dim=self.embed_dim,
                depth=self.depth,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                qk_norm=partial(nn.LayerNorm, eps=1e-6),
                qk_scale=None,
                drop_rate=self.dropout_rate,
                attn_drop_rate=self.attn_dropout_rate,
                drop_path_rate=self.drop_path_rate,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                init_values=self.init_values,
                use_abs_pos_emb=self.use_abs_pos_emb,
                use_rel_pos_bias=self.use_rel_pos_bias,
                use_shared_rel_pos_bias=self.use_shared_rel_pos_bias,
                use_mean_pooling=self.use_mean_pooling,
                init_scale=self.init_scale,
            )
        except Exception as exc:
            return self._stub_or_raise(f"Could not import or instantiate LaBraM: {exc}")

        ckpt = self._resolve_checkpoint_path(checkpoint_path)
        if ckpt is None:
            logger.warning("LaBraM checkpoint not found; using random-init LaBraM architecture.")
            return model

        try:
            self._load_checkpoint(model, ckpt)
        except Exception as exc:
            logger.warning("Could not load LaBraM checkpoint from %s: %s", ckpt, exc)
        return model

    def _stub_or_raise(self, reason: str) -> nn.Module:
        if not self.allow_stub_backbone:
            raise RuntimeError(
                f"LaBraM real backbone is unavailable: {reason}. "
                "Set smoke_test=True only for lightweight wrapper tests."
            )
        logger.warning("%s. Using LaBraM stub because smoke-test stubs are enabled.", reason)
        return _LaBraMStub(self.embed_dim, self.n_labram_channels)

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path) -> Optional[Path]:
        candidates = []
        if checkpoint_path is not None:
            ckpt = Path(str(checkpoint_path))
            if not ckpt.is_absolute():
                ckpt = _PROJECT_ROOT / ckpt
            candidates.append(ckpt)
        candidates.append(_LABRAM_CHECKPOINT)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                if candidate.stat().st_size < 10_000:
                    logger.warning(
                        "LaBraM checkpoint at %s looks too small (%d bytes); ignoring.",
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
            state = state.get("model", state.get("state_dict", state))

        cleaned = {}
        for key, value in state.items():
            if key.startswith("student."):
                key = key[len("student."):]
            if key in {"head.weight", "head.bias", "mask_token"}:
                continue
            cleaned[key] = value

        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            logger.warning("LaBraM checkpoint missing %d keys", len(missing))
        if unexpected:
            logger.warning("LaBraM checkpoint unexpected %d keys", len(unexpected))
        logger.info("Loaded LaBraM weights from %s", ckpt_path)

    # ------------------------------------------------------------------
    # Freeze policy
    # ------------------------------------------------------------------

    def _apply_freeze_policy(self, policy: str, n_partial_layers: int):
        if not isinstance(self.backbone, _LaBraMStub):
            if policy in ("frozen", "head_only"):
                for p in self.backbone.parameters():
                    p.requires_grad = False
            elif policy == "partial":
                for p in self.backbone.parameters():
                    p.requires_grad = False
                for block in list(self.backbone.blocks)[-n_partial_layers:]:
                    for p in block.parameters():
                        p.requires_grad = True
                if getattr(self.backbone, "fc_norm", None) is not None:
                    for p in self.backbone.fc_norm.parameters():
                        p.requires_grad = True
            elif policy == "full":
                for p in self.backbone.parameters():
                    if p.is_floating_point() or p.is_complex():
                        p.requires_grad = True

        for p in self.classifier_head.parameters():
            p.requires_grad = True
        for p in self.spatial_adapter.parameters():
            p.requires_grad = True
        if self.spatial_pos_gain is not None:
            self.spatial_pos_gain.requires_grad = True

    # ------------------------------------------------------------------
    # Input shaping + spatial injection
    # ------------------------------------------------------------------

    def _select_supported_inputs(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[str]]:
        b, c, _ = x.shape
        input_names = metadata.get("channel_names")
        if not isinstance(input_names, Sequence) or len(input_names) != c:
            input_names = [f"ch{i}" for i in range(c)]

        input_lookup = {_normalize_channel_name(name): i for i, name in enumerate(input_names)}

        supported_indices: list[int] = []
        input_indices: list[int] = []
        selected_names: list[str] = []
        for vocab_idx, vocab_name in enumerate(self.labram_channels):
            input_idx = input_lookup.get(_normalize_channel_name(vocab_name))
            if input_idx is None:
                continue
            supported_indices.append(vocab_idx)
            input_indices.append(input_idx)
            selected_names.append(vocab_name)

        if not input_indices:
            fallback_n = min(c, self.n_labram_channels)
            supported_indices = list(range(fallback_n))
            input_indices = list(range(fallback_n))
            selected_names = [str(input_names[i]) for i in input_indices]

        channel_mask = normalize_channel_mask(metadata.get("channel_mask"), b, c, x.device)
        selected_mask = channel_mask[:, input_indices]
        x_selected = x[:, input_indices, :].float() * self.input_scale
        x_selected = x_selected * selected_mask.to(x_selected.dtype).unsqueeze(-1)
        return x_selected.contiguous(), selected_mask, supported_indices, input_indices, selected_names

    def _prepare_patches(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int], list[int], list[str]]:
        x, channel_mask, supported_indices, input_indices, selected_names = self._select_supported_inputs(x, metadata)
        b, n, t = x.shape

        n_patches = max(1, t // self.patch_size)
        target_len = n_patches * self.patch_size
        if target_len != t:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)

        patches = x.reshape(b, n, n_patches, self.patch_size).contiguous()
        return patches, channel_mask, supported_indices, input_indices, selected_names

    @staticmethod
    def _select_coords(
        coords: Optional[torch.Tensor],
        input_indices: list[int],
        batch_size: int,
        last_dim: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if coords is None:
            return None
        coords = coords.to(device).float()
        if coords.dim() == 2:
            coords = coords.unsqueeze(0).expand(batch_size, -1, -1)
        if coords.dim() != 3 or coords.shape[-1] != last_dim:
            raise ValueError(
                f"Expected coords with shape (B, C, {last_dim}) or (C, {last_dim}), got {tuple(coords.shape)}"
            )
        return coords[:, input_indices, :].contiguous()

    @staticmethod
    def _select_reference_meta(reference_meta, input_indices: list[int]):
        if not isinstance(reference_meta, list) or not reference_meta:
            return None
        if isinstance(reference_meta[0], list):
            reference_meta = reference_meta[0]
        if len(reference_meta) < max(input_indices, default=-1) + 1:
            return None
        return [reference_meta[i] for i in input_indices]

    def _build_labram_spatial_state(
        self,
        metadata: dict,
        input_indices: list[int],
        selected_names: list[str],
        batch_size: int,
        device: torch.device,
        channel_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        coords_2d = self._select_coords(metadata.get("coords_2d"), input_indices, batch_size, 2, device)
        coords_3d = self._select_coords(metadata.get("coords_3d"), input_indices, batch_size, 3, device)
        reference_meta = self._select_reference_meta(metadata.get("reference_meta"), input_indices)

        spatial_emb, _ = self.spatial_adapter.build_embedding(
            metadata=metadata,
            batch_size=batch_size,
            n_channels=len(selected_names),
            device=device,
            channel_names=selected_names,
            coords_2d=coords_2d,
            coords_3d=coords_3d,
            reference_meta=reference_meta,
        )
        return self.spatial_adapter.prepare(spatial_emb, channel_mask)

    @staticmethod
    def _input_chans(supported_indices: list[int], device: torch.device) -> torch.Tensor:
        chans = [0] + [idx + 1 for idx in supported_indices]
        return torch.tensor(chans, device=device, dtype=torch.long)

    def _run_real_backbone(
        self,
        patches: torch.Tensor,
        spatial_state: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
        input_chans: torch.Tensor,
    ) -> torch.Tensor:
        b, n, a, _ = patches.shape
        x = self.backbone.patch_embed(patches)

        cls_tokens = self.backbone.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        native_pos = self.backbone.pos_embed[:, input_chans]
        cls_pos = native_pos[:, 0:1, :].expand(b, -1, -1)
        native_patch_pos = native_pos[:, 1:, :].unsqueeze(2).expand(b, -1, a, -1).flatten(1, 2)

        patch_pos = native_patch_pos
        if spatial_state is not None and self.spatial_pos_gain is not None:
            spatial_tokens = spatial_state.unsqueeze(2).expand(b, -1, a, -1).flatten(1, 2)
            gain = self.spatial_pos_gain.to(dtype=x.dtype)
            if self.spatial_pos_mode == "replace":
                patch_pos = gain * spatial_tokens + (1.0 - gain.clamp(0.0, 1.0)) * native_patch_pos
            else:
                patch_pos = native_patch_pos + gain * spatial_tokens

        x = x + torch.cat((cls_pos, patch_pos), dim=1)

        if getattr(self.backbone, "time_embed", None) is not None:
            time_embed = self._time_embedding(a, x.device, x.dtype)
            time_embed = time_embed.unsqueeze(0).unsqueeze(1).expand(b, n, -1, -1).flatten(1, 2)
            x[:, 1:, :] = x[:, 1:, :] + time_embed

        token_mask = channel_mask.to(x.dtype).unsqueeze(-1).expand(-1, -1, a).flatten(1)
        x[:, 1:, :] = x[:, 1:, :] * token_mask.unsqueeze(-1)

        x = self.backbone.pos_drop(x)
        for block in self.backbone.blocks:
            x = block(x, rel_pos_bias=None)

        x = self.backbone.norm(x)
        patch_tokens = x[:, 1:, :]
        if getattr(self.backbone, "fc_norm", None) is not None:
            patch_tokens = self.backbone.fc_norm(patch_tokens)
        return patch_tokens.view(b, n, a, self.embed_dim)

    def _time_embedding(
        self,
        time_steps: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        time_embed = self.backbone.time_embed
        if time_embed.shape[1] == time_steps:
            return time_embed[0].to(device=device, dtype=dtype)

        interp = F.interpolate(
            time_embed.transpose(1, 2),
            size=time_steps,
            mode="linear",
            align_corners=False,
        )
        return interp.transpose(1, 2)[0].to(device=device, dtype=dtype)

    @staticmethod
    def _pool_tokens(tokens: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        b, n, a, d = tokens.shape
        mask = channel_mask.to(tokens.dtype).unsqueeze(-1).unsqueeze(-1)
        summed = (tokens * mask).sum(dim=(1, 2))
        denom = channel_mask.to(tokens.dtype).sum(dim=1).clamp_min(1.0) * float(a)
        return summed / denom.unsqueeze(-1)

    def forward(self, x: torch.Tensor, metadata: dict) -> torch.Tensor:
        b = x.shape[0]
        patches, channel_mask, supported_indices, input_indices, selected_names = self._prepare_patches(x, metadata)
        spatial_state = self._build_labram_spatial_state(
            metadata=metadata,
            input_indices=input_indices,
            selected_names=selected_names,
            batch_size=b,
            device=x.device,
            channel_mask=channel_mask,
        )

        if isinstance(self.backbone, _LaBraMStub):
            tokens = self.backbone(patches, spatial_state=spatial_state, channel_mask=channel_mask)
        else:
            input_chans = self._input_chans(supported_indices, x.device)
            tokens = self._run_real_backbone(
                patches=patches,
                spatial_state=spatial_state,
                channel_mask=channel_mask,
                input_chans=input_chans,
            )

        features = self._pool_tokens(tokens, channel_mask)
        return self.classifier_head(features)

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        head = sum(p.numel() for p in self.classifier_head.parameters())
        spatial = sum(p.numel() for p in self.spatial_adapter.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "classifier_head": head,
            "spatial_adapter": spatial,
        }


class _LaBraMStub(nn.Module):
    """Lightweight LaBraM-shaped backbone for explicit smoke tests."""

    def __init__(self, embed_dim: int, n_channels: int):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.n_channels = int(n_channels)
        self.proj = nn.Linear(1, embed_dim)

    def forward(
        self,
        patches: torch.Tensor,
        spatial_state: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.proj(patches.mean(dim=-1, keepdim=True))
        if spatial_state is not None:
            feats = feats + 0.1 * spatial_state.unsqueeze(2)
        if channel_mask is not None:
            feats = feats * channel_mask.to(feats.dtype).unsqueeze(-1).unsqueeze(-1)
        return feats
