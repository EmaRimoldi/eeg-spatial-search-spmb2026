"""
CBraMod model wrapper.

This wrapper integrates the vendored EEG-FM-Bench CBraMod baseline into the
project's backbone registry and spatial-ablation interface.

Primary reference:
- external/EEG-FM-Bench/baseline/cbramod/{cbramod_adapter.py, model.py}

Integration strategy:
- Reuse EEG-FM-Bench's CBraMod backbone definition instead of re-implementing it.
- Reuse its signal scaling convention (x * 0.01) and 200-sample patching.
- Accept arbitrary channel layouts from our metadata pipeline.
- Build per-channel spatial states from our spatial variants and inject them
  inside CBraMod's spatial self-attention path as an additive attention-logit
  bias for every encoder layer.
- Preserve a fair `none` path that bypasses spatial conditioning entirely.
- Keep the lightweight post-backbone graph layers available as an optional
  ablation, but make the default spatial path native to the backbone block stack.

Pretrained checkpoint handling:
- If a local checkpoint is available, load it.
- If no checkpoint is available, keep the real architecture with random-init
  weights and log a warning.
- If the reference implementation cannot be imported at all, fall back to a
  small stub so the training pipeline still works for smoke tests.
"""

import importlib.util
import logging
from pathlib import Path
from typing import Optional, Union, Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.wrappers.spatial_adapter import (
    SpatialBackboneAdapter,
    channel_names_from_metadata,
    normalize_channel_mask,
    reference_meta_from_metadata,
)

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
_EEG_FM_BENCH_CBRAMOD_MODEL = (
    _PROJECT_ROOT / "external" / "EEG-FM-Bench" / "baseline" / "cbramod" / "model.py"
)
_CBRAMOD_DEFAULT_CHECKPOINT = (
    _PROJECT_ROOT / "external" / "CBraMod" / "pretrained_weights" / "pretrained_weights.pth"
)

# The vendored implementation is architecturally tied to 200-sample patches
# and a 200-dim patch embedding (25 conv channels × 8 temporal positions).
_CBRAMOD_PATCH_SIZE = 200
_CBRAMOD_BACKBONE_DIM = 200
_CBRAMOD_DIM_FFN = 800
_CBRAMOD_DEPTH = 12
_CBRAMOD_NUM_HEADS = 8


class CBraModWrapper(nn.Module):
    """
    Wrapper around the vendored EEG-FM-Bench CBraMod backbone.

    Args:
        spatial_variant: Spatial embedding variant name.
        embed_dim: Output feature dim after CBraMod's proj_out.
        num_classes: Number of downstream classes.
        freeze_policy: Backbone freeze policy.
        checkpoint_path: Optional local checkpoint path.
        dropout: Dropout rate in the classification head.
        input_scale: Scalar applied before patching (matches EEG-FM-Bench adapter).
        n_partial_layers: Number of encoder layers to unfreeze for partial tuning.
        dist_bias_scale: Initial scale used for coords3d_distbias attention bias.
        spatial_init_gain: Initial scale for native spatial attention bias.
        graph_depth: Number of lightweight post-backbone channel-graph layers.
        graph_k_neighbors: Optional number of nearest spatial neighbors used by
            graph layers. None keeps the dense RBF graph.
        graph_sigma_scale: Multiplier on the median-distance graph RBF scale.
        native_spatial_mode: How to inject spatial context inside CBraMod.
        native_pair_dim: Hidden size used to map channel spatial states into
            pairwise attention biases.
        spatial_embedding_kwargs: Optional kwargs for the selected spatial
            embedding, such as RBF basis count or anchor mode.
        classifier_head: Optional head config. Supported head_type values:
            avg_pool, flatten_linear, flatten_mlp.
        smoke_test: If true, use the lightweight stub instead of the real
            backbone. This is intended for unit tests only.
    """

    def __init__(
        self,
        spatial_variant: str = "coords3d",
        embed_dim: int = 200,
        num_classes: int = 4,
        freeze_policy: Literal["frozen", "head_only", "partial", "full"] = "head_only",
        checkpoint_path: Optional[Union[str, Path]] = None,
        dropout: float = 0.1,
        input_scale: float = 0.01,
        n_partial_layers: int = 2,
        dist_bias_scale: float = 0.05,
        spatial_init_gain: float = 0.15,
        graph_depth: int = 0,
        graph_k_neighbors: Optional[int] = None,
        graph_sigma_scale: float = 1.0,
        native_spatial_mode: Literal["none", "spatial_attn_bias"] = "spatial_attn_bias",
        native_pair_dim: int = 64,
        spatial_embedding_kwargs: Optional[dict[str, Any]] = None,
        classifier_head: Optional[dict[str, Any]] = None,
        n_channels: int = 22,
        eeg_size: int = 1000,
        smoke_test: bool = False,
        allow_stub_backbone: bool = False,
    ):
        super().__init__()
        self.spatial_variant = spatial_variant
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.freeze_policy = freeze_policy
        self.input_scale = input_scale
        self.patch_size = _CBRAMOD_PATCH_SIZE
        self.backbone_dim = _CBRAMOD_BACKBONE_DIM
        self.graph_depth = int(graph_depth)
        self.graph_k_neighbors = None if graph_k_neighbors is None else int(graph_k_neighbors)
        self.graph_sigma_scale = float(graph_sigma_scale)
        self.native_spatial_mode = native_spatial_mode
        self.native_pair_dim = int(native_pair_dim)
        self.n_channels = int(n_channels)
        self.eeg_size = int(eeg_size)
        self.smoke_test = bool(smoke_test)
        self.allow_stub_backbone = bool(allow_stub_backbone or smoke_test)
        self.use_native_spatial = (
            spatial_variant != "none" and native_spatial_mode == "spatial_attn_bias"
        )

        self.spatial_adapter = SpatialBackboneAdapter(
            variant=spatial_variant,
            embed_dim=self.backbone_dim,
            normalize=True,
            pairwise_bias=self.use_native_spatial,
            pair_dim=native_pair_dim,
            dist_bias_scale=dist_bias_scale,
            spatial_embedding_kwargs=spatial_embedding_kwargs,
        )
        self.native_attn_gains = nn.ParameterList()
        if self.use_native_spatial:
            for _ in range(_CBRAMOD_DEPTH):
                self.native_attn_gains.append(
                    nn.Parameter(torch.tensor(float(spatial_init_gain)))
                )

        self.graph_layers = nn.ModuleList()
        self.graph_gains = nn.ParameterList()
        for _ in range(self.graph_depth):
            self.graph_layers.append(
                nn.Sequential(
                    nn.Linear(self.embed_dim, self.embed_dim),
                    nn.GELU(),
                    nn.Linear(self.embed_dim, self.embed_dim),
                )
            )
            self.graph_gains.append(nn.Parameter(torch.tensor(0.0)))

        self.backbone = self._load_backbone(checkpoint_path)

        self.classifier_head = self._build_classifier_head(
            classifier_head=classifier_head,
            dropout=dropout,
        )

        self._apply_freeze_policy(freeze_policy, n_partial_layers)

        logger.info(
            "CBraModWrapper: variant=%s, freeze=%s, classes=%s, backbone=%s",
            spatial_variant,
            freeze_policy,
            num_classes,
            "stub" if isinstance(self.backbone, _CBraModStub) else "real",
        )

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_backbone(
        self,
        checkpoint_path: Optional[Union[str, Path]],
    ) -> nn.Module:
        if self.smoke_test:
            logger.warning("CBraMod smoke_test=True; using stub backbone.")
            return _CBraModStub(out_dim=self.embed_dim)

        cbramod_cls = self._import_bench_cbramod()
        if cbramod_cls is None:
            return self._stub_or_raise("EEG-FM-Bench CBraMod import failed")

        try:
            model = cbramod_cls(
                in_dim=self.patch_size,
                out_dim=self.embed_dim,
                d_model=self.backbone_dim,
                dim_ffn=_CBRAMOD_DIM_FFN,
                n_layer=_CBRAMOD_DEPTH,
                n_head=_CBRAMOD_NUM_HEADS,
            )
        except Exception as e:
            return self._stub_or_raise(f"Could not instantiate CBraMod architecture: {e}")

        ckpt_path = self._resolve_checkpoint_path(checkpoint_path)
        if ckpt_path is None:
            logger.warning(
                "CBraMod pretrained checkpoint not found. "
                "Using random-init CBraMod architecture. "
                "Expected local file like %s",
                _CBRAMOD_DEFAULT_CHECKPOINT,
            )
            return model

        try:
            state = torch.load(str(ckpt_path), map_location="cpu")
            if isinstance(state, dict):
                if "state_dict" in state:
                    state = state["state_dict"]
                elif "model" in state:
                    state = state["model"]
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                logger.warning("CBraMod checkpoint missing %d keys", len(missing))
            if unexpected:
                logger.warning("CBraMod checkpoint unexpected %d keys", len(unexpected))
            logger.info("Loaded CBraMod weights from %s", ckpt_path)
            return model
        except Exception as e:
            logger.warning(
                "Could not load CBraMod checkpoint from %s: %s. "
                "Using random-init architecture.",
                ckpt_path,
                e,
            )
            return model

    def _stub_or_raise(self, reason: str) -> nn.Module:
        if not self.allow_stub_backbone:
            raise RuntimeError(
                f"CBraMod real backbone is unavailable: {reason}. "
                "Set smoke_test=True only for lightweight wrapper tests."
            )
        logger.warning("%s. Using CBraMod stub because smoke-test stubs are enabled.", reason)
        return _CBraModStub(out_dim=self.embed_dim)

    @staticmethod
    def _import_bench_cbramod():
        if not _EEG_FM_BENCH_CBRAMOD_MODEL.exists():
            logger.warning("CBraMod reference file not found at %s", _EEG_FM_BENCH_CBRAMOD_MODEL)
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                "eeg_fm_bench_cbramod_model",
                str(_EEG_FM_BENCH_CBRAMOD_MODEL),
            )
            if spec is None or spec.loader is None:
                raise ImportError("Could not create import spec")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return getattr(module, "CBraMod")
        except Exception as e:
            logger.warning("Failed to import EEG-FM-Bench CBraMod model: %s", e)
            return None

    @staticmethod
    def _resolve_checkpoint_path(
        checkpoint_path: Optional[Union[str, Path]],
    ) -> Optional[Path]:
        candidates: list[Path] = []

        if checkpoint_path is not None:
            ckpt = Path(str(checkpoint_path))
            if not ckpt.is_absolute():
                ckpt = _PROJECT_ROOT / ckpt
            candidates.append(ckpt)

        candidates.append(_CBRAMOD_DEFAULT_CHECKPOINT)

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                if candidate.stat().st_size < 10_000:
                    logger.warning(
                        "CBraMod checkpoint at %s looks too small (%d bytes); ignoring.",
                        candidate,
                        candidate.stat().st_size,
                    )
                    continue
                return candidate
        return None

    # ------------------------------------------------------------------
    # Freeze policy
    # ------------------------------------------------------------------

    def _apply_freeze_policy(self, policy: str, n_partial_layers: int):
        if isinstance(self.backbone, _CBraModStub):
            return

        if policy in ("frozen", "head_only"):
            for p in self.backbone.parameters():
                p.requires_grad = False
            logger.info("CBraMod backbone frozen.")

        elif policy == "partial":
            for p in self.backbone.parameters():
                p.requires_grad = False

            layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
            if layers is not None:
                for layer in list(layers)[-n_partial_layers:]:
                    for p in layer.parameters():
                        p.requires_grad = True

            if hasattr(self.backbone, "proj_out"):
                for p in self.backbone.proj_out.parameters():
                    p.requires_grad = True

            logger.info("CBraMod partial fine-tuning: last %d encoder layers unfrozen", n_partial_layers)

        elif policy == "full":
            for p in self.backbone.parameters():
                if p.is_floating_point() or p.is_complex():
                    p.requires_grad = True

        for p in self.classifier_head.parameters():
            p.requires_grad = True

        for p in self.spatial_adapter.parameters():
            p.requires_grad = True

    # ------------------------------------------------------------------
    # Input shaping + spatial injection
    # ------------------------------------------------------------------

    def _build_classifier_head(
        self,
        classifier_head: Optional[dict[str, Any]],
        dropout: float,
    ) -> nn.Module:
        cfg = dict(classifier_head or {})
        head_type = str(cfg.get("head_type", "avg_pool")).lower()
        hidden_dims = list(cfg.get("hidden_dims", []))
        head_dropout = float(cfg.get("dropout", dropout))

        if head_type == "avg_pool":
            return _CBraModAvgPoolHead(
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
                hidden_dims=hidden_dims,
                dropout=head_dropout,
            )

        n_channels = int(cfg.get("n_channels", self.n_channels))
        n_patches = int(cfg.get("n_patches", max(1, self.eeg_size // self.patch_size)))
        if head_type == "flatten_linear":
            return _CBraModFlattenLinearHead(
                n_patches=n_patches,
                n_channels=n_channels,
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
            )
        if head_type == "flatten_mlp":
            return _CBraModFlattenMLPHead(
                n_patches=n_patches,
                n_channels=n_channels,
                embed_dim=self.embed_dim,
                num_classes=self.num_classes,
                dropout=head_dropout,
            )
        raise ValueError(
            f"Unknown CBraMod classifier head_type={head_type!r}. "
            "Supported: avg_pool, flatten_linear, flatten_mlp."
        )

    def _prepare_input(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply EEG-FM-Bench-style scaling and reshape to CBraMod patches.

        Returns:
            patches: (B, C, n_patches, patch_size)
            channel_mask: (B, C) boolean mask
        """
        B, C, T = x.shape
        x = x.float() * self.input_scale

        channel_mask = normalize_channel_mask(metadata.get("channel_mask"), B, C, x.device)

        x = x * channel_mask.unsqueeze(-1).to(x.dtype)

        n_patches = max(1, T // self.patch_size)
        target_len = n_patches * self.patch_size
        if target_len != T:
            x = F.interpolate(x, size=target_len, mode="linear", align_corners=False)

        patches = x.reshape(B, C, n_patches, self.patch_size).contiguous()
        return patches, channel_mask

    @staticmethod
    def _channel_names_from_metadata(metadata: dict, n_channels: int) -> list[str]:
        return channel_names_from_metadata(metadata, n_channels)

    @staticmethod
    def _reference_meta_from_metadata(metadata: dict):
        return reference_meta_from_metadata(metadata)

    def _build_spatial_embedding(
        self,
        metadata: dict,
        batch_size: int,
        n_channels: int,
        device: torch.device,
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.spatial_adapter.build_embedding(
            metadata=metadata,
            batch_size=batch_size,
            n_channels=n_channels,
            device=device,
        )

    def _prepare_spatial_injection(
        self,
        spatial_emb: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Normalize and center spatial embeddings before converting them into
        native spatial attention biases.

        Raw coordinate MLP outputs can be much larger than CBraMod's pretrained
        internal activations. To keep the spatial signal well behaved across
        datasets and channel layouts, we mask invalid channels, layer-normalize,
        and zero-center the channel states before any pairwise projection.
        """
        return self.spatial_adapter.prepare(spatial_emb, channel_mask)

    def _build_native_attention_bias(
        self,
        spatial_emb: Optional[torch.Tensor],
        attention_bias: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Convert per-channel spatial states into pairwise spatial-attention bias.

        The learned low-rank pairwise term makes every spatial variant usable in
        the same native path, while coords3d_distbias can additionally contribute
        its explicit geometric pairwise bias.
        """
        if spatial_emb is None or not self.use_native_spatial:
            return None
        return self.spatial_adapter.build_pairwise_attention_bias(
            spatial_state=spatial_emb,
            channel_mask=channel_mask,
            attention_bias=attention_bias,
        )

    @staticmethod
    def _expand_spatial_attn_bias(
        spatial_attn_bias: torch.Tensor,
        patch_count: int,
        num_heads: int,
    ) -> torch.Tensor:
        """Expand a (B, C, C) bias to MultiheadAttention's 3D attn_mask format."""
        B, C, _ = spatial_attn_bias.shape
        expanded = spatial_attn_bias.unsqueeze(1).expand(B, patch_count, C, C)
        expanded = expanded.reshape(B * patch_count, C, C)
        if num_heads > 1:
            expanded = expanded.repeat_interleave(num_heads, dim=0)
        return expanded

    def _run_encoder_layer(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        spatial_attn_bias: Optional[torch.Tensor],
        layer_gain: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Execute one CBraMod encoder layer with optional native spatial bias."""
        sa_out = self._spatially_conditioned_sa_block(
            layer,
            layer.norm1(x),
            channel_mask=channel_mask,
            spatial_attn_bias=spatial_attn_bias,
            layer_gain=layer_gain,
        )
        x = x + sa_out
        x = x + layer._ff_block(layer.norm2(x))
        return x

    def _spatially_conditioned_sa_block(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        channel_mask: torch.Tensor,
        spatial_attn_bias: Optional[torch.Tensor],
        layer_gain: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """CBraMod criss-cross self-attention with native channel-attention bias."""
        bz, ch_num, patch_num, patch_size = x.shape
        half_dim = patch_size // 2
        xs = x[:, :, :, :half_dim]
        xt = x[:, :, :, half_dim:]

        xs = xs.transpose(1, 2).contiguous().view(bz * patch_num, ch_num, half_dim)
        xt = xt.contiguous().view(bz * ch_num, patch_num, half_dim)

        attn_mask = None
        if spatial_attn_bias is not None and layer_gain is not None:
            gain = layer_gain.to(device=xs.device, dtype=xs.dtype)
            attn_mask = self._expand_spatial_attn_bias(
                spatial_attn_bias * gain,
                patch_count=patch_num,
                num_heads=layer.self_attn_s.num_heads,
            ).to(dtype=xs.dtype, device=xs.device)

        xs = layer.self_attn_s(
            xs,
            xs,
            xs,
            attn_mask=attn_mask,
            need_weights=False,
        )[0]
        xs = xs.contiguous().view(bz, patch_num, ch_num, half_dim).transpose(1, 2)
        xt = layer.self_attn_t(
            xt,
            xt,
            xt,
            need_weights=False,
        )[0]
        xt = xt.contiguous().view(bz, ch_num, patch_num, half_dim)
        x = torch.concat((xs, xt), dim=3)
        x = x * channel_mask.to(x.dtype).unsqueeze(-1).unsqueeze(-1)
        return layer.dropout1(x)

    def _run_real_backbone(
        self,
        patches: torch.Tensor,
        spatial_emb: Optional[torch.Tensor],
        channel_mask: torch.Tensor,
        attention_bias: Optional[torch.Tensor],
    ) -> torch.Tensor:
        patch_emb = self.backbone.patch_embedding(patches)

        if spatial_emb is not None and not self.use_native_spatial:
            patch_emb = patch_emb + spatial_emb.unsqueeze(2)

        patch_emb = patch_emb * channel_mask.to(patch_emb.dtype).unsqueeze(-1).unsqueeze(-1)

        spatial_attn_bias = self._build_native_attention_bias(
            spatial_emb=spatial_emb,
            attention_bias=attention_bias,
            channel_mask=channel_mask,
        )

        feats = patch_emb
        layers = getattr(getattr(self.backbone, "encoder", None), "layers", None)
        if layers is None:
            feats = self.backbone.encoder(patch_emb)
        else:
            for idx, layer in enumerate(layers):
                layer_gain = self.native_attn_gains[idx] if idx < len(self.native_attn_gains) else None
                feats = self._run_encoder_layer(
                    layer,
                    feats,
                    channel_mask=channel_mask,
                    spatial_attn_bias=spatial_attn_bias,
                    layer_gain=layer_gain,
                )
            encoder_norm = getattr(self.backbone.encoder, "norm", None)
            if encoder_norm is not None:
                feats = encoder_norm(feats)

        feats = self.backbone.proj_out(feats)
        return feats

    def _graph_coords_from_metadata(
        self,
        metadata: dict,
        batch_size: int,
        n_channels: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        coords_3d = metadata.get("coords_3d")
        if coords_3d is not None:
            if coords_3d.dim() == 2:
                coords_3d = coords_3d.unsqueeze(0).expand(batch_size, -1, -1)
            return coords_3d.to(device).float()

        coords_2d = metadata.get("coords_2d")
        if coords_2d is None:
            return None
        if coords_2d.dim() == 2:
            coords_2d = coords_2d.unsqueeze(0).expand(batch_size, -1, -1)
        coords_2d = coords_2d.to(device).float()
        zeros = torch.zeros(batch_size, n_channels, 1, device=device, dtype=coords_2d.dtype)
        return torch.cat([coords_2d, zeros], dim=-1)

    def _build_graph_adjacency(
        self,
        metadata: dict,
        channel_mask: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        B, C = channel_mask.shape
        mask = channel_mask.to(device)
        eye = torch.eye(C, device=device).unsqueeze(0)
        eye_bool = eye.bool()

        coords = self._graph_coords_from_metadata(metadata, B, C, device)
        if coords is None:
            return eye.expand(B, -1, -1)

        dists = torch.cdist(coords, coords)
        valid_pair = mask.unsqueeze(1) & mask.unsqueeze(2)
        valid_dists = dists[valid_pair]
        valid_dists = valid_dists[valid_dists > 0]
        if valid_dists.numel() == 0:
            sigma = torch.tensor(1.0, device=device, dtype=dists.dtype)
        else:
            sigma = valid_dists.median().clamp_min(1e-6)
        sigma = (sigma * self.graph_sigma_scale).clamp_min(1e-6)

        affinity = torch.exp(-torch.square(dists / sigma))
        affinity = affinity * valid_pair.to(affinity.dtype)

        if self.graph_k_neighbors is not None and self.graph_k_neighbors > 0 and C > 1:
            k = min(int(self.graph_k_neighbors), C - 1)
            neighbor_dists = dists.masked_fill(~valid_pair | eye_bool, float("inf"))
            nearest = neighbor_dists.topk(k=k, dim=-1, largest=False).indices
            neighborhood = torch.zeros(B, C, C, dtype=torch.bool, device=device)
            neighborhood.scatter_(-1, nearest, True)
            neighborhood = (neighborhood | neighborhood.transpose(1, 2) | eye_bool) & valid_pair
            affinity = affinity * neighborhood.to(affinity.dtype)

        affinity = affinity + eye * mask.unsqueeze(1).to(affinity.dtype)
        denom = affinity.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return affinity / denom

    def _apply_graph_layers(
        self,
        feats: torch.Tensor,
        metadata: dict,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.graph_layers:
            return feats

        mask = channel_mask.to(feats.dtype).unsqueeze(-1)
        channel_feats = feats.mean(dim=2)
        graph_feats = channel_feats
        adjacency = self._build_graph_adjacency(metadata, channel_mask, feats.device)

        for layer, gain in zip(self.graph_layers, self.graph_gains):
            propagated = torch.matmul(adjacency, graph_feats)
            graph_feats = (graph_feats + gain * layer(propagated)) * mask

        delta = (graph_feats - channel_feats) * mask
        return feats + delta.unsqueeze(2)

    @staticmethod
    def _pool_features(
        feats: torch.Tensor,
        channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        patch_count = feats.shape[2]
        mask = channel_mask.to(feats.dtype).unsqueeze(-1).unsqueeze(-1)
        summed = (feats * mask).sum(dim=(1, 2))
        denom = channel_mask.to(feats.dtype).sum(dim=1).clamp_min(1.0) * float(patch_count)
        return summed / denom.unsqueeze(-1)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        x: torch.Tensor,
        metadata: dict,
    ) -> torch.Tensor:
        B, C, _ = x.shape
        device = x.device

        patches, channel_mask = self._prepare_input(x, metadata)
        spatial_emb, attention_bias = self._build_spatial_embedding(metadata, B, C, device)
        spatial_emb = self._prepare_spatial_injection(spatial_emb, channel_mask)

        if isinstance(self.backbone, _CBraModStub):
            feats = self.backbone(patches, spatial_emb=spatial_emb, channel_mask=channel_mask)
        else:
            try:
                feats = self._run_real_backbone(
                    patches,
                    spatial_emb=spatial_emb,
                    channel_mask=channel_mask,
                    attention_bias=attention_bias,
                )
            except Exception as e:
                if not self.allow_stub_backbone:
                    raise RuntimeError(
                        "CBraMod real backbone forward failed. "
                        "Set smoke_test=True only for lightweight wrapper tests."
                    ) from e
                logger.warning("CBraMod forward error: %s. Falling back to stub features.", e)
                fallback = _CBraModStub(out_dim=self.embed_dim).to(device)
                feats = fallback(patches, spatial_emb=spatial_emb, channel_mask=channel_mask)

        feats = self._apply_graph_layers(feats, metadata, channel_mask)
        return self.classifier_head(feats, channel_mask)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def parameter_count(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        backbone = sum(p.numel() for p in self.backbone.parameters())
        head = sum(p.numel() for p in self.classifier_head.parameters())
        spatial = sum(p.numel() for p in self.spatial_adapter.spatial_embedding.parameters())
        adapter_total = sum(p.numel() for p in self.spatial_adapter.parameters())
        spatial_adapter = adapter_total - spatial + sum(p.numel() for p in self.native_attn_gains)
        graph = sum(p.numel() for p in self.graph_layers.parameters()) + sum(p.numel() for p in self.graph_gains)
        return {
            "total": total,
            "trainable": trainable,
            "backbone": backbone,
            "classifier_head": head,
            "spatial": spatial,
            "spatial_adapter": spatial_adapter,
            "graph": graph,
        }


class _CBraModAvgPoolHead(nn.Module):
    """Masked global average pool over CBraMod channel and patch tokens."""

    def __init__(
        self,
        embed_dim: int,
        num_classes: int,
        hidden_dims: list[int],
        dropout: float,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(embed_dim)]
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, int(hidden_dim)), nn.ELU(), nn.Dropout(dropout)])
            in_dim = int(hidden_dim)
        if not hidden_dims:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(in_dim, num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, feats: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        pooled = CBraModWrapper._pool_features(feats, channel_mask)
        return self.mlp(pooled)


class _CBraModFlattenLinearHead(nn.Module):
    """Officialish flatten-linear head over [patch, channel, feature] tokens."""

    def __init__(
        self,
        n_patches: int,
        n_channels: int,
        embed_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.n_patches = int(n_patches)
        self.n_channels = int(n_channels)
        self.embed_dim = int(embed_dim)
        self.linear = nn.Linear(self.n_patches * self.n_channels * self.embed_dim, num_classes)

    def forward(self, feats: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        flat = self._flatten_checked(feats, channel_mask)
        return self.linear(flat)

    def _flatten_checked(self, feats: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        b, c, p, d = feats.shape
        if (c, p, d) != (self.n_channels, self.n_patches, self.embed_dim):
            raise ValueError(
                "CBraMod flatten head input shape mismatch: "
                f"expected channels={self.n_channels}, patches={self.n_patches}, dim={self.embed_dim}; "
                f"got channels={c}, patches={p}, dim={d}. "
                "Set classifier_head.n_channels and classifier_head.n_patches for this dataset."
            )
        feats = feats * channel_mask.to(feats.dtype).unsqueeze(-1).unsqueeze(-1)
        feats = feats.permute(0, 2, 1, 3).contiguous()
        return feats.reshape(b, -1)


class _CBraModFlattenMLPHead(_CBraModFlattenLinearHead):
    """Officialish three-layer flatten MLP head used by EEG-FM-Bench variants."""

    def __init__(
        self,
        n_patches: int,
        n_channels: int,
        embed_dim: int,
        num_classes: int,
        dropout: float,
    ):
        super().__init__(
            n_patches=n_patches,
            n_channels=n_channels,
            embed_dim=embed_dim,
            num_classes=num_classes,
        )
        hidden1 = self.n_patches * self.embed_dim
        hidden2 = self.embed_dim
        self.linear = nn.Sequential(
            nn.Linear(self.n_patches * self.n_channels * self.embed_dim, hidden1),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, num_classes),
        )


class _CBraModStub(nn.Module):
    """Small fallback backbone used when the reference implementation is unavailable."""

    def __init__(self, out_dim: int = 200):
        super().__init__()
        self.out_dim = out_dim
        self.patch_proj = nn.Linear(1, out_dim)

    def forward(
        self,
        patches: torch.Tensor,
        spatial_emb: Optional[torch.Tensor] = None,
        channel_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        feats = self.patch_proj(patches.mean(dim=-1, keepdim=True))
        if spatial_emb is not None:
            feats = feats + 0.1 * spatial_emb.unsqueeze(2)
        if channel_mask is not None:
            feats = feats * channel_mask.to(feats.dtype).unsqueeze(-1).unsqueeze(-1)
        return feats
