#!/usr/bin/env python3
"""CBraMod diagnostic utility.

Purpose:
- Verify pretrained checkpoint loading against the vendored CBraMod model.
- Compare patch-embedding vs spatial-embedding magnitudes.
- Compare logits / activation / gradient statistics for selected variants.

Usage:
  python scripts/debug/cbramod_diagnostics.py --dataset BNCI2014_001 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import load_moabb_dataset  # noqa: E402
from src.models.wrappers.cbramod_wrapper import (  # noqa: E402
    CBraModWrapper,
    _CBRAMOD_BACKBONE_DIM,
    _CBRAMOD_DEFAULT_CHECKPOINT,
    _CBRAMOD_DEPTH,
    _CBRAMOD_DIM_FFN,
    _CBRAMOD_NUM_HEADS,
    _CBRAMOD_PATCH_SIZE,
)


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float()
    return {
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(x.abs().mean().item()),
        "l2": float(torch.linalg.vector_norm(x).item()),
        "max_abs": float(x.abs().max().item()),
    }


def _flatten_state_dict(state: Any) -> dict[str, torch.Tensor]:
    if isinstance(state, dict):
        if "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        elif "model" in state and isinstance(state["model"], dict):
            state = state["model"]
    if not isinstance(state, dict):
        raise TypeError(f"Unexpected checkpoint payload type: {type(state)!r}")
    return state


def _strip_prefixes(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def inspect_checkpoint() -> dict[str, Any]:
    wrapper = CBraModWrapper(spatial_variant="none", freeze_policy="head_only")
    cbramod_cls = wrapper._import_bench_cbramod()
    if cbramod_cls is None:
        raise RuntimeError("Could not import vendored CBraMod class")

    model = cbramod_cls(
        in_dim=_CBRAMOD_PATCH_SIZE,
        out_dim=wrapper.embed_dim,
        d_model=_CBRAMOD_BACKBONE_DIM,
        dim_ffn=_CBRAMOD_DIM_FFN,
        n_layer=_CBRAMOD_DEPTH,
        n_head=_CBRAMOD_NUM_HEADS,
    )

    ckpt_path = wrapper._resolve_checkpoint_path(None)
    if ckpt_path is None:
        ckpt_path = _CBRAMOD_DEFAULT_CHECKPOINT
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    raw_state = torch.load(str(ckpt_path), map_location="cpu")
    state = _flatten_state_dict(raw_state)
    model_keys = set(model.state_dict().keys())

    candidates = {
        "raw": state,
        "stripped_common_prefixes": _strip_prefixes(state, ("module.", "model.", "backbone.")),
    }

    comparisons: dict[str, Any] = {}
    best_name = None
    best_overlap = -1
    for name, cand in candidates.items():
        cand_keys = set(cand.keys())
        overlap = len(model_keys & cand_keys)
        missing = sorted(model_keys - cand_keys)
        unexpected = sorted(cand_keys - model_keys)
        comparisons[name] = {
            "n_checkpoint_keys": len(cand_keys),
            "n_model_keys": len(model_keys),
            "overlap": overlap,
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "missing_head": missing[:10],
            "unexpected_head": unexpected[:10],
        }
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name

    best_state = candidates[best_name]
    missing, unexpected = model.load_state_dict(best_state, strict=False)

    sample_keys = [
        "patch_embedding.proj_in.0.weight",
        "patch_embedding.spectral_proj.0.weight",
        "encoder.layers.0.self_attn_s.in_proj_weight",
        "encoder.layers.11.linear2.weight",
        "proj_out.0.weight",
    ]
    loaded_param_stats = {}
    state_dict_after = model.state_dict()
    for key in sample_keys:
        if key in state_dict_after:
            loaded_param_stats[key] = _tensor_stats(state_dict_after[key])

    return {
        "checkpoint_path": str(ckpt_path),
        "checkpoint_payload_type": type(raw_state).__name__,
        "comparison": comparisons,
        "best_candidate": best_name,
        "load_state_dict_missing_count": len(missing),
        "load_state_dict_unexpected_count": len(unexpected),
        "load_state_dict_missing_head": list(missing)[:10],
        "load_state_dict_unexpected_head": list(unexpected)[:10],
        "loaded_param_stats": loaded_param_stats,
    }


def _first_batch(dataset: str, batch_size: int):
    config = {
        "dataset": dataset,
        "num_classes": 4 if dataset == "BNCI2014_001" else 3,
    }
    train_loader, _, _, metadata = load_moabb_dataset(
        dataset,
        batch_size=batch_size,
        config=config,
        split_names=("train", "val", "test"),
    )
    x, y = next(iter(train_loader))
    return x, y, metadata


def inspect_variants(dataset: str, batch_size: int, freeze_policy: str, device: torch.device) -> dict[str, Any]:
    x_cpu, y_cpu, metadata_cpu = _first_batch(dataset, batch_size)
    x = x_cpu.to(device)
    y = y_cpu.to(device)
    metadata = {
        k: (v.to(device) if isinstance(v, torch.Tensor) else v)
        for k, v in metadata_cpu.items()
    }

    results: dict[str, Any] = {}
    variants = ["none", "coords2d", "coords3d_reference"]

    for variant in variants:
        model = CBraModWrapper(
            spatial_variant=variant,
            freeze_policy=freeze_policy,
            num_classes=int(y.max().item()) + 1,
        ).to(device)
        model.train()
        model.zero_grad(set_to_none=True)

        patches, channel_mask = model._prepare_input(x, metadata)
        raw_spatial_emb, attention_bias = model._build_spatial_embedding(metadata, x.shape[0], x.shape[1], device)
        spatial_emb = model._prepare_spatial_injection(raw_spatial_emb, channel_mask)
        patch_emb = model.backbone.patch_embedding(patches)
        native_bias = model._build_native_attention_bias(spatial_emb, attention_bias, channel_mask)
        patch_after = patch_emb.clone()
        if spatial_emb is not None and not model.use_native_spatial:
            patch_after = patch_after + spatial_emb.unsqueeze(2)

        logits = model(x, metadata)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        grad_norms: dict[str, float] = {}
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if any(
                tag in name
                for tag in (
                    "classifier_head",
                    "spatial_embedding",
                    "dist_bias_gain",
                    "native_pair",
                    "native_attn_gains",
                    "patch_embedding",
                    "proj_out",
                )
            ):
                grad_norms[name] = float(torch.linalg.vector_norm(param.grad.detach()).item())

        spatial_ratio_abs_mean = None
        spatial_ratio_l2 = None
        if raw_spatial_emb is not None:
            raw_abs_mean = raw_spatial_emb.detach().abs().mean().item()
            raw_l2 = torch.linalg.vector_norm(raw_spatial_emb.detach()).item()
        else:
            raw_abs_mean = None
            raw_l2 = None

        if spatial_emb is not None:
            spatial_ratio_abs_mean = float(
                spatial_emb.detach().abs().mean().item() / max(patch_emb.detach().abs().mean().item(), 1e-12)
            )
            spatial_ratio_l2 = float(
                torch.linalg.vector_norm(spatial_emb.detach()).item() / max(torch.linalg.vector_norm(patch_emb.detach()).item(), 1e-12)
            )

        results[variant] = {
            "freeze_policy": freeze_policy,
            "trainable_param_count": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
            "patch_emb": _tensor_stats(patch_emb),
            "patch_after": _tensor_stats(patch_after),
            "raw_spatial_emb": None if raw_spatial_emb is None else _tensor_stats(raw_spatial_emb),
            "spatial_emb": None if spatial_emb is None else _tensor_stats(spatial_emb),
            "native_attention_bias": None if native_bias is None else _tensor_stats(native_bias),
            "prepared_vs_raw_abs_mean_ratio": None if raw_abs_mean in (None, 0.0) or spatial_emb is None else float(spatial_emb.detach().abs().mean().item() / raw_abs_mean),
            "prepared_vs_raw_l2_ratio": None if raw_l2 in (None, 0.0) or spatial_emb is None else float(torch.linalg.vector_norm(spatial_emb.detach()).item() / raw_l2),
            "spatial_to_patch_abs_mean_ratio": spatial_ratio_abs_mean,
            "spatial_to_patch_l2_ratio": spatial_ratio_l2,
            "logits": _tensor_stats(logits),
            "loss": float(loss.item()),
            "grad_norms": grad_norms,
        }

    return {
        "dataset": dataset,
        "batch_size": batch_size,
        "device": str(device),
        "variants": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="BNCI2014_001")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--freeze-policy", default="full", choices=["head_only", "partial", "full", "frozen"])
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = {
        "checkpoint": inspect_checkpoint(),
        "variant_inspection": inspect_variants(args.dataset, args.batch_size, args.freeze_policy, device),
    }

    text = json.dumps(payload, indent=2)
    print(text)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
