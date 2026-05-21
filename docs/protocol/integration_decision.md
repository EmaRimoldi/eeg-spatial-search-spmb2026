# Integration Decision

**Date**: 2026-04-08
**Decision**: Standalone wrapper + EEG-FM-Bench evaluation hybrid

---

## Reconnaissance summary

After inspecting all five external repositories:

| Repo | Role | Integration path |
|------|------|-----------------|
| EEG-FM-Bench | Benchmark analysis/viz | Reference only |
| REVE | Primary backbone | HF loading + custom wrapper |
| LaBraM | Secondary backbone | Local checkpoint + custom wrapper |
| CBraMod | Deferred | Use if LaBraM insufficient |
| BIOT | Deferred | Use only if needed |

---

## Chosen route

**Standalone wrapper with EEG-FM-Bench as reference**

### Rationale

1. **EEG-FM-Bench** is primarily an analysis/visualization framework, not a
   unified training engine for custom model variants. It would require significant
   invasive modifications to support our custom spatial embedding modules.

2. **REVE's `src/dt.py`** is designed for its specific downstream tasks and
   config structure. Integrating our 7 spatial variants cleanly would require
   deep modification of their training loop.

3. **Our custom wrappers** (`src/models/wrappers/`) provide:
   - A clean, ablation-friendly interface
   - Backbone swapping via registry
   - Consistent metadata injection
   - No modification of external repos

4. **REVE's HuggingFace model** can be loaded via `from_pretrained` and used
   as a frozen/partially-frozen backbone without touching their codebase.

### Architecture

```
Our training loop (src/training/train.py)
    ↓
Model registry (src/models/wrappers/registry.py)
    ↓
{REVEWrapper, LaBraMWrapper, EEGNetWrapper}
    ↓
SpatialEmbedding variant (injected)
    ↓
Backbone (frozen/unfrozen per policy)
    ↓
Classifier head
```

### Spatial injection mechanism

For REVE: positions tensor is intercepted. Custom spatial embedding replaces
or augments REVE's `FourierEmb4D`. Three modes:
- `replace`: Zero out REVE's positions, add our embedding instead
- `reve_default`: Use REVE's FourierEmb4D unchanged (for comparison)
- `augment`: Keep REVE's PE, add our embedding on top

### EEG-FM-Bench usage

Use EEG-FM-Bench as:
- Dataset preprocessing reference
- Baseline numbers for comparison tables
- Final visualization tooling

---

## Risks and mitigations

| Risk | Mitigation |
|------|------------|
| REVE checkpoint not available | HF download script; stub mode for testing |
| LaBraM Git LFS weights | `git lfs pull` script provided |
| MOABB not installed | Hardcoded channel lists as fallback |
| MNE not installed | Hardcoded coordinate table |

---

## Decision author

Codex agent, based on reconnaissance notes in `docs/notes/`.
