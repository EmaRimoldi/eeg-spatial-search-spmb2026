# Experiment Matrix

## Stage 1 — Pilot (9 runs)

| Backbone | Dataset | Variants | Regime | Seeds | Runs |
|----------|---------|----------|--------|-------|------|
| REVE | BNCI2014_001 | channel_id, coords2d, coords3d | head_only | 3 | 9 |

**Success criterion**: all runs complete, metrics exported, no silent bugs.

---

## Stage 2 — Core Ablation (63 runs)

| Backbone | Dataset | Variants | Regimes | Seeds | Runs |
|----------|---------|----------|---------|-------|------|
| REVE | BNCI2014_001 | all 7 | linear_probe, head_only, partial | 3 | 63 |

**Scientific questions**:
- Does spatial information help at all? (none vs channel_id)
- Does 3D beat 2D? (coords2d vs coords3d)
- Does distance bias add anything? (coords3d vs coords3d_distbias)
- Does reference metadata matter? (coords3d vs coords3d_reference)
- Is topology-agnostic competitive? (topology_agnostic vs coords3d_reference)

---

## Stage 3 — Cross-Layout Transfer (~36 runs)

| Variants | Transfer conditions | Seeds | Runs |
|----------|---------------------|-------|------|
| 4 (strongest) | same, reduced, cross-dataset | 3 | ~36 |

---

## Stage 4 — Channel Dropout Robustness (~60 runs)

| Variants | Dropout protocols | Seeds | Runs |
|----------|-------------------|-------|------|
| 4 | random_p01, p03, p05, structured, sparse | 3 | ~60 |

---

## Stage 5 — Few-Shot (~75 runs)

| Variants | Label fractions | Seeds | Runs |
|----------|-----------------|-------|------|
| 5 | 1%, 5%, 10%, 25%, 100% | 3 | 75 |

---

## Stage 6 — Secondary Backbone (partial replication)

| Backbone | Variants | Regimes | Seeds | Runs |
|----------|----------|---------|-------|------|
| LaBraM | 4 | head_only | 3 | ~12 |

---

## Total estimated runs: ~255

All runs must save:
- `config_resolved.yaml`
- `metrics.json`
- `git_state.txt`
- `system_info.json`
- `channel_metadata_snapshot.csv`
