# Paper Outline

## Title

**What Spatial Information Matters for EEG Generalization?**  
**A Controlled Study of Coordinates, Reference, and Layout Robustness**

---

## Abstract structure

1. **Problem**: EEG foundation models increasingly claim montage robustness,
   but the role of explicit spatial information remains unclear.

2. **Gap**: Prior work mixes coordinates with broader architectural changes
   and large-scale pretraining, making it impossible to isolate the effect
   of spatial encoding.

3. **Method**: Controlled comparison of 7 spatial-information variants under
   matched training budgets, architectures, and transfer regimes.

4. **Findings**: [To be filled after experiments]

5. **Conclusion**: [To be filled after experiments — one of the following]
   - Explicit 3D geometry is necessary for robust cross-layout transfer
   - Reference-scheme encoding adds meaningful signal beyond coordinates
   - Layout-robust design explains most of the apparent geometry benefit
   - Geometry matters primarily under scarce supervision

---

## Section 1: Introduction (~800 words)

- The practical problem: EEG is acquired with diverse montages; models must generalize
- What current FMs do for spatial information (REVE, LaBraM, CBraMod, BIOT)
- The gap: no controlled study isolating the spatial component
- Our contribution: systematic ablation of spatial variants
- Claims we can make (map to experiments via claim-evidence bridge)

## Section 2: Related Work (~600 words)

- EEG foundation models (REVE, LaBraM, BIOT, CBraMod, EEGFormer, ...)
- Spatial encoding in EEG (electrode coordinates, positional embeddings)
- Transfer learning for EEG and the montage heterogeneity problem
- Graph neural approaches for EEG topology

## Section 3: Experimental Framework (~500 words)

- Backbone architecture (REVE)
- Downstream datasets (BNCI2014_001, PhysionetMI, TUAB if available)
- Training regimes (linear probe, head-only, partial fine-tune)
- Evaluation metrics (balanced accuracy, AUROC)
- Transfer conditions (same/reduced/unseen layout)

## Section 4: Spatial Representation Variants (~600 words)

- Taxonomy of spatial encoding approaches
- 7 variants described (none → channel_id → coords2d → coords3d → ...)
- Shared interface design and injection mechanism
- What each variant tests scientifically

## Section 5: Main Results (~800 words)

- Core ablation table (Stages 1–2 results)
- Which variants help and in which regimes
- Interaction with training regime (is geometry more useful with less tuning?)
- Primary finding summary

## Section 6: Transfer and Robustness (~700 words)

- Cross-layout transfer results (Stage 3)
- Missing-channel robustness curves (Stage 4)
- Few-shot performance curves (Stage 5)
- Key insight: when does geometry help most?

## Section 7: Discussion (~500 words)

- What actually matters for EEG generalization?
- Interpreting the result (one of the 4 scenarios from statistical plan)
- Practical implications for model design
- Connection to brain topology and reference standardization

## Section 8: Limitations (~200 words)

- Limited to motor imagery / classification tasks in initial pass
- REVE as primary backbone; conclusions may not generalize to all FMs
- Coordinate quality depends on standardized montage assumption
- Compute constraints (limited seed count, limited dataset range)

## Section 9: Conclusion (~200 words)

---

## Appendix

- Full ablation tables (all variants × all regimes × all datasets)
- Checkpoint provenance table
- Channel metadata methodology
- Statistical tests (all pairwise comparisons)

---

## Figures plan

| Figure | Content | Script |
|--------|---------|--------|
| 1 | System diagram: variants + wrapper architecture | Manual |
| 2 | Core ablation bar chart | plot_core_ablation.py |
| 3 | Robustness curves vs dropout rate | plot_robustness.py |
| 4 | Few-shot learning curves | plot_fewshot.py |
| 5 | Transfer gap heatmap | (to be added) |

---

## Writing policy (from manual)

**Allowed claims**:
- "Under matched architecture and budget, variant X outperformed variant Y in regime Z."
- "Geometry was most beneficial under montage shift / few-shot settings."
- "Reference-aware encoding added benefit beyond coordinates in condition A."

**Forbidden overclaims**:
- "3D coordinates universally improve EEG models."
- "This proves brain-like spatial reasoning."
- Any claim not directly backed by ablations.
