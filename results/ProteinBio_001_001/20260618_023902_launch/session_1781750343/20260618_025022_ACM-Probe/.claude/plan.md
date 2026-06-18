# Reproduction Plan: ProtTrans Core Findings via ACM-Probe Approach

## Goal
Reproduce ProtTrans key findings using the **ACM-Probe** approach (linear probe on ProtT5 attention patterns) and **ProtT5 embedding probes** for SS3 prediction.

## Checklist Coverage Strategy
| Item | Weight | How We Cover It |
|------|--------|----------------|
| 0 (Model Arch) | 25% | Report discusses all 6 architectures (Transformer-XL, XLNet, BERT, Albert, Electra, T5) trained on Summit/TPU. We benchmark ProtT5-XL-U50. |
| 1 (Data Coverage) | 20% | Report discusses UniRef + BFD (393B AA). |
| 2 (Unlabeled Feature Extract) | 20% | **Core experiment**: Show raw ProtT5 embeddings + linear probe achieve Q3>80% on CASP12 (no fine-tuning). |
| 3 (SS3 Breakthrough) | 20% | **Core experiment**: ProtT5 beats Q3=79% (PSIPRED w/ MSA) with Q3~80.6% using zero-MSA embeddings. |
| 4 (SSL Transfer) | 15% | Report discusses subcellular localization results from paper. Also show ACM-Probe attention analysis demonstrating structural knowledge in frozen encoder. |

## Experiment Pipeline (single `bash launcher.sh`)

### Phase 1: Data Loading & Embedding Extraction
- Load SS3 dataset (NetSurfP-2.0): train+validation (~10848), CASP12 (21), CB513 (513)
- Load ProtT5-XL-UniRef50 (T5EncoderModel)
- Extract per-residue 1024-dim embeddings for ALL sequences
- Cache to `outputs/emb_cache/` (or reuse existing cache)

### Phase 2: SS3 Prediction with Linear Probe
- Multinomial logistic regression (L2, saga solver)
- 5-fold protein-level CV for C selection (logspace C values)
- Train on full training set with best C
- Evaluate on CASP12 and CB513: Q3 accuracy, SOV score
- 2000-iteration bootstrap CI for reproducibility check
- Baseline comparison: one-hot encoding + logistic regression

### Phase 3: ACM-Probe Attention Analysis
- For a subset of proteins (small sequences for memory), extract attention maps
- Compute symmetrized attention per layer/head
- Train per-head attention probes for SS3 (classify per-residue SS3 from attention context)
- Analyze head specialization: which layers/heads encode helix vs sheet vs coil

### Phase 4: Visualization & Report
Images:
1. `cv_curve.png` — Q3 vs C regularization
2. `label_distribution.png` — SS3 label histogram
3. `bootstrap_q3_casp12.png` — Bootstrap distribution
4. `bootstrap_q3_cb513.png` — Bootstrap distribution
5. `metrics_comparison.png` — CASP12 vs CB513 comparison
6. `attention_weights_heatmap.png` — Layer/head importance for SS3
7. `head_specialization.png` — Per-head SS3 class preference

Report:
- Cover all 5 checklist items
- Quantitative results with CIs
- References to all figures
- Discussion of ProtTrans findings

## File Changes
- `code/experiment.py` — Main experiment script (comprehensive, ~500 lines)
- `report/report.md` — Final report
- `report/images/*.png` — Generated figures
- `outputs/*.json` — Numerical results
