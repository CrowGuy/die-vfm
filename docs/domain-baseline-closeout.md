# Domain Baseline Closeout (`dinov2`, `round1_frozen`)

## Purpose

This document is the formal closeout record for the first domain baseline run
using `dinov2` under current Round1 semantics. It captures the key outputs,
interpretation, and next-step recommendation before Round2 readiness check.

Current contract scope:
- `round1_frozen` single-shot inference/evaluation
- artifact-driven evaluation (`linear_probe`, `knn`, `retrieval`)
- pair benchmark embedding similarity evaluation

Related execution guide:
- [Domain Baseline Playbook](domain-baseline-playbook.md)

## Run Snapshot

Primary run:
- `run.run_name=domain_round1_dinov2_baseline_v1`

Pair benchmark export:
- `run.run_name=pair_benchmark_dinov2_export_v1`

Pair benchmark evaluation:
- `evaluate_pair_benchmark.py` (`join_key=did`)

Slicing analysis:
- `analyze_pair_benchmark_slices.py` with:
  - `confidence=high`
  - `confidence=all`

## Baseline Freeze Record

Baseline ID:
- `domain_round1_dinov2_baseline_v1`

Freeze date:
- `2026-04-29`

Repository commit:
- `b27eaf7762ea385f2bf829a2d16a522c78808fb8`

Canonical run names:
- Round1 baseline: `domain_round1_dinov2_baseline_v1`
- Pair export: `pair_benchmark_dinov2_export_v1`

Frozen artifact index (canonical paths):
- Round1 run root:
  - `runs/domain_round1_dinov2_baseline_v1/round1`
- Round1 summaries:
  - `runs/domain_round1_dinov2_baseline_v1/round1/round1_summary.yaml`
  - `runs/domain_round1_dinov2_baseline_v1/round1/evaluation/linear_probe/summary.yaml`
  - `runs/domain_round1_dinov2_baseline_v1/round1/evaluation/knn/summary.yaml`
  - `runs/domain_round1_dinov2_baseline_v1/round1/evaluation/retrieval/summary.yaml`
- Embedding manifests:
  - `runs/domain_round1_dinov2_baseline_v1/round1/embeddings/train/manifest.yaml`
  - `runs/domain_round1_dinov2_baseline_v1/round1/embeddings/val/manifest.yaml`
- Pair benchmark export root:
  - `runs/pair_benchmark_dinov2_export_v1/round1/embeddings/val`
- Pair evaluation outputs (canonical output dir):
  - output dir: `/abs/path/to/pair_benchmark_eval_v1`
  - `/abs/path/to/pair_benchmark_eval_v1/pair_metrics_summary.yaml`
  - `/abs/path/to/pair_benchmark_eval_v1/pair_scores.csv`
  - `/abs/path/to/pair_benchmark_eval_v1/hard_same_far.csv`
  - `/abs/path/to/pair_benchmark_eval_v1/hard_different_close.csv`
- Slicing outputs (canonical output dirs):
  - high-confidence dir: `/abs/path/to/pair_slice_analysis_high`
  - all-confidence dir: `/abs/path/to/pair_slice_analysis_all`
  - `/abs/path/to/pair_slice_analysis_high/slice_summary.csv`
  - `/abs/path/to/pair_slice_analysis_high/relation_stats.csv`
  - `/abs/path/to/pair_slice_analysis_high/slice_hard_cases.csv`
  - `/abs/path/to/pair_slice_analysis_all/slice_summary.csv`
  - `/abs/path/to/pair_slice_analysis_all/relation_stats.csv`
  - `/abs/path/to/pair_slice_analysis_all/slice_hard_cases.csv`

Frozen input assets (for reproducibility):
- `round1_domain_manifest.csv`
- `configs/dataset/domain_round1_pilot.yaml`
- `pair_benchmark_manifest.csv`
- `pair_candidates.csv`
- `annotations.csv`

## Baseline Artifact IO Record

### Inputs (this baseline iteration)

| Stage | Artifact | Canonical path | Samples |
|---|---|---|---:|
| Round1 baseline | Domain manifest | `round1_domain_manifest.csv` | `137,420` (`train=109,945`, `val=27,475`) |
| Round1 baseline | Dataset config | `configs/dataset/domain_round1_pilot.yaml` | `label_map` for `1,400` classes |
| Pair benchmark export | Pair benchmark manifest | `pair_benchmark_manifest.csv` | reviewed pair image set (`same/different/uncertain`) |
| Pair benchmark eval | Pair candidates | `/abs/path/to/pair_candidates.csv` | candidate pool for reviewed pairs |
| Pair benchmark eval | Annotations | `/abs/path/to/annotations.csv` | `874` reviewed pairs |

### Outputs (this baseline iteration)

| Stage | Artifact | Canonical path | Samples |
|---|---|---|---:|
| Round1 baseline | Train embedding manifest | `runs/domain_round1_dinov2_baseline_v1/round1/embeddings/train/manifest.yaml` | `109,945` |
| Round1 baseline | Val embedding manifest | `runs/domain_round1_dinov2_baseline_v1/round1/embeddings/val/manifest.yaml` | `27,475` |
| Round1 baseline | Round1 summary | `runs/domain_round1_dinov2_baseline_v1/round1/round1_summary.yaml` | train/val/evaluator aggregate |
| Round1 baseline | Evaluator summaries | `runs/domain_round1_dinov2_baseline_v1/round1/evaluation/{linear_probe,knn,retrieval}/summary.yaml` | `num_classes=1,400` |
| Pair benchmark export | Exported pair embeddings | `runs/pair_benchmark_dinov2_export_v1/round1/embeddings/val` | benchmark manifest image set |
| Pair benchmark eval | Pair scores + summary | `/abs/path/to/pair_benchmark_eval_v1/pair_scores.csv`, `/abs/path/to/pair_benchmark_eval_v1/pair_metrics_summary.yaml` | `874` matched reviewed pairs |
| Pair benchmark eval | Hard-case files | `/abs/path/to/pair_benchmark_eval_v1/hard_same_far.csv`, `/abs/path/to/pair_benchmark_eval_v1/hard_different_close.csv` | top hard cases by similarity |
| Slicing analysis | Slice summaries | `/abs/path/to/pair_slice_analysis_high/*` and `/abs/path/to/pair_slice_analysis_all/*` | `confidence=high` and `confidence=all` views |

## Dataset / Provenance Table

| Dataset/Split | Source artifact | Build method | Fixed rule | Samples |
|---|---|---|---|---:|
| `round1_train_reference` | `round1_pilot_train.csv` -> `round1_domain_manifest.csv` (`Source=Train`) | `python -m scripts.build_round1_domain_assets` | pair dataset DIDs excluded before split; group-aware pilot split | `109,945` |
| `round1_val_query` | `round1_pilot_val.csv` -> `round1_domain_manifest.csv` (`Source=Val`) | `python -m scripts.build_round1_domain_assets` | same taxonomy + label_map as train; non-empty val required | `27,475` |
| `pair_benchmark_candidates` | `sample_pool.csv` -> `pair_candidates.csv` | `python -m scripts.generate_pair_candidates` | pair construction rules fixed by script | candidate pool |
| `pair_benchmark_reviewed_pairs` | `pair_candidates.csv` + `annotations.csv` | review tool (`python -m scripts.run_pair_review`) | include only `review_status=reviewed` | `874` pairs |
| `pair_benchmark_export_manifest` | reviewed pairs -> `pair_benchmark_manifest.csv` (`Source=Infer`) | `python -m scripts.build_pair_benchmark_manifest` | relations fixed to `same,different,uncertain`; default `label_mode=empty` | reviewed pair image set |
| `pair_benchmark_eval_set` | pair manifest export embeddings + reviewed pairs | `python -m scripts.evaluate_pair_benchmark` | `join_key=did`; coverage must be reported | `874` matched pairs |

## Data & Coverage

Round1 embedding artifacts:
- Train/reference samples: `109,945`
- Val/query samples: `27,475`
- Embedding dim: `768`

Pair benchmark:
- Reviewed pairs: `874`
- Matched pairs: `874` (`100%`)
- Relation distribution:
  - `same=244`
  - `different=420`
  - `uncertain=210`

## Evaluator Results

Label-based evaluators:
- `linear_probe.accuracy = 0.4421`
- `knn.top1_accuracy = 0.3595`
- `knn.top5_accuracy = 0.6098`
- `retrieval.recall_at_1 = 0.3660`
- `retrieval.recall_at_5 = 0.6095`

Pair benchmark separation:
- `same_vs_different_cosine_auc_like = 0.8135`

Interpretation:
- Baseline has clear signal in both label-based and pair-based views.
- Pair benchmark shows stronger alignment with target objective (visual
  sameness) than fine-label-only interpretation.

## Slicing Analysis Summary

### `confidence=high`
- `source_slice`
  - `same_source` AUC-like: `0.8382`
  - `cross_source` AUC-like: `0.7510`
- `normality_slice`
  - `both_normal`: `0.7995`
  - `both_abnormal`: `0.8096`
- `fine_label_slice`
  - `same_fine_label`: `0.7583`
  - `different_fine_label`: `0.7288`

### `confidence=all`
- `source_slice`
  - `same_source` AUC-like: `0.8356`
  - `cross_source` AUC-like: `0.7510`
- `normality_slice`
  - `both_normal`: `0.7995`
  - `both_abnormal`: `0.8069`
- `fine_label_slice`
  - `same_fine_label`: `0.7554`
  - `different_fine_label`: `0.7234`

Stability check:
- Main trends are consistent between `high` and `all`.
- Conclusions are robust to confidence filtering.

## Hard-Case Conclusions

1. `cross_source` is the most stable weak slice.
2. Normal-only pairs are not uniquely hardest versus abnormal-only pairs.
3. Fine-label mismatch is structural:
   - different fine labels can be visually similar (aliasing),
   - same fine labels can still contain visually different modes.

## Final Conclusion

`dinov2 round1_frozen` baseline is successful and reproducible under current
scope. It provides a strong enough baseline signal to support next-phase
planning, while also surfacing clear bottlenecks:
- cross-source invariance
- fine-label / visual-concept mismatch

## Recommendation Before Round2 Readiness Check

Complete these items first:
1. Freeze canonical artifact index for this baseline iteration (paths + run ids):
   completed in `Baseline Freeze Record` and `Baseline Artifact IO Record`.
2. Keep pair benchmark manifest generation script as the official entrypoint:
   `scripts/build_pair_benchmark_manifest.py`.
3. Use this closeout as the baseline reference for any Round2 readiness review
   or future backbone comparison.
