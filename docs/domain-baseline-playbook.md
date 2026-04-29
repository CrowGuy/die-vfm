# Domain Baseline Playbook (`dinov2`, `round1_frozen`)

This playbook captures the full domain baseline flow and closeout artifacts so
the run is reproducible by anyone on the team.

Current scope follows repository current-spec semantics:
- `bootstrap`
- `round1_frozen`
- artifact-driven evaluators
- offline local-asset `dinov2` path

Round2+ training semantics are not part of this playbook.

## 1. What This Playbook Delivers

1. Reproducible run commands for:
   - Round1 domain baseline
   - Pair benchmark embedding export
   - Pair benchmark evaluation
   - Pair benchmark slicing analysis
2. Input/output contract for each script step.
3. Baseline closeout summary for the latest run.

## 2. Pipeline Dependency Graph

1. `sample_pool.csv`
2. `scripts/generate_pair_candidates.py` -> `pair_candidates.csv`
3. Pair review tool (`scripts/run_pair_review.py`) -> `annotations.csv`
4. Prepare Round1 pilot train/val CSVs (outside this repo flow)
5. `scripts/build_round1_domain_assets.py` ->
   - `round1_domain_manifest.csv`
   - `configs/dataset/domain_round1_pilot.yaml` (with `label_map`)
6. `python -m scripts.run experiment=round1_frozen` ->
   - baseline embeddings/evaluators/summary
7. `scripts/build_pair_benchmark_manifest.py` -> `pair_benchmark_manifest.csv`
8. `python -m scripts.run experiment=domain_inference_export` ->
   - pair benchmark embeddings
9. `scripts/evaluate_pair_benchmark.py` ->
   - `pair_scores.csv`, summary, hard-cases
10. `scripts/analyze_pair_benchmark_slices.py` ->
    - slicing summary (`high` and `all`)

## 3. Script IO Contract

### `scripts/generate_pair_candidates.py`
- Input:
  - `--input` sampled image pool CSV with required columns:
    `did,image_id,image_path,fine_label,freq_bin,lot,machine,time_bucket`
- Output:
  - `pair_candidates.csv`
  - `pair_candidates_summary.csv`

### Pair review tool (`scripts/run_pair_review.py`)
- Input:
  - `--pairs` `pair_candidates.csv`
  - `--annotations` `annotations.csv` (created if missing)
- Output:
  - Updated `annotations.csv` with `review_status`, `visual_relation`, etc.

### `scripts/build_round1_domain_assets.py`
- Input:
  - `--train-csv` `round1_pilot_train.csv`
  - `--val-csv` `round1_pilot_val.csv`
- Output:
  - Domain manifest CSV (`DID, IMG_1, Source, Label, PATH`)
  - Dataset config YAML with embedded `label_map`

### `scripts/evaluate_pair_benchmark.py`
- Input:
  - `--pair-candidates` `pair_candidates.csv`
  - `--annotations` `annotations.csv`
  - `--embedding-split-dir` one or more embedding split dirs
- Output:
  - `pair_scores.csv`
  - `unmatched_pairs.csv`
  - `pair_metrics_summary.yaml` / `.json`
  - `hard_same_far.csv`
  - `hard_different_close.csv`
  - `uncertain_high_similarity.csv`

### `scripts/build_pair_benchmark_manifest.py`
- Input:
  - `--pair-candidates` `pair_candidates.csv`
  - `--annotations` `annotations.csv`
- Output:
  - `pair_benchmark_manifest.csv` with `Source=Infer` rows for reviewed pairs
- Required columns:
  - `pair_candidates.csv`: `pair_id,did_a,did_b,image_id_a,image_id_b,image_path_a,image_path_b`
  - `annotations.csv`: `pair_id,review_status,visual_relation`
- Default behavior:
  - includes only `review_status=reviewed`
  - relation filter defaults to `same,different,uncertain`
  - output `Source` defaults to `Infer`
  - `--label-mode empty` keeps `Label` blank for inference-only export
  - `--label-mode fine_label` copies `fine_label_a/fine_label_b` into manifest `Label`
    (use this only when you explicitly want a labeled pair-manifest variant)
  - `DID` is deduplicated; conflicting duplicated `DID` rows fail fast

### `scripts/analyze_pair_benchmark_slices.py`
- Input:
  - `--pair-scores` `pair_scores.csv`
  - `--pair-candidates` `pair_candidates.csv`
- Output:
  - `slice_summary.csv`
  - `relation_stats.csv`
  - `slice_hard_cases.csv`
  - `slice_analysis_summary.yaml` / `.json`

## 4. Reproducible Commands

All commands below use module mode (`python -m ...`) to avoid import-path issues.

### 4.1 Generate pair candidates

```bash
python -m scripts.generate_pair_candidates \
  --input /abs/path/to/sample_pool.csv \
  --output /abs/path/to/pair_candidates.csv \
  --summary-output /abs/path/to/pair_candidates_summary.csv
```

### 4.2 Launch pair review tool

```bash
python -m scripts.run_pair_review \
  --pairs /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv \
  --host 0.0.0.0 \
  --port 8000
```

### 4.3 Build Round1 domain assets

```bash
python -m scripts.build_round1_domain_assets \
  --train-csv /abs/path/to/round1_pilot_train.csv \
  --val-csv /abs/path/to/round1_pilot_val.csv \
  --manifest-output /abs/path/to/round1_domain_manifest.csv \
  --dataset-config-output /abs/path/to/configs/dataset/domain_round1_pilot.yaml \
  --require-non-empty-val
```

### 4.4 Run Round1 baseline (`dinov2`, offline local assets)

```bash
python -m scripts.run \
  experiment=round1_frozen \
  run.run_name=domain_round1_dinov2_baseline_v1 \
  system.device=cuda \
  system.num_workers=4 \
  dataloader.pin_memory=true \
  model/backbone=dinov2 \
  model/pooler=mean \
  model.backbone.variant=vit_base \
  model.backbone.pretrained=true \
  model.backbone.freeze=true \
  model.backbone.allow_network=false \
  model.backbone.local_repo_path=/opt/dinov2_assets/repo/dinov2 \
  model.backbone.local_checkpoint_path=/opt/dinov2_assets/checkpoints/dinov2_vitb14_pretrain.pth \
  train.freeze_backbone=true \
  train.freeze_pooler=true \
  dataset=domain_round1_pilot \
  evaluation.knn.evaluator.k=5 \
  evaluation.linear_probe.trainer.device=cuda \
  evaluation.knn.evaluator.device=cuda \
  evaluation.retrieval.evaluator.device=cuda
```

### 4.5 Export pair benchmark embeddings (inference-only)

Build pair benchmark manifest first:

```bash
python -m scripts.build_pair_benchmark_manifest \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv \
  --output /abs/path/to/pair_benchmark_manifest.csv \
  --relations same,different,uncertain \
  --label-mode empty \
  --image-path-mode directory
```

`label-mode` selection:
- `empty` (recommended): for `experiment=domain_inference_export` embedding-only path.
- `fine_label`: for an explicit labeled variant (for debugging or custom
  label-based evaluator experiments on the pair image set).

```bash
python -m scripts.run \
  experiment=domain_inference_export \
  run.run_name=pair_benchmark_dinov2_export_v1 \
  system.device=cuda \
  system.num_workers=4 \
  dataloader.pin_memory=true \
  model/backbone=dinov2 \
  model/pooler=mean \
  model.backbone.variant=vit_base \
  model.backbone.pretrained=true \
  model.backbone.freeze=true \
  model.backbone.allow_network=false \
  model.backbone.local_repo_path=/opt/dinov2_assets/repo/dinov2 \
  model.backbone.local_checkpoint_path=/opt/dinov2_assets/checkpoints/dinov2_vitb14_pretrain.pth \
  train.freeze_backbone=true \
  train.freeze_pooler=true \
  dataset=domain \
  dataset.manifest_path=/abs/path/to/pair_benchmark_manifest.csv \
  dataset.require_non_empty_val=true
```

### 4.6 Evaluate pair benchmark

```bash
python -m scripts.evaluate_pair_benchmark \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv \
  --embedding-split-dir /abs/path/to/runs/pair_benchmark_dinov2_export_v1/round1/embeddings/val \
  --join-key did \
  --output-dir /abs/path/to/pair_benchmark_eval_v1 \
  --hard-limit 50
```

### 4.7 Slicing analysis (`high` and `all`)

```bash
python -m scripts.analyze_pair_benchmark_slices \
  --pair-scores /abs/path/to/pair_benchmark_eval_v1/pair_scores.csv \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --output-dir /abs/path/to/pair_slice_analysis_high \
  --confidence high \
  --hard-limit 20
```

```bash
python -m scripts.analyze_pair_benchmark_slices \
  --pair-scores /abs/path/to/pair_benchmark_eval_v1/pair_scores.csv \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --output-dir /abs/path/to/pair_slice_analysis_all \
  --confidence all \
  --hard-limit 20
```

## 5. Offline Asset Path Rules (`dinov2`)

For `allow_network=false` runs:
1. `model.backbone.local_repo_path` must point to a local DINOv2 repo.
2. If `model.backbone.pretrained=true`, `model.backbone.local_checkpoint_path` is required.
3. `variant` must match checkpoint family (e.g., `vit_base` with `dinov2_vitb14_pretrain.pth`).

## 6. Baseline Closeout Snapshot (Current Run)

Reference run:
- `run.run_name=domain_round1_dinov2_baseline_v1`

### Round1 summary
- Train embeddings: `109,945`
- Val embeddings: `27,475`
- Embedding dim: `768`
- Linear probe accuracy: `0.4421`
- kNN top1/top5: `0.3595 / 0.6098`
- Retrieval recall@1/recall@5: `0.3660 / 0.6095`

### Pair benchmark summary
- Reviewed pairs: `874`
- Matched pairs: `874` (`100%`)
- Relation counts:
  - `same=244`
  - `different=420`
  - `uncertain=210`
- `same_vs_different_cosine_auc_like=0.8135`

### Slicing summary (high confidence)
- `same_source` AUC-like: `0.8382`
- `cross_source` AUC-like: `0.7510`
- `both_normal` AUC-like: `0.7995`
- `both_abnormal` AUC-like: `0.8096`
- `same_fine_label` AUC-like: `0.7583`
- `different_fine_label` AUC-like: `0.7288`

### Slicing summary (all confidence)
- `same_source` AUC-like: `0.8356`
- `cross_source` AUC-like: `0.7510`
- `both_normal` AUC-like: `0.7995`
- `both_abnormal` AUC-like: `0.8069`
- `same_fine_label` AUC-like: `0.7554`
- `different_fine_label` AUC-like: `0.7234`

### Hard-case conclusions
1. `cross_source` is the most stable weak slice.
2. Normal-only pairs are not uniquely hardest versus abnormal-only pairs.
3. Fine-label mismatch is structural:
   - different fine labels can be visually similar (aliasing),
   - same fine labels can still contain visually different modes.

## 7. Closeout Artifact Checklist

For each baseline iteration, archive:
1. Run command block (exact CLI + run_name).
2. Dataset assets:
   - `round1_domain_manifest.csv`
   - `configs/dataset/domain_round1_pilot.yaml`
3. Round1 outputs:
   - `round1_summary.yaml`
   - `embeddings/train/manifest.yaml`
   - `embeddings/val/manifest.yaml`
   - evaluator summaries (`linear_probe`, `knn`, `retrieval`)
4. Pair benchmark outputs:
   - `pair_metrics_summary.yaml`
   - `pair_scores.csv`
   - `hard_same_far.csv`
   - `hard_different_close.csv`
5. Slicing outputs:
   - `slice_summary.csv`
   - `relation_stats.csv`
   - `slice_hard_cases.csv`
6. Final narrative:
   - baseline signal
   - hard-case pattern
   - next-step recommendation
