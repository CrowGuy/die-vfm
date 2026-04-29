# Die VFM

**Die-level Visual Foundation Model platform.**

Die VFM is currently an artifact-centric visual embedding platform for die-level image experiments. The current implementation focuses on a token-centric model contract, embedding export, artifact-driven evaluators, and M1 checkpoint/resume behavior for `bootstrap` scope.

## Current Status

The repository currently treats the following as formal, testable capabilities:

- Token-centric model pipeline: `Dataset -> Model -> Embedding Artifact -> Evaluator`
- Artifact-driven evaluators: `linear_probe`, `knn`, `centroid`, `retrieval`; `centroid` is currently supported as a standalone evaluator, while `round1_frozen` currently orchestrates `linear_probe`, `knn`, and `retrieval`
- Runtime modes: `bootstrap`, `round1_frozen`
- Current supported backbones: `dummy`, `dinov2` (current promotion scope:
  `bootstrap` and `round1_frozen`)
- Current supported poolers: `mean`, `identity`, `attn_pooler_v1`
- Domain dataset adapter v0 ingestion path (`dataset=domain`) is available for
  CSV-manifest-based domain data in `bootstrap` and `round1_frozen`
- Single-shard M1 embedding artifact layout: `manifest.yaml` plus `part-00000.pt`
- `round1_frozen` is a single-shot inference/evaluation flow and does not define epoch/resume continuation semantics
- `latest.pt`, `best.pt`, and `epoch_xxxx.pt` are currently part of the `bootstrap` checkpoint flow, not the formal `round1_frozen` contract.
- bootstrap checkpoint/resume contract currently includes strict payload validation plus explicit-path-first resume resolution

The repository does not yet treat `round2_ssl`, `round3_supcon`, or training-stage full-resume guarantees as current production scope.

## Documentation

- [Current Spec](docs/current-spec.md)
- [Future Spec](docs/future-spec.md)
- [Testing Spec](docs/testing-spec.md)
- [Checkpoint / Resume Spec](docs/checkpoint-resume-spec.md)
- [Domain Adapter Spec](docs/domain-adapter-spec.md)
- [Domain Baseline Closeout](docs/domain-baseline-closeout.md)
- [Implementation Roadmap](docs/implementation-roadmap.md)

Document roles:

- `README.md`: project entrypoint, current status, and navigation
- `docs/current-spec.md`: current formal implementation and testing contract
- `docs/future-spec.md`: future-facing roadmap and non-current design direction
- `docs/testing-spec.md`: verification scope for current formal behavior
- `docs/checkpoint-resume-spec.md`: current checkpoint and resume contract
- `docs/domain-adapter-spec.md`: domain dataset ingestion contract and v0
  implementation boundary
- `docs/domain-baseline-closeout.md`: finalized baseline findings for the
  current domain round1 execution
- `docs/implementation-roadmap.md`: execution order for implementation work

If documentation drifts, current implementation work should follow runtime code, `configs/`, and `docs/current-spec.md` before future-facing design notes.

## Quick Start

### Install

```bash
pip install -e .[dev]
```

### Install review tool extras

```bash
pip install -e .[review]
```

Use this extra when you want to launch the local pair review web tool for
`pair_candidates.csv` / `annotations.csv` workflows.

### Domain baseline onboarding quickstart (1-page)

Use this section when you want the shortest path to reproduce the current
`dinov2` domain baseline flow end-to-end.

Full contract, IO schema, and detailed variants:
- [Domain Baseline Playbook](docs/domain-baseline-playbook.md)

1. Build pair candidates from sampled pool
```bash
python -m scripts.generate_pair_candidates \
  --input /abs/path/to/sample_pool.csv \
  --output /abs/path/to/pair_candidates.csv \
  --summary-output /abs/path/to/pair_candidates_summary.csv
```

2. Review and label pairs
```bash
python -m scripts.run_pair_review \
  --pairs /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv \
  --host 0.0.0.0 \
  --port 8000
```

3. Build Round1 domain manifest and dataset config
```bash
python -m scripts.build_round1_domain_assets \
  --train-csv /abs/path/to/round1_pilot_train.csv \
  --val-csv /abs/path/to/round1_pilot_val.csv \
  --manifest-output /abs/path/to/round1_domain_manifest.csv \
  --dataset-config-output /abs/path/to/configs/dataset/domain_round1_pilot.yaml \
  --require-non-empty-val
```

4. Run Round1 baseline (`dinov2`, offline local assets)
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

5. Build pair benchmark manifest from reviewed pairs
```bash
python -m scripts.build_pair_benchmark_manifest \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv \
  --output /abs/path/to/pair_benchmark_manifest.csv \
  --relations same,different,uncertain \
  --label-mode empty \
  --image-path-mode directory
```

This step selects only reviewed pairs and writes an inference-only manifest
(`Source=Infer`) for pair benchmark embedding export. Required columns:
`pair_id,did_a,did_b,image_id_a,image_id_b,image_path_a,image_path_b` in
`pair_candidates.csv`, and `pair_id,review_status,visual_relation` in
`annotations.csv`. Use `--label-mode empty` for export-only flow; switch to
`--label-mode fine_label` only when you intentionally need a labeled manifest.

6. Export pair benchmark embeddings (inference-only)
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

7. Evaluate pair benchmark and run slicing analysis
```bash
python -m scripts.evaluate_pair_benchmark \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv \
  --embedding-split-dir /abs/path/to/runs/pair_benchmark_dinov2_export_v1/round1/embeddings/val \
  --join-key did \
  --output-dir /abs/path/to/pair_benchmark_eval_v1 \
  --hard-limit 50

python -m scripts.analyze_pair_benchmark_slices \
  --pair-scores /abs/path/to/pair_benchmark_eval_v1/pair_scores.csv \
  --pair-candidates /abs/path/to/pair_candidates.csv \
  --output-dir /abs/path/to/pair_slice_analysis_high \
  --confidence high \
  --hard-limit 20
```

### Run bootstrap smoke test

```bash
python scripts/run.py \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=dummy
```

This path exercises the default `bootstrap` flow and writes run artifacts, smoke metadata, and checkpoints under `runs/default/`.

### Run dinov2 bootstrap smoke with offline local assets

```bash
python scripts/run.py \
  run.run_name=my_dinov2_offline_bootstrap \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dinov2 \
  model.backbone.pretrained=false \
  model.backbone.allow_network=false \
  model.backbone.local_repo_path=/abs/path/to/local/dinov2_repo \
  model/pooler=mean \
  dataset=dummy
```

This path forces local architecture loading (`source=local`) and rejects
network fallback. If you set `model.backbone.pretrained=true`, provide
`model.backbone.local_checkpoint_path=/abs/path/to/checkpoint.pt` for offline
runs.

### Run Round1 frozen flow

```bash
python scripts/run.py \
  experiment=round1_frozen \
  run.run_name=my_round1_run \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=attn_pooler_v1 \
  dataset=dummy
```

This path exports train and val embeddings, runs the current Round1 evaluator
set from the experiment config (`linear_probe`, `knn`, `retrieval`), and writes
single-run outputs under `runs/my_round1_run/`.

### Run Round1 frozen with dinov2 offline local assets

```bash
python scripts/run.py \
  experiment=round1_frozen \
  run.run_name=my_round1_dinov2_offline \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dinov2 \
  model/pooler=mean \
  model.backbone.pretrained=false \
  model.backbone.freeze=true \
  model.backbone.allow_network=false \
  model.backbone.local_repo_path=/abs/path/to/local/dinov2_repo \
  train.freeze_backbone=true \
  train.freeze_pooler=true
```

This path keeps Round1 in current frozen orchestration semantics while forcing
offline local architecture loading for `dinov2`. If you set
`model.backbone.pretrained=true`, also set
`model.backbone.local_checkpoint_path=/abs/path/to/checkpoint.pt`.

#### Offline failure example (missing local repo)

If `allow_network=false` and `model.backbone.local_repo_path` is not set,
Round1 fails fast before export/evaluation:

```bash
python scripts/run.py \
  experiment=round1_frozen \
  run.run_name=my_round1_dinov2_offline_fail \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dinov2 \
  model/pooler=mean \
  model.backbone.pretrained=false \
  model.backbone.freeze=true \
  model.backbone.allow_network=false \
  train.freeze_backbone=true \
  train.freeze_pooler=true
```

Expected failure wording includes:
`DINOv2 architecture source is unavailable: set model.backbone.local_repo_path or enable model.backbone.allow_network=true.`

#### Offline failure example (missing local checkpoint)

If `model.backbone.pretrained=true` and `allow_network=false`, Round1 also
requires `model.backbone.local_checkpoint_path`:

```bash
python scripts/run.py \
  experiment=round1_frozen \
  run.run_name=my_round1_dinov2_offline_missing_checkpoint \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dinov2 \
  model/pooler=mean \
  model.backbone.pretrained=true \
  model.backbone.freeze=true \
  model.backbone.allow_network=false \
  model.backbone.local_repo_path=/abs/path/to/local/dinov2_repo \
  train.freeze_backbone=true \
  train.freeze_pooler=true
```

Expected failure wording includes:
`DINOv2 pretrained offline load requires model.backbone.local_checkpoint_path when model.backbone.allow_network=false.`

### Run domain dataset bootstrap quickstart

```bash
python scripts/run.py \
  run.run_name=my_domain_bootstrap \
  train.mode=bootstrap \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=domain \
  dataset.manifest_path=/abs/path/to/domain_manifest.csv \
  +dataset.label_map.ok=1
```

Use this when you want to validate domain CSV ingestion and bootstrap smoke
artifacts. `PATH` values inside the manifest must be absolute directories.

### Run domain inference-only export flow (val required)

```bash
python scripts/run.py \
  experiment=domain_inference_export \
  run.run_name=my_domain_infer_export \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=domain \
  dataset.manifest_path=/abs/path/to/domain_manifest.csv \
  dataset.require_non_empty_val=true
```

This preset keeps Round1 in export-only mode. Setting
`dataset.require_non_empty_val=true` enforces inference-only `val` guarding, so
the run fails fast when the filtered `val` split is empty.

#### Empty-val failure example

If your manifest only contains `Source=Train` rows (no `Source=Infer` rows),
the inference-only guard will fail fast:

```bash
python scripts/run.py \
  experiment=domain_inference_export \
  run.run_name=my_domain_infer_export_fail \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=domain \
  dataset.manifest_path=/abs/path/to/train_only_manifest.csv \
  dataset.require_non_empty_val=true \
  +dataset.label_map.ok=1
```

Expected failure wording includes:
`Filtered val split is empty under inference-only policy.`

#### Mixed-label val failure example

If your `Source=Infer` subset mixes labeled and unlabeled rows, current artifact
contract validation fails fast:

```bash
python scripts/run.py \
  experiment=domain_inference_export \
  run.run_name=my_domain_infer_export_mixed_val_fail \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=domain \
  dataset.manifest_path=/abs/path/to/mixed_val_manifest.csv \
  dataset.require_non_empty_val=true \
  +dataset.label_map.ok=1
```

Expected failure wording includes:
`Filtered val split must not mix labeled and unlabeled samples under current artifact contract.`

### Launch pair review tool

```bash
python scripts/run_pair_review.py \
  --pairs /abs/path/to/pair_candidates.csv \
  --annotations /abs/path/to/annotations.csv
```

This launches a local FastAPI app on `http://127.0.0.1:8000/` for reviewing
pair candidates, labeling `same` / `different` / `uncertain`, and writing
results back to `annotations.csv`.

### Export embeddings

```bash
python scripts/export_embeddings.py \
  run.run_name=my_run \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=dummy
```

This writes embedding artifacts to `runs/my_run/embeddings/<split>/`.

### Run evaluation

```bash
python scripts/run_linear_probe.py \
  evaluation.linear_probe.enabled=true \
  evaluation.linear_probe.input.train_split_dir=runs/my_run/embeddings/train \
  evaluation.linear_probe.input.val_split_dir=runs/my_run/embeddings/val \
  evaluation.linear_probe.output.output_dir=runs/my_run/eval/linear_probe

python scripts/run_knn.py \
  evaluation.knn.input.train_split_dir=runs/my_run/embeddings/train \
  evaluation.knn.input.val_split_dir=runs/my_run/embeddings/val \
  evaluation.knn.output.output_dir=runs/my_run/eval/knn
```

Evaluator outputs are written as filesystem artifacts such as `metrics.yaml`, `summary.yaml`, `config.yaml`, and optional prediction files.

## Repository Layout

```text
die-vfm/
├── configs/                     # Hydra configs
├── die_vfm/                     # Runtime package
│   ├── artifacts/               # Embedding artifact schemas and I/O
│   ├── datasets/                # Dataset adapters and dataloader builder
│   ├── evaluator/               # Artifact-driven evaluators
│   ├── models/                  # Backbone, pooler, top-level model
│   └── trainer/                 # Bootstrap checkpoint helpers and Round1 runner
├── docs/                        # Specs and engineering guidance
├── scripts/                     # CLI entrypoints
└── tests/                       # Test suite
```

## Design Principles

- `patch_tokens` are the primary model representation.
- Evaluators consume saved artifacts, not dataloaders or model objects.
- Current scope is intentionally narrow and favors stable contracts over breadth.
- `configs/` plus runtime code are the current source of truth when documentation and older schema drift.

## Notes

- `dummy` and `dinov2` are currently supported backbones.
- current `dinov2` promotion scope is limited to `bootstrap` and
  `round1_frozen`; Round2+ training semantics remain future scope.
- `bootstrap` is the implicit default when `train.mode` is not set by the selected config composition.
- Round1 is currently positioned as a single-shot inference/evaluation runner; training-stage resume semantics are future Round2+ scope.
