# Die VFM

**Die-level Visual Foundation Model platform.**

Die VFM is currently an artifact-centric visual embedding platform for die-level image experiments. The current implementation focuses on a token-centric model contract, embedding export, artifact-driven evaluators, and M1 checkpoint/resume behavior for `bootstrap` and `round1_frozen`.

## Current Status

The repository currently treats the following as formal, testable capabilities:

- Token-centric model pipeline: `Dataset -> Model -> Embedding Artifact -> Evaluator`
- Artifact-driven evaluators: `linear_probe`, `knn`, `centroid`, `retrieval`; `centroid` is currently supported as a standalone evaluator, while `round1_frozen` currently orchestrates `linear_probe`, `knn`, and `retrieval`
- Runtime modes: `bootstrap`, `round1_frozen`
- Current supported backbone: `dummy`
- Current supported poolers: `mean`, `identity`, `attn_pooler_v1`
- Single-shard M1 embedding artifact layout: `manifest.yaml` plus `part-00000.pt`
- M1 checkpoint set: `latest.pt`, `best.pt` when selected, `epoch_xxxx.pt`

The repository does not yet treat `round2_ssl`, `round3_supcon`, or a fully featured `full_resume` contract as current production scope.

## Documentation

- [Current Spec](docs/current-spec.md)
- [Future Spec](docs/future-spec.md)
- [Testing Spec](docs/testing-spec.md)
- [Checkpoint / Resume Spec](docs/checkpoint-resume-spec.md)
- [Implementation Roadmap](docs/implementation-roadmap.md)

Document roles:

- `README.md`: project entrypoint, current status, and navigation
- `docs/current-spec.md`: current formal implementation and testing contract
- `docs/future-spec.md`: future-facing roadmap and non-current design direction
- `docs/testing-spec.md`: verification scope for current formal behavior
- `docs/checkpoint-resume-spec.md`: current checkpoint and resume contract
- `docs/implementation-roadmap.md`: execution order for implementation work

If documentation drifts, current implementation work should follow runtime code, `configs/`, and `docs/current-spec.md` before future-facing design notes.

## Quick Start

### Install

```bash
pip install -e .[dev]
```

### Run bootstrap smoke test

```bash
python scripts/train.py \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=dummy
```

This path exercises the default `bootstrap` flow and writes run artifacts, smoke metadata, and checkpoints under `runs/default/`.

### Run Round1 frozen flow

```bash
python scripts/train.py \
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
outputs under `runs/my_round1_run/`.

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
│   └── trainer/                 # Bootstrap and Round1 orchestration
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

- `dummy` is the only formally supported backbone in the current spec.
- `dinov2` exists in the codebase but is not yet fully wired as a current formal capability.
- `bootstrap` is the implicit default when `train.mode` is not set by the selected config composition.
- The checkpoint system already supports `warm_start` and M1-style `full_resume`, but future rounds require a richer resume contract.
