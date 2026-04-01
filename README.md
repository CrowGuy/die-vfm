# Die VFM

Die-level visual foundation model platform.

This repository provides a modular ML infrastructure for building
**die-level visual foundation models (VFM)**.

This repository skeleton provides:

- Config composition via Hydra
- Run directory creation
- Logging
- Training entry point bootstrap
- Basic tests

---

## PR-2 Scope

PR-2 introduces the **dataset adapter interface and dataloader pipeline**.

New functionality includes:

- Dataset adapter base interface
- Dummy dataset implementation
- Dataset builder
- Dataloader builder
- Dataset metadata logging and artifact
- Train entrypoint dataloader smoke test
- Dataset unit tests

# PR-3 Scope

PR-3 introduces the **model pipeline** and connects it with the existing
data pipeline.

New functionality includes:

- Model abstraction (`Backbone`, `Pooler`, `DieVFMModel`)
- Model builder (`build_model`)
- Dummy backbone (for testing)
- Mean / Identity poolers
- End-to-end model forward pipeline
- Model forward smoke test
- Model smoke artifact

# Pipeline Overview

## Data Pipeline (PR-2)
```text
config → dataloader → batch
```

## Model Pipeline (PR-3)
```text
batch["image"]
↓
backbone
↓
patch tokens
↓
pooler
↓
embedding
```

## Pooling Strategies (PR-4)

The pooling module aggregates patch/token-level features into a fixed-dimensional embedding.

Currently supported poolers:

### 1. Mean Pooler

Averages valid token features.
```yaml
model:
  pooler:
    name: mean
```
- Simple and stable baseline
- Ignores token importance differences

### 2. Identity Pooler

Returns raw token features without pooling.
```yaml
model:
  pooler:
    name: identity
```
- Mainly for debugging or downstream custom pooling

### 3. AttnPoolerV1

Attention-based pooling that learns to weight tokens.

```yaml
model:
  pooler:
    name: attn_pooler_v1
    hidden_dim: 256
    output_dim: null
    dropout: 0.0
    l2_norm: false
    use_cls_token_as_query: false
```

**Key idea:**

The model learns attention weights over tokens:

`embedding = Σ (attention_i * token_i)`

**Features:**

- Learns token importance dynamically
- Supports masking invalid tokens
- Optionally uses CLS token as attention query
- Returns attention weights (token_weights) for analysis

**When to use:**

- When token-level importance matters (e.g. defect localization)
- When mean pooling is too coarse

## Embedding Artifact (PR-5)
### Overview

PR-5 introduces an embedding artifact system that decouples model inference from downstream evaluation.

Instead of running evaluators directly on dataloaders and models, we now:

`Dataset → Model → Embedding Artifact → Evaluator`

This enables:

- reproducible evaluation
- faster iteration (no repeated forward passes)
- clean separation between model and evaluator

### Artifact Structure

Embedding artifacts are stored per run and per split:
```text
runs/<run_name>/
  embeddings/
    train/
      manifest.yaml
      part-00000.pt
    val/
      manifest.yaml
      part-00000.pt
    test/  # optional
```
Each split is **independent and self-contained**.

### Shard Format (`.pt`)

Each shard file contains:
```python
{
    "embeddings": Tensor[N, D],
    "labels": Tensor[N] | None,
    "image_ids": list[str],
    "metadata": list[dict],
}
```
**Contract**
- `embeddings[i] ↔ image_ids[i] ↔ metadata[i]`
- `labels[i]` (if present) must align with `embeddings[i]`
- `image_ids` must be unique within a split

### Manifest Format

Each split contains a `manifest.yaml`:
```yaml
artifact_type: embedding_split
artifact_version: v1
format: torch_pt

split: train
num_samples: 12345
embedding_dim: 768
dtype: float32
has_labels: true
num_shards: 1

shards:
  - file_name: part-00000.pt
    num_samples: 12345
```
**Notes**
- Manifest is strictly validated
- Sum of shard samples must equal num_samples
- Designed for future multi-shard support

### Export API
```python
from die_vfm.artifacts import export_split_embeddings

manifest = export_split_embeddings(
    model=model,
    dataloader=dataloader,
    output_dir=split_dir,
    split="train",
    device="cpu",
)
```
### Load API
```python
from die_vfm.artifacts import load_embedding_split

artifact = load_embedding_split(split_dir)

artifact.embeddings   # Tensor[N, D]
artifact.labels       # Tensor[N] | None
artifact.image_ids    # list[str]
artifact.metadata     # list[dict]
artifact.manifest
```

### Script Usage

You can export embeddings using:
```bash
python scripts/export_embeddings.py \
  run.run_name=my_run \
  dataset=dummy \
  model/backbone=dummy \
  model/pooler=mean
```
Output will be written to:
```bash
runs/my_run/embeddings/
```

### Config

Embedding export is controlled by:
```yaml
artifact:
  embedding:
    enabled: true
    output_subdir: embeddings
    export_splits: [train, val]
    include_test_split: false
```

### Design Constraints

This artifact system enforces:

- Evaluators MUST consume artifacts (not dataloaders)
- Artifacts must be split-based
- Artifacts must be loadable without model code
- Alignment between embeddings and metadata must be preserved

---
## Linear Probe Evaluator (PR-6)

This PR introduces an **artifact-driven linear probe evaluator** that trains a linear classifier on embedding artifacts and evaluates on validation embeddings.

### Key Principles

- Evaluators **consume embedding artifacts only**
- Evaluators **do not depend on dataloaders or models**
- Fully **decoupled from training pipeline**
- **Reproducible** and **configurable via Hydra**

---

## Pipeline
```text
Embedding Artifacts (PR-5)
├── train/
└── val/
↓
Linear Probe Evaluator (PR-6)
↓
Evaluation Outputs
```
---

## Input: Embedding Artifacts

Expected directory structure:
```text
runs/<run_name>/embeddings/
├── train/
│ ├── manifest.yaml
│ └── part-00000.pt
└── val/
├── manifest.yaml
└── part-00000.pt
```

Each shard (`.pt`) contains:

```python
{
  "embeddings": Tensor[N, D],
  "labels": Tensor[N],
  "image_ids": list[str],
  "metadata": list[dict],
}
```
---
Output: Evaluation Artifacts

The evaluator writes results to:

```text
<output_dir>/
  ├── metrics.yaml
  ├── summary.yaml
  ├── config.yaml
  ├── history.yaml          # optional
  └── predictions.pt        # optional
```

### `metrics.yaml`
```yaml
evaluator_type: linear_probe
evaluator_version: v1

input:
  train_split: train
  val_split: val
  train_num_samples: ...
  val_num_samples: ...
  embedding_dim: ...
  num_classes: ...
  class_ids: [...]

best_epoch: ...

train:
  loss: ...
  accuracy: ...

val:
  loss: ...
  accuracy: ...
```

### `predictions.pt`
```python
{
  "split": "val",
  "image_ids": [...],
  "labels": Tensor[N],
  "pred_labels": Tensor[N],
  "logits": Tensor[N, C],
  "class_ids": Tensor[C],
}
```

### Usage
**Run via CLI**
```bash
python scripts/run_linear_probe.py \
  evaluation.run_linear_probe=true \
  evaluation.linear_probe.input.train_split_dir=runs/<run_name>/embeddings/train \
  evaluation.linear_probe.input.val_split_dir=runs/<run_name>/embeddings/val \
  evaluation.linear_probe.output.output_dir=runs/<run_name>/evaluations/linear_probe
```
**Common overrides**
```bash
evaluation.linear_probe.trainer.optimizer_name=adamw
evaluation.linear_probe.trainer.learning_rate=0.05
evaluation.linear_probe.trainer.batch_size=256
evaluation.linear_probe.trainer.num_epochs=50
```

### Config Structure
```text
evaluation.linear_probe
  ├── input
  ├── output
  ├── model
  └── trainer
```
**Example**
```yaml
evaluation:
  linear_probe:
    input:
      train_split_dir: ...
      val_split_dir: ...
      normalize_embeddings: false

    output:
      output_dir: ...
      save_predictions: true
      save_history: true

    model:
      bias: true

    trainer:
      batch_size: 256
      num_epochs: 50
      learning_rate: 0.01
      optimizer_name: sgd
      selection_metric: val_accuracy
```
---
### API
**Run programmatically**
```python
from die_vfm.evaluator import run_linear_probe, build_linear_probe_run_config

config = build_linear_probe_run_config(
    train_split_dir="...",
    val_split_dir="...",
    output_dir="...",
)

result = run_linear_probe(config)

print(result.val_metrics)
```
---
### Design Notes
- No DataLoader usage
    - batching is implemented via tensor slicing
- Strict artifact alignment
    - embeddings ↔ labels ↔ image_ids
- Model-free evaluation
    - artifacts are sufficient for evaluation
- Minimal M1 implementation
    - single-shard only
    - single-device only
---

## Repository Structure
```text
die_vfm/
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── config.yaml
│   ├── dataset/
│   │   └── dummy.yaml
│   ├── experiment/
│   │   └── round1_frozen.yaml
│   ├── model/
│   │   ├── backbone/
│   │   │   └── dinov2.yaml
│   │   └── pooler/
│   │       ├── attn_pooler_v1.yaml
│   │       └── mean_pooler.yaml
├── scripts/
│   └── train.py
├── die_vfm/
│   ├── datasets/
│   │   ├── base.py
│   │   ├── builder.py
│   │   └── dummy_dataset.py
│   ├── utils/
│   ├── trainer/
│   ├── models/
│   ├── evaluator/
│   └── artifacts/
└── tests/
    ├── test_config.py
    ├── test_dummy_dataset.py
    ├── test_train_bootstrap.py
```

## Dataset Sample Contract

Each dataset adapter must return samples with the following structure:

```python
{
    "image": Tensor,        # shape: [C, H, W]
    "label": int | None,
    "image_id": str,
    "meta": dict
}
```
Example:
```python
{
    "image": tensor(3,224,224),
    "label": 0,
    "image_id": "train_00000",
    "meta": {
        "split": "train",
        "index": 0,
        "source": "dummy"
    }
}
```
---
## Batch Contract

The dataloader collates dataset samples into batches:

```python
{
    "image": Tensor,        # shape: [B, C, H, W]
    "label": Tensor | None, # shape: [B]
    "image_id": list[str],
    "meta": list[dict],
}
```
---
## Model Contract (PR-3)
### Input
```python
image: Tensor[B, C, H, W]
```
### Output
```python
ModelOutput:
{
    "embedding": Tensor[B, D],
    "backbone": BackboneOutput | None,
    "pooler": PoolerOutput | None,
}
```

## Smoke Test

### 1. Dataloader + Model Smoke Test
```bash
python scripts/train.py \
    system.num_workers=0 \
    system.device=cpu \
    model/backbone=dummy \
    model/pooler=mean \
    dataset=dummy \
    train.run_dataloader_smoke_test=true \
    train.run_model_forward_smoke_test=true
```
Expected logs:
```text
Dataloader smoke test passed.
Batch image shape: (4, 3, 224, 224)

Built model: DieVFMModel
Backbone: DummyBackbone
Pooler: MeanPooler

Model forward completed.
Embedding shape: (4, 192)

Training bootstrap completed successfully.
```
---
## Run Artifacts

During training bootstrap the following artifacts are generated:
```text
runs/<run_name>/
├── config.yaml
├── dataset_metadata.yaml
├── model_smoke.yaml
└── logs/
    └── run.log
```
`dataset_metadata.yaml` contains dataset-level information such as:
- dataset_name
- split
- num_samples
- num_classes

`model_smoke.yaml` contains model-level information such as:
- model name
- backbone / pooler
- embedding shape
- patch token shape
---

## Quick Start

### Install

```bash
pip install -e .[dev]
```
### Run
```bash
python scripts/train.py
```

### Run with overrides
```bash
python scripts/train.py run.run_name=debug
```

### Run smoke test

```bash
python scripts/train.py \
    system.device=cpu \
    system.num_workers=0
```
---
## Test
Run all tests:
```bash
pytest
```
- `test_dummy_dataset.py` — dataset contract
- `test_model_builder.py` — model construction
- `test_dummy_backbone.py` — backbone contract
- `test_mean_pooler.py` — pooler contract
- `test_model_forward_smoke.py` — model pipeline
- `test_train_bootstrap.py` — end-to-end bootstrap
---
## Design Principles
- Clear contract between components:
    - Dataset → Batch → Model → Embedding
- Minimal abstraction per PR
- Fail-fast config validation
- Testability first (dummy components)
- Separation of:
    - data pipeline
    - model pipeline
    - training logic (future work)

---
## `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.so

# Packaging
*.egg-info/
build/
dist/

# Virtual env
.venv/
venv/

# Testing
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage

# IDE
.vscode/
.idea/

# Hydra / outputs
outputs/
multirun/

# Project runs
runs/

# Logs
*.log
```