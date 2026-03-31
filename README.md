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

### 3. AttnPoolerV1 (NEW)

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