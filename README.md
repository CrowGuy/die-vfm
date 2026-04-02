# Die VFM

**Die-level Visual Foundation Model platform.**

A modular ML infrastructure for building and evaluating die-level visual foundation models using a token-centric, pooler-based representation pipeline.

---

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Pipeline](#pipeline)
5. [Backbones](#backbones)
6. [Poolers](#poolers)
7. [Training](#training)
8. [Embedding Artifacts](#embedding-artifacts)
9. [Evaluators](#evaluators)
10. [Checkpoint & Resume](#checkpoint--resume)
11. [Configuration Reference](#configuration-reference)
12. [Testing](#testing)
13. [Repository Structure](#repository-structure)
14. [Design Principles](#design-principles)

---

## Overview

Die VFM uses a fixed two-stage pipeline:

```
Dataset → Model (Backbone + Pooler) → Embedding Artifact → Evaluator
```

Key design decisions:
- **Token-centric**: backbone outputs `patch_tokens [B, N, D]`, not logits
- **Artifact-driven evaluation**: evaluators consume saved embeddings, not dataloaders
- **Hydra config**: all behavior is composition-based and override-friendly
- **Checkpoint-safe**: stable schema across training rounds

---

## Quick Start

### Install

```bash
pip install -e .[dev]
```

### Run a smoke test (CPU, no data required)

```bash
python scripts/train.py \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=dummy
```

### Run Round1 frozen backbone experiment

```bash
python scripts/train.py \
  experiment=round1_frozen \
  model/backbone=dinov2 \
  model/pooler=attn_pooler_v1
```

### Export embeddings

```bash
python scripts/export_embeddings.py \
  run.run_name=my_run \
  model/backbone=dinov2 \
  model/pooler=attn_pooler_v1
```

### Run evaluation

```bash
# Linear probe
python scripts/run_linear_probe.py \
  evaluation.linear_probe.input.train_split_dir=runs/my_run/embeddings/train \
  evaluation.linear_probe.input.val_split_dir=runs/my_run/embeddings/val \
  evaluation.linear_probe.output.output_dir=runs/my_run/eval/linear_probe

# kNN
python scripts/run_knn.py \
  evaluation.knn.input.train_split_dir=runs/my_run/embeddings/train \
  evaluation.knn.input.val_split_dir=runs/my_run/embeddings/val \
  evaluation.knn.output.output_dir=runs/my_run/eval/knn
```

### Run tests

```bash
pytest
```

---

## Architecture

### Component Contracts

#### `BackboneOutput`

```python
@dataclass
class BackboneOutput:
    patch_tokens: Tensor    # [B, N, D]
    cls_token:    Tensor | None  # [B, D]
    token_mask:   Tensor | None  # [B, N]
    feature_dim:  int
    patch_grid:   tuple[int, int] | None
    metadata:     dict
```

#### `PoolerOutput`

```python
@dataclass
class PoolerOutput:
    embedding:     Tensor        # [B, D_out]
    token_weights: Tensor | None # [B, N]
    metadata:      dict
```

#### `ModelOutput`

```python
@dataclass
class ModelOutput:
    embedding: Tensor            # [B, D_out]
    backbone:  BackboneOutput | None
    pooler:    PoolerOutput | None
    metadata:  dict
```

#### `DatasetSample` (per item)

```python
{
    "image":    Tensor,      # [C, H, W]
    "label":    int | None,
    "image_id": str,
    "meta":     dict,
}
```

#### Batch (collated)

```python
{
    "image":    Tensor,      # [B, C, H, W]
    "label":    Tensor | None,  # [B]
    "image_id": list[str],
    "meta":     list[dict],
}
```

---

## Pipeline

```
image [B,C,H,W]
    │
    ▼
Backbone ──────────────────────────────► BackboneOutput
  patch_tokens [B, N, D]
  cls_token    [B, D]
    │
    ▼
Pooler ─────────────────────────────────► PoolerOutput
  embedding [B, D_out]
    │
    ▼
EmbeddingExporter ─────────────────────► Artifact (manifest.yaml + part-*.pt)
    │
    ▼
Evaluator (LinearProbe / kNN / Centroid / Retrieval)
    │
    ▼
metrics.yaml / summary.yaml / predictions.pt
```

---

## Backbones

### DINOv2

```yaml
# configs/model/backbone/dinov2.yaml
model:
  backbone:
    name: dinov2
    variant: vit_base      # vit_small | vit_base | vit_large | vit_giant
    pretrained: true
    freeze: true
    return_cls_token: true
```

| Variant     | `feature_dim` |
|-------------|--------------|
| `vit_small` | 384          |
| `vit_base`  | 768          |
| `vit_large` | 1024         |
| `vit_giant` | 1536         |

Output contract:
- `patch_tokens`: `[B, N, D]` — `N = (H/14) × (W/14)` for patch_size=14
- `cls_token`: `[B, D]`
- `token_mask`: always `None` (DINOv2 standard path)

### DummyBackbone

For testing without GPU or DINOv2 weights:

```yaml
model/backbone: dummy
```

---

## Datasets

### CIFAR-10

A real image classification dataset via `torchvision.datasets.CIFAR10`.

**Install dependency** (`pillow` is required for image decoding):
```bash
pip install -e .[dev]   # already included
```

**Config** (`configs/dataset/cifar10.yaml`):
```yaml
name: cifar10
root: ${oc.env:DIE_VFM_DATA_ROOT,./data/cifar10}  # override via env var
image_size: [224, 224]
download: false
normalize:
  mean: [0.485, 0.456, 0.406]
  std: [0.229, 0.224, 0.225]
```

**Usage:**
```bash
# Set data root (or let it default to ./data/cifar10)
export DIE_VFM_DATA_ROOT=/path/to/datasets

python scripts/train.py \
  dataset=cifar10 \
  model/backbone=dinov2 \
  model/pooler=attn_pooler_v1 \
  experiment=round1_frozen
```

**Split convention:**

| `split` arg | CIFAR-10 subset | Samples |
|-------------|----------------|---------|
| `train`     | training set   | 50,000  |
| `val`       | test set       | 10,000  |

> [!NOTE]
> CIFAR-10 has no official validation split. `split="val"` maps to the CIFAR-10 test set (`train=False`).

**Sample contract:**
```python
{
    "image":    Tensor[3, 224, 224],   # resized to image_size, normalized
    "label":    int,                    # 0–9
    "image_id": "cifar10_train_00042", # f"cifar10_{split}_{index:05d}"
    "meta": {
        "split":      "train",
        "index":      42,
        "source":     "cifar10",
        "class_name": "cat",            # human-readable label
        "raw_label":  3,
    },
}
```

**10 classes:** airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

**Dataset metadata** (written to `dataset_metadata.yaml` on each run):
```yaml
dataset_name: cifar10
split: train
num_samples: 50000
num_channels: 3
image_size: [224, 224]
num_classes: 10
class_names: [airplane, automobile, ...]
```

### DummyDataset

For testing without any real data:
```bash
dataset=dummy
```
Generates synthetic random images in memory. No download or external files.

---

## Poolers

### MeanPooler

Averages patch tokens (ignoring mask).

```yaml
model/pooler: mean
```

Best for: stable baseline, debugging.

### AttnPoolerV1

Single-query attention pooler. Learns per-token importance weights.

```
score_i = w_q^T tanh(W_k x_i)
alpha   = softmax(score, dim=N)
embed   = Σ alpha_i * x_i
```

```yaml
model:
  pooler:
    name: attn_pooler_v1
    hidden_dim: 256
    output_dim: null            # null = keep input_dim
    dropout: 0.0
    l2_norm: false
    use_cls_token_as_query: false
    return_token_weights: true
```

Output: `token_weights [B, N]` (sum to 1 per sample).

Best for: token-importance analysis, defect localization.

### IdentityPooler

Returns raw patch tokens without aggregation. For debugging only.

```yaml
model/pooler: identity
```

---

## Training

### Modes

Training mode is selected by `train.mode` (or through an experiment override):

| Mode | Description |
|------|-------------|
| `bootstrap` | Smoke test: reads one batch and runs one forward pass |
| `round1_frozen` | Full Round1 orchestration: frozen backbone → embed → evaluate → checkpoint |

### Round1 Frozen

Runs the complete M1 experiment loop:

```bash
python scripts/train.py \
  experiment=round1_frozen \
  model/backbone=dinov2 \
  model/pooler=attn_pooler_v1 \
  run.run_name=round1_exp1
```

**What happens each epoch:**

1. Build train + val dataloaders
2. Freeze backbone (and optionally pooler)
3. Export `train_embeddings` and `val_embeddings` to disk
4. Run enabled evaluators (linear probe / kNN / retrieval)
5. Track best metric, save checkpoint
6. Write `round1_summary.yaml` + `round1_summary.json`

**Output layout:**

```text
runs/<run_name>/
├── config.yaml                    # frozen config snapshot
├── logs/
│   └── run.log
├── checkpoints/
│   ├── latest.pt
│   ├── best.pt
│   └── epoch_0000.pt
└── round1/
    └── epoch_0000/
        ├── embeddings/
        │   ├── train/
        │   │   ├── manifest.yaml
        │   │   └── part-00000.pt
        │   └── val/
        │       ├── manifest.yaml
        │       └── part-00000.pt
        ├── evaluation/
        │   ├── linear_probe/
        │   ├── knn/
        │   └── retrieval/
        ├── round1_summary.yaml
        └── round1_summary.json
```

**Relevant config knobs:**

```yaml
train:
  mode: round1_frozen          # set automatically by experiment=round1_frozen
  num_epochs: 1
  freeze_backbone: true
  freeze_pooler: true
  selection_metric: linear_probe.val_accuracy

evaluation:
  run_linear_probe: true
  knn:
    enabled: true
  retrieval:
    enabled: false
```

### Bootstrap (smoke test)

```bash
python scripts/train.py \
  system.device=cpu \
  system.num_workers=0 \
  model/backbone=dummy \
  model/pooler=mean \
  dataset=dummy
```

Produces in `runs/default/`:
```text
config.yaml
dataset_metadata.yaml
model_smoke.yaml
logs/run.log
```

---

## Embedding Artifacts

Embeddings are saved as **first-class artifacts** — split-level, self-contained, and evaluator-readable without model code.

### Structure

```text
<split_dir>/
├── manifest.yaml
└── part-00000.pt
```

### `manifest.yaml`

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

### Shard format (`.pt`)

```python
{
    "embeddings": Tensor[N, D],
    "labels":     Tensor[N] | None,
    "image_ids":  list[str],
    "metadata":   list[dict],
}
```

Alignment invariant: `embeddings[i] ↔ labels[i] ↔ image_ids[i] ↔ metadata[i]`

### Export API

```python
from die_vfm.artifacts import export_split_embeddings

manifest = export_split_embeddings(
    model=model,
    dataloader=dataloader,
    output_dir="runs/my_run/embeddings/train",
    split="train",
    device="cuda",
)
```

### Load API

```python
from die_vfm.artifacts import load_embedding_split

artifact = load_embedding_split("runs/my_run/embeddings/train")

artifact.embeddings  # Tensor[N, D]
artifact.labels      # Tensor[N] | None
artifact.image_ids   # list[str]
artifact.metadata    # list[dict]
artifact.manifest    # EmbeddingManifest
```

---

## Evaluators

All evaluators follow the same contract:
- **Input**: embedding artifact directories (train + val)
- **Output**: `metrics.yaml`, `summary.yaml`, `config.yaml`, `predictions.pt`
- **No dataloaders**, **no model code** required

### Evaluator Comparison

| Evaluator    | Training | Complexity | Primary Metric  | Best for                    |
|-------------|----------|------------|-----------------|------------------------------|
| Linear Probe | Yes (linear) | O(N·E) | `val_accuracy`  | Learned classifier baseline |
| kNN          | None     | O(N)       | `top1_accuracy` | Training-free quality check |
| Centroid     | None     | O(C)       | `top1_accuracy` | Fast cluster structure check |
| Retrieval    | None     | O(N·Q)     | `recall_at_1`   | Ranking / retrieval quality |

### Linear Probe

Trains a linear classifier on frozen embeddings.

```bash
python scripts/run_linear_probe.py \
  evaluation.run_linear_probe=true \
  evaluation.linear_probe.input.train_split_dir=runs/<run>/embeddings/train \
  evaluation.linear_probe.input.val_split_dir=runs/<run>/embeddings/val \
  evaluation.linear_probe.output.output_dir=runs/<run>/eval/linear_probe
```

Key config:

```yaml
evaluation:
  linear_probe:
    input:
      normalize_embeddings: false
    model:
      bias: true
    trainer:
      batch_size: 256
      num_epochs: 50
      learning_rate: 0.01
      optimizer_name: sgd      # sgd | adamw
      selection_metric: val_accuracy
    output:
      save_predictions: true
      save_history: true
```

Output `metrics.yaml` includes: `train.accuracy`, `val.accuracy`, `best_epoch`, input metadata.

### kNN Evaluator

Training-free nearest-neighbor classifier.

```bash
python scripts/run_knn.py \
  evaluation.knn.input.train_split_dir=runs/<run>/embeddings/train \
  evaluation.knn.input.val_split_dir=runs/<run>/embeddings/val \
  evaluation.knn.output.output_dir=runs/<run>/eval/knn
```

Key config:

```yaml
evaluation:
  knn:
    enabled: true
    evaluator:
      k: 20
      metric: cosine       # cosine | l2
      weighting: uniform   # uniform | distance
      temperature: 0.07
      batch_size: 1024
      topk: [1, 5]
```

`predictions.pt` includes `neighbor_indices`, `neighbor_labels`, `neighbor_scores` for debugging.

### Centroid Evaluator

Computes one prototype (mean embedding) per class. O(C) at query time.

```bash
python scripts/run_centroid.py \
  evaluation.centroid.enabled=true \
  evaluation.centroid.input.train_split_dir=runs/<run>/embeddings/train \
  evaluation.centroid.input.val_split_dir=runs/<run>/embeddings/val \
  evaluation.centroid.output.output_dir=runs/<run>/eval/centroid
```

| | kNN | Centroid |
|-|-----|---------|
| Reference set | All train samples | Class means (C) |
| Query complexity | O(N) | O(C) |
| Flexibility | High | Low |

### Retrieval Evaluator

Measures embedding ranking quality via Recall@K and mAP@K.
- Gallery = train split; Query = val split
- Relevance is label-based

```bash
python scripts/run_retrieval.py \
  evaluation.retrieval.enabled=true \
  evaluation.retrieval.input.train_split_dir=runs/<run>/embeddings/train \
  evaluation.retrieval.input.val_split_dir=runs/<run>/embeddings/val \
  evaluation.retrieval.output.output_dir=runs/<run>/eval/retrieval
```

Key config:

```yaml
evaluation:
  retrieval:
    enabled: true
    evaluator:
      metric: cosine
      topk: [1, 5, 10]
      batch_size: 1024
      exclude_same_image_id: false
```

Metrics: `recall_at_1`, `recall_at_5`, `recall_at_10`, `map_at_1`, `map_at_5`.

---

## Checkpoint & Resume

### Layout

```text
<run_dir>/checkpoints/
├── latest.pt       # always updated (last epoch)
├── best.pt         # best selection metric
└── epoch_0000.pt   # per-epoch snapshot
```

### Checkpoint Schema

```python
{
    "checkpoint_version":     "v1",
    "epoch":                  int,
    "global_step":            int,
    "model_state_dict":       dict,
    "optimizer_state_dict":   dict | None,
    "lr_scheduler_state_dict":dict | None,
    "grad_scaler_state_dict": dict | None,
    "trainer_state": {
        "epoch":             int,
        "global_step":       int,
        "best_metric_name":  str | None,
        "best_metric_value": float | None,
    },
    "metadata":               dict,
}
```

### Resume Config

```yaml
train:
  resume:
    enabled: false
    mode: full_resume          # full_resume | warm_start
    checkpoint_path: null      # explicit path, or null for auto
    auto_resume_latest: false
```

| Mode | Restores |
|------|---------|
| `warm_start` | model weights only (epoch resets to 0) |
| `full_resume` | model + trainer state (epoch/global_step resume) |

### Examples

**Auto-resume from latest:**

```bash
python scripts/train.py \
  experiment=round1_frozen \
  train.resume.enabled=true \
  train.resume.mode=full_resume \
  train.resume.auto_resume_latest=true
```

**Warm start from a specific checkpoint:**

```bash
python scripts/train.py \
  experiment=round1_frozen \
  train.resume.enabled=true \
  train.resume.mode=warm_start \
  train.resume.checkpoint_path=/path/to/epoch_0000.pt
```

### Python API

```python
from die_vfm.trainer.checkpoint_manager import CheckpointManager

ckpt = CheckpointManager("runs/my_run/checkpoints")

# Save
ckpt.save(model=model, trainer_state=state, epoch=0, global_step=100, is_best=True)

# Warm start
ckpt.load_warm_start(checkpoint_path="runs/my_run/checkpoints/best.pt", model=model)

# Full resume
ckpt.load_full_resume(
    checkpoint_path="runs/my_run/checkpoints/latest.pt",
    model=model,
    trainer_state=state,
)
```

---

## Configuration Reference

### Root config (`configs/config.yaml`)

```yaml
defaults:
  - experiment: debug_model_smoke
  - model/backbone: dummy
  - model/pooler: mean
  - dataset: dummy
  - artifact: embedding
  - evaluation: linear_probe
  - evaluation/knn
  - evaluation/centroid
  - evaluation/retrieval

run:
  output_root: runs
  run_name: null            # null → "default"
  save_config_snapshot: true

system:
  seed: 42
  device: cpu
  num_workers: 4

train:
  mode: bootstrap           # bootstrap | round1_frozen
  num_epochs: 1
  freeze_backbone: true
  freeze_pooler: true
  selection_metric: linear_probe.val_accuracy
  resume:
    enabled: false
    mode: full_resume
    checkpoint_path: null
    auto_resume_latest: false
```

### Experiment presets

| Name | File | Description |
|------|------|-------------|
| `debug_model_smoke` | `experiment/debug_model_smoke.yaml` | Minimal smoke test |
| `round1_frozen` | `experiment/round1_frozen.yaml` | Round1 frozen backbone run |

### Available model configs

| Component | Config key | Options |
|-----------|-----------|---------|
| Backbone  | `model/backbone` | `dummy`, `dinov2` |
| Pooler    | `model/pooler` | `mean`, `identity`, `attn_pooler_v1` |

### Artifact config

```yaml
artifact:
  embedding:
    enabled: true
    output_subdir: embeddings
    export_splits: [train, val]
    include_test_split: false
```

---

## Testing

```bash
# All tests
pytest

# Specific areas
pytest tests/test_attn_pooler_v1.py
pytest tests/test_embedding_artifact.py
pytest tests/test_knn_evaluator.py
pytest tests/test_linear_probe.py
pytest tests/test_train_bootstrap.py
```

Key test files:

| File | What it tests |
|------|--------------|
| `test_config.py` | Hydra config loading |
| `test_dummy_dataset.py` | Dummy dataset sample contract |
| `test_cifar10_dataset.py` | CIFAR-10 adapter contract, split mapping, collation |
| `test_attn_pooler_v1.py` | Attention pooler shapes and weights |
| `test_pooler_builder.py` | Pooler construction from config |
| `test_embedding_artifact.py` | Manifest schema and shard validation |
| `test_embedding_exporter.py` | Export pipeline |
| `test_embedding_loader.py` | Load and alignment checks |
| `test_linear_probe.py` | Linear probe forward + metrics |
| `test_linear_probe_trainer.py` | Full training loop |
| `test_linear_probe_runner.py` | End-to-end runner |
| `test_knn_evaluator.py` | kNN distance + voting |
| `test_knn_runner.py` | End-to-end kNN runner |
| `test_centroid_evaluator.py` | Centroid prototype logic |
| `test_retrieval_evaluator.py` | Recall@K / mAP@K |
| `test_train_bootstrap.py` | Full bootstrap smoke test |

---

## Repository Structure

```text
die-vfm/
├── configs/
│   ├── config.yaml                # root Hydra config
│   ├── artifact/
│   │   └── embedding.yaml
│   ├── dataset/
│   │   ├── dummy.yaml
│   │   └── cifar10.yaml
│   ├── evaluation/
│   │   ├── linear_probe.yaml
│   │   ├── knn.yaml
│   │   ├── centroid.yaml
│   │   └── retrieval.yaml
│   ├── experiment/
│   │   ├── debug_model_smoke.yaml
│   │   └── round1_frozen.yaml
│   └── model/
│       ├── backbone/
│       │   ├── dinov2.yaml
│       │   └── dummy.yaml
│       └── pooler/
│           ├── attn_pooler_v1.yaml
│           ├── identity.yaml
│           └── mean.yaml
│
├── scripts/
│   ├── train.py                   # main entrypoint (bootstrap + round1)
│   ├── export_embeddings.py       # standalone embedding export
│   ├── run_linear_probe.py
│   ├── run_knn.py
│   ├── run_centroid.py
│   └── run_retrieval.py
│
├── die_vfm/
│   ├── models/
│   │   ├── outputs.py             # BackboneOutput, PoolerOutput, ModelOutput
│   │   ├── model.py               # DieVFMModel
│   │   ├── builder.py
│   │   ├── backbone/
│   │   │   ├── base.py
│   │   │   ├── dinov2_backbone.py
│   │   │   ├── dummy_backbone.py
│   │   │   └── builder.py
│   │   └── pooler/
│   │       ├── base.py
│   │       ├── mean_pooler.py
│   │       ├── attn_pooler_v1.py
│   │       ├── identity_pooler.py
│   │       └── builder.py
│   │
│   ├── datasets/
│   │   ├── base.py                # DatasetAdapter, DatasetSample
│   │   ├── dummy_dataset.py
│   │   ├── cifar10_dataset.py     # Cifar10DatasetAdapter
│   │   └── builder.py
│   │
│   ├── artifacts/
│   │   ├── embedding_artifact.py  # EmbeddingManifest, LoadedEmbeddingSplit
│   │   ├── embedding_exporter.py
│   │   └── embedding_loader.py
│   │
│   ├── trainer/
│   │   ├── base_trainer.py        # TrainerState
│   │   ├── checkpoint_manager.py  # CheckpointManager
│   │   └── round1_trainer.py      # Round1FrozenTrainer
│   │
│   ├── evaluator/
│   │   ├── io.py                  # shared artifact loader
│   │   ├── metrics.py
│   │   ├── result_writer.py
│   │   ├── linear_probe.py
│   │   ├── linear_probe_trainer.py
│   │   ├── linear_probe_runner.py
│   │   ├── knn_evaluator.py
│   │   ├── knn_runner.py
│   │   ├── centroid_evaluator.py
│   │   ├── centroid_runner.py
│   │   ├── retrieval_evaluator.py
│   │   └── retrieval_runner.py
│   │
│   ├── config/
│   │   └── schema.py
│   └── utils/
│       └── run_dir.py
│
└── tests/
    ├── models/
    └── test_*.py
```

---

## Design Principles

1. **Token-centric backbone output** — `patch_tokens [B, N, D]` is the primary representation; poolers convert it to a global embedding.

2. **Artifact-driven evaluation** — evaluators never touch dataloaders or model code. This enables reproducible evaluation, fast iteration, and clean component boundaries.

3. **Stable contracts** — `BackboneOutput`, `PoolerOutput`, `DatasetSample`, `EmbeddingManifest`, and checkpoint schema are frozen. Adding a new backbone or pooler does not break existing evaluator code.

4. **Minimal M1 scope** — Round1 uses frozen backbone + pooler. No optimizer tuition, no distributed training, no fancy loss. The loop is: embed → evaluate → checkpoint.

5. **Fail-fast config validation** — Hydra composition + strict dataclass validation means misconfigured runs fail at startup, not mid-epoch.

6. **Testability** — every component has a `Dummy*` counterpart so tests are fast, offline, and deterministic.