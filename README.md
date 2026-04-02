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
## End-to-End Evaluation Workflow

All evaluators (linear probe, kNN, centroid) follow a **two-stage pipeline**:

```text
Dataset → Model → Embedding Artifact → Evaluator
```

### Step 1: Export embeddings (PR-5)
```bash
python scripts/export_embeddings.py \
  run.run_name=demo \
  dataset=dummy \
  model/backbone=dummy \
  model/pooler=mean
```
---
This produces:
```text
runs/demo/embeddings/
  train/
    manifest.yaml
    part-00000.pt
  val/
    manifest.yaml
    part-00000.pt
```
---
### Step 2: Run evaluator (PR-6 / PR-7 / PR-8)

Example (centroid):

```bash
python scripts/run_centroid.py \
  evaluation.centroid.enabled=true \
  evaluation.centroid.input.train_split_dir=runs/demo/embeddings/train \
  evaluation.centroid.input.val_split_dir=runs/demo/embeddings/val \
  evaluation.centroid.output.output_dir=runs/demo/eval/centroid
```
---

All evaluators operate on embedding artifacts defined in PR-5.

The artifact format is:

```text
<split>/
  manifest.yaml
  part-00000.pt
```
Each shard contains:

{
  "embeddings": Tensor[N, D],
  "labels": Tensor[N] | None,
  "image_ids": list[str],
  "metadata": list[dict],
}

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

## kNN Evaluator (PR-7)

The kNN evaluator provides a training-free baseline to assess embedding quality using nearest neighbor classification.

- `Reference set`: train split embeddings
- `Query set`: val split embeddings
- `Input`: embedding artifacts (no dataloaders)
- `Output`: metrics + predictions + neighbor metadata

This evaluator is fully artifact-driven and follows the same contract as the linear probe evaluator.

### Pipeline
```text
Embedding Artifacts (train / val)
        ↓
kNN Evaluator
        ↓
Predictions + Metrics
```
---
### Key Properties
- No training required
- Deterministic evaluation
- Supports cosine and L2 distance
- Supports uniform and distance-weighted voting
- Fully Hydra-configurable
- Compatible with existing artifact format

---
### Usage
**1. Prepare embedding artifacts**

Make sure you already exported embeddings:
```bash
python scripts/export_embeddings.py ...
```
You should have:
```text
runs/<run_name>/embeddings/
  train/
    manifest.yaml
    part-00000.pt
  val/
    manifest.yaml
    part-00000.pt
```
Each split follows the embedding artifact contract defined in PR-5.

---
**2. Run kNN evaluation**
```bash
python scripts/run_knn.py \
  evaluation.knn.input.train_split_dir=runs/<run_name>/embeddings/train \
  evaluation.knn.input.val_split_dir=runs/<run_name>/embeddings/val \
  evaluation.knn.output.output_dir=runs/<run_name>/evaluations/knn
```
---
### Outputs

The evaluator writes the following artifacts:

**`metrics.yaml`**
```yaml
evaluator_type: knn
evaluator_version: v1

input:
  train_split: train
  val_split: val
  train_num_samples: 10240
  val_num_samples: 2048
  embedding_dim: 384
  num_classes: 12
  class_ids: [0, 1, ...]

val:
  accuracy: 0.91
  top1_accuracy: 0.91
  top5_accuracy: 0.98
```
---

**`summary.yaml`**
```yaml
status: success
evaluator: knn
train_split: train
val_split: val
val_accuracy: 0.91
output_dir: outputs/knn_eval
```
---
**`config.yaml`**

Resolved run configuration used for this evaluation.

---

**`predictions.pt`**
```python
{
  "split": "val",
  "image_ids": [...],
  "labels": Tensor[N],
  "pred_labels": Tensor[N],
  "logits": Tensor[N, C],
  "class_ids": Tensor[C],

  # kNN-specific
  "neighbor_indices": Tensor[N, K],
  "neighbor_labels": Tensor[N, K],
  "neighbor_scores": Tensor[N, K],
}
```
---

### Configuration Reference
`Input`
| Field                  | Description                        |
| ---------------------- | ---------------------------------- |
| `train_split_dir`      | Path to train embedding artifacts  |
| `val_split_dir`        | Path to val embedding artifacts    |
| `normalize_embeddings` | Whether to L2 normalize embeddings |
| `map_location`         | torch.load device                  |

---

`Output`
| Field              | Description                       |
| ------------------ | --------------------------------- |
| `output_dir`       | Directory to write outputs        |
| `save_predictions` | Whether to write `predictions.pt` |

---

`evaluator`
| Field         | Description                 |
| ------------- | --------------------------- |
| `k`           | Number of nearest neighbors |
| `metric`      | `cosine` or `l2`            |
| `weighting`   | `uniform` or `distance`     |
| `temperature` | Used for distance weighting |
| `batch_size`  | Query batch size            |
| `device`      | cpu / cuda                  |
| `topk`        | Top-k metrics               |

---
### Notes
- k must be ≤ number of train samples
- topk must be ≤ number of classes
- distance weighting uses softmax over neighbor scores
- For cosine similarity, embeddings are normalized internally
---
### Comparison with Linear Probe
| Aspect    | Linear Probe        | kNN                        |
| --------- | ------------------- | -------------------------- |
| Training  | Required            | Not required               |
| Speed     | Slower              | Faster                     |
| Stability | Depends on training | Deterministic              |
| Use case  | Learned classifier  | Embedding quality baseline |

---
### When to use kNN
Use kNN when you want:
- Quick sanity check of embedding quality
- Training-free evaluation
- Baseline comparison with learned classifiers
- Debugging embedding space structure

---
## Centroid / Prototype Classifier Evaluator (PR-8)

### Overview

This PR introduces a centroid-based (prototype) classifier evaluator for embedding quality assessment.

- Train split → used to build class prototypes (centroids)
- Val split → used as query set
- Fully artifact-driven (no dataloaders)
- Compatible with existing evaluator framework

Pipeline:
```text
train embeddings ──► class prototypes [C, D]
val embeddings   ──► similarity to prototypes ──► predictions
```
---
### Key Properties
- One prototype per class (mean embedding)
- Supports:
    - cosine similarity
    - L2 distance
- Fast and memory-efficient baseline
- Deterministic (no training / no randomness)
- Output format aligned with:
    - Linear Probe (PR-6)
    - kNN (PR-7)
---
### Usage
**1. Prepare embedding artifacts**

You should already have embedding artifacts from the pipeline:
```text
outputs/<run_name>/embeddings/
  train/
    manifest.yaml
    part-00000.pt
  val/
    manifest.yaml
    part-00000.pt
```
Generate embeddings if not already done:
```bash
python scripts/export_embeddings.py \
  run.run_name=demo \
  dataset=dummy \
  model/backbone=dummy \
  model/pooler=mean
```
---

**2. Run centroid evaluation**
Centroid evaluation is controlled by:

```bash
evaluation.centroid.enabled=true
```

```bash
python scripts/run_centroid.py \
  evaluation.centroid.enabled=true \
  evaluation.centroid.input.train_split_dir=runs/demo/embeddings/train \
  evaluation.centroid.input.val_split_dir=runs/demo/embeddings/val \
  evaluation.centroid.output.output_dir=runs/demo/eval/centroid
```
---
**Optional overrides**
```bash
evaluation.centroid.evaluator.metric=l2
evaluation.centroid.evaluator.batch_size=512
evaluation.centroid.evaluator.topk=[1]
```
---
**3. Example config**
```yaml
evaluation:
  centroid:
    input:
      train_split_dir: runs/<run_name>/embeddings/train
      val_split_dir: runs/<run_name>/embeddings/val
      normalize_embeddings: false
      map_location: cpu

    output:
      output_dir: runs/<run_name>/eval/centroid
      save_predictions: true

    evaluator:
      metric: cosine
      batch_size: 1024
      device: cpu
      topk: [1, 5]
```
---
### Output Artifacts

After running, the following files will be generated:
```text
output_dir/
  metrics.yaml
  summary.yaml
  config.yaml
  predictions.pt
```
---
**metrics.yaml**
Stable metric output:
```yaml
evaluator_type: centroid
evaluator_version: v1

input:
  train_split: train
  val_split: val
  train_num_samples: 50000
  val_num_samples: 10000
  embedding_dim: 768
  num_classes: 10

prototype:
  num_prototypes: 10
  prototype_dim: 768

val:
  accuracy: 0.85
  top1_accuracy: 0.85
  top5_accuracy: 0.98
```
---
**summary.yaml**

Compact run summary:
```yaml
status: success
evaluator: centroid
train_split: train
val_split: val
num_prototypes: 10
val_accuracy: 0.85
```
---
**predictions.pt**

Saved as a PyTorch dictionary:
```yaml
{
    "image_ids": [...],
    "labels": Tensor[N],
    "pred_labels": Tensor[N],
    "logits": Tensor[N, C],

    # Prototype information
    "prototype_labels": Tensor[C],
    "prototypes": Tensor[C, D],
}
```
---
### Design Notes
**Why centroid?**
- Provides a strong, cheap baseline
- Much faster than kNN (O(C) vs O(N))
- Easy to interpret
- No hyperparameter tuning required
---
## Relationship to kNN
| Evaluator | Reference set     | Complexity | Notes           |
| --------- | ----------------- | ---------- | --------------- |
| kNN       | all train samples | O(N)       | more flexible   |
| centroid  | class means       | O(C)       | faster, simpler |

---
### Constraints (unchanged)
- Evaluators consume embedding artifacts only
- No dataloaders
- Model / pooler contract is frozen
- Artifact format is stable
---
### Future Extensions (not in PR-8)
- Multi-prototype per class
- Class-conditional covariance / Mahalanobis distance
- Prototype refinement
- Temperature scaling / calibration
- Open-set recognition
---
### Summary

Centroid evaluator provides:

✅ Fast baseline
✅ Fully artifact-driven evaluation
✅ Consistent interface with existing evaluators
✅ Minimal configuration
---

## Evaluator Comparison

| Evaluator     | Training Required | Complexity | Use Case                      |
|--------------|------------------|------------|-------------------------------|
| Linear Probe | Yes              | O(N)       | Learned classifier baseline   |
| kNN          | No               | O(N)       | Local structure evaluation    |
| Centroid     | No               | O(C)       | Global cluster structure      |

---

## Retrieval Evaluator (PR-9)

PR-9 introduces the **Retrieval Evaluator** for artifact-driven embedding evaluation.

New functionality includes:

- Retrieval evaluator based on embedding artifacts only
- Train split as gallery / reference set
- Val split as query set
- Recall@K metrics
- mAP@K metrics
- Hydra-configurable retrieval runner
- Retrieval predictions artifact for debugging and inspection
- Unit tests / runner tests / script tests

This evaluator follows the existing evaluator framework:

```text
embedding artifacts
  -> io loader
  -> evaluator
  -> runner
  -> result writer
```
Critical constraints preserved in PR-9:

- Evaluators consume embedding artifacts only
- Evaluators do not use dataloaders directly
- Model + Pooler contract remains unchanged
- Artifact format remains stable

### Retrieval Evaluator

The Retrieval Evaluator measures how well embeddings support ranking-style retrieval.

Evaluation protocol in PR-9:

- gallery split: `train`
- query split: `val`

For each query embedding, the evaluator retrieves nearest gallery embeddings and computes retrieval metrics from label-based relevance.

Supported metrics:

- `Recall@K`
- `mAP@K`

Supported similarity metrics:

- `cosine`
- `l2`

### Retrieval Evaluator Config

Example config structure:
```yaml
evaluation:
  retrieval:
    enabled: true

    input:
      train_split_dir: runs/<run_name>/embeddings/train
      val_split_dir: runs/<run_name>/embeddings/val
      normalize_embeddings: false
      map_location: cpu

    output:
      output_dir: runs/<run_name>/eval/retrieval
      save_predictions: true

    evaluator:
      metric: cosine
      batch_size: 1024
      device: cpu
      topk: [1, 5, 10]
      save_predictions_topk: 10
      exclude_same_image_id: false
```

### Run Retrieval Evaluation

Example:
```bash
python scripts/run_retrieval.py \
  evaluation.retrieval.enabled=true \
  evaluation.retrieval.input.train_split_dir=runs/<run_name>/embeddings/train \
  evaluation.retrieval.input.val_split_dir=runs/<run_name>/embeddings/val \
  evaluation.retrieval.output.output_dir=runs/<run_name>/eval/retrieval
```
Typical usage writes outputs under:
```text
runs/<run_name>/eval/retrieval/
```

### Retrieval Outputs

The Retrieval Evaluator writes:
```text
runs/<run_name>/eval/retrieval/
├── metrics.yaml
├── summary.yaml
├── config.yaml
└── predictions.pt
```

**`metrics.yaml`**

Contains structured retrieval metrics and evaluator metadata, including:

- gallery/query split information
- number of gallery/query samples
- embedding dimension
- number of classes
- recall_at_k
- map_at_k

**`summary.yaml`**

Contains a compact summary for quick inspection, typically including:

- evaluator name
- gallery/query split
- sample counts
- key retrieval metrics such as:
  - recall_at_1
  - recall_at_5
  - map_at_1
  - map_at_5

**`config.yaml`**

Contains the resolved run configuration used for this retrieval evaluation.

**`predictions.pt`**

Contains retrieval evidence for debugging and inspection, including:

- query image ids
- query labels
- retrieved gallery indices
- retrieved gallery labels
- retrieved gallery scores
- retrieved gallery image ids
- top-k relevance matches

### Retrieval Metric Definition
**Recall@K**

For each query, Recall@K is counted as a hit if at least one relevant gallery sample appears in the top-K retrieved results.

**mAP@K**

mAP@K is computed from the ranked top-K retrieval results using label-based relevance.

Queries without any positive sample in the gallery are excluded from the mAP denominator.

### Design Notes

PR-9 keeps the retrieval evaluator intentionally narrow and stable:

- artifact-driven only
- no dataloaders
- no model contract changes
- no artifact format changes
- no ANN / FAISS dependency in M1
- no multi-label retrieval in M1

This keeps the evaluator aligned with the existing evaluation stack while leaving room for future work such as:

- val-to-val retrieval
- additional ranking metrics
- large-scale retrieval backends
- richer retrieval diagnostics

---
## Checkpoint & Resume (M1)

### Checkpoint Layout

After each training bootstrap run, checkpoints are written to:
```text
<run_dir>/checkpoints/
  ├── latest.pt
  ├── best.pt
  └── epoch_0000.pt
```

- `latest.pt`: most recent checkpoint (always updated)
- `best.pt`: best checkpoint (first run = best)
- `epoch_xxxx.pt`: per-epoch snapshot (M1 uses epoch_0000)

### Resume Modes

Resume behavior is controlled by:

```yaml
train:
  resume:
    enabled: false
    mode: "warm_start"   # or "full_resume"
    checkpoint_path: null
    auto_resume_latest: false
```
**warm_start**
- Loads model weights only
- Does NOT restore training state

**full_resume**
- Loads:
  - model weights
  - trainer state (epoch, global_step)
- Designed for continuing training


### Auto Resume

When:

```yaml
train.resume.enabled = true
train.resume.auto_resume_latest = true
train.resume.checkpoint_path = null
```

The system will automatically resume from:
```text
<run_dir>/checkpoints/latest.pt
```

### Examples

**Fresh run**

```bash
python scripts/train.py
```

**Auto resume latest**
```bash
python scripts/train.py \
  train.resume.enabled=true \
  train.resume.mode=full_resume \
  train.resume.auto_resume_latest=true \
  train.resume.checkpoint_path=null
```
**Warm start from checkpoint**
```bash
python scripts/train.py \
  train.resume.enabled=true \
  train.resume.mode=warm_start \
  train.resume.checkpoint_path=/path/to/checkpoint.pt
```

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