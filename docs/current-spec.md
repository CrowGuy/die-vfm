# DIE-VFM Current Spec (M1 / Round1)

## Purpose

This document defines the current formal spec for the repository as it exists today. It is the contract that implementation and testing should follow for M1 and Round1 work.

If this document conflicts with `docs/future-spec.md`, implementation and testing must follow this document for current-scope work.

## Scope

Current scope is intentionally narrow:

- Stable token-centric model contract
- Embedding artifact export
- Artifact-driven evaluation
- `bootstrap` runtime flow
- `round1_frozen` runtime flow
- M1 checkpoint/resume behavior

The current spec does not formally include SSL training, supervised contrastive training, or advanced distributed resume guarantees.

Current-scope changes should not be merged conceptually with future-scope design unless the feature has been explicitly promoted.

## System Positioning

Die VFM is currently an artifact-centric visual embedding platform. The present repository is not yet a full multi-stage training platform.

The current canonical pipeline is:

```text
Dataset -> Model (Backbone + Pooler) -> Embedding Artifact -> Evaluator
```

## Runtime Modes

### `bootstrap`

Purpose:

- Dataloader smoke test
- Model forward smoke test
- Basic checkpoint smoke test

Behavior:

- Builds train dataloader
- Reads one batch
- Builds model
- Runs one forward pass
- Writes smoke metadata
- Saves checkpoint set

`bootstrap` is not a full training algorithm.

### `round1_frozen`

Purpose:

- Frozen embedding export
- Offline artifact-driven evaluation
- Per-epoch summary and checkpoint writing

Behavior:

- Builds model
- Applies freeze policy
- Exports train and val embeddings
- Runs enabled Round1 evaluators
- Writes epoch summary
- Saves checkpoint set

`round1_frozen` should be treated as the current production experiment runner.

Current `round1_frozen` orchestration scope:

- `linear_probe`
- `knn`
- `retrieval`

`centroid` is part of the current standalone evaluator surface, but it is not
currently orchestrated by `round1_frozen`.

## Dataset Contract

Per-sample dataset contract:

```python
{
    "image": Tensor,
    "label": int | None,
    "image_id": str,
    "meta": dict,
}
```

Batch contract:

```python
{
    "image": Tensor,       # [B, C, H, W]
    "label": Tensor | None,
    "image_id": list[str],
    "meta": list[dict],
}
```

Required rules:

- `image_id` must be non-empty and unique within a shard
- `meta` must remain aligned with batch rows
- labels may be absent for unlabeled flows

## Model Contract

Current model outputs are defined by the runtime code and are treated as formal contracts.

### `BackboneOutput`

```python
@dataclass
class BackboneOutput:
    patch_tokens: torch.Tensor
    cls_token: torch.Tensor | None
    token_mask: torch.Tensor | None
    feature_dim: int
    patch_grid: tuple[int, int] | None
    metadata: dict[str, Any]
```

Rules:

- `patch_tokens` are the primary representation
- downstream code must not assume `cls_token` exists
- `token_mask` is optional

### `PoolerOutput`

```python
@dataclass
class PoolerOutput:
    embedding: torch.Tensor
    token_weights: torch.Tensor | None
    metadata: dict[str, Any]
```

### `ModelOutput`

```python
@dataclass
class ModelOutput:
    embedding: torch.Tensor
    backbone: BackboneOutput | None
    pooler: PoolerOutput | None
    metadata: dict[str, Any]
```

## Supported Components

### Backbones

Formally supported in current spec:

- `dummy`

Present in the codebase but not yet formal current-spec support:

- `dinov2`

`dinov2` remains future-facing until builder wiring, runtime validation, and tests are aligned.

### Poolers

Formally supported in current spec:

- `mean`
- `identity`
- `attn_pooler_v1`

## Embedding Artifact Spec

Embedding artifacts are first-class outputs.

### Layout

```text
runs/<run_name>/embeddings/<split>/
├── manifest.yaml
└── part-00000.pt
```

### Manifest contract

Required fields:

- `artifact_type`
- `artifact_version`
- `format`
- `split`
- `num_samples`
- `embedding_dim`
- `dtype`
- `has_labels`
- `num_shards`
- `shards`

### Shard payload contract

Required payload fields:

- `embeddings`
- `labels`
- `image_ids`
- `metadata`

Current spec does not require `token_weights` to be saved in the artifact.

### Artifact export config status

Current runtime-effective `artifact.embedding` fields:

- `enabled`
- `output_subdir`
- `export_splits`
- `include_test_split`

Current M1 placeholder `artifact.embedding` fields:

- `save_labels`
- `save_metadata`
- `artifact_version`
- `shard_size`

Placeholder fields exist in the current config surface but do not yet change
runtime exporter behavior.

## Evaluator Spec

Current repository formally supports these artifact-driven evaluators:

- `linear_probe`
- `knn`
- `centroid`
- `retrieval`

Evaluator rules:

- evaluators must read embedding artifacts
- evaluators must not depend on dataloader runtime
- evaluators must write stable filesystem outputs

Current evaluator positioning:

- `linear_probe`, `knn`, and `retrieval` are current standalone evaluators and
  are also the evaluator set currently orchestrated by `round1_frozen`
- `centroid` is current standalone evaluator support, but not part of the
  current `round1_frozen` orchestration path

Expected evaluator outputs:

- `metrics.yaml`
- `summary.yaml`
- `config.yaml`

Optional outputs depending on evaluator:

- `predictions.pt`
- `history.yaml`

## Checkpoint / Resume Current Scope

Current checkpoint behavior is M1-scoped rather than the future full contract.

Current formal checkpoint set:

- `latest.pt`
- `best.pt`
- `epoch_xxxx.pt`

Current formal resume modes:

- `warm_start`
- `full_resume`

Current formal payload fields:

- `checkpoint_version`
- `epoch`
- `global_step`
- `model_state_dict`
- `optimizer_state_dict` when present
- `lr_scheduler_state_dict` when present
- `grad_scaler_state_dict` when present
- `trainer_state`
- `metadata`

Current spec does not formally guarantee:

- RNG restoration
- EMA restoration
- sampler/data state restoration
- corrupted checkpoint fallback
- config compatibility enforcement beyond minimal payload validation

## Config Contract

Current canonical configuration source is:

- `configs/`
- runtime code

`die_vfm/config/schema.py` is intentionally treated as a typed helper and current-config mirror for M1 / Round1 work.

Under the current repository policy:

- `configs/` and runtime code are the canonical configuration source
- `schema.py` is not a formal enforcement layer
- `schema.py` may be used to improve readability, typing, and future cleanup
- when `schema.py` disagrees with runtime behavior, runtime behavior wins

Current config expectations:

- `train.mode` selects the current runtime path and defaults to `bootstrap` when omitted by config composition
- `train.num_epochs` is the current formal epoch-count field for Round1 work
- `evaluation.run_*` fields are top-level orchestration toggles in the root config and are the control point used by `round1_frozen` for its current evaluator set: `linear_probe`, `knn`, and `retrieval`
- nested `evaluation.<name>` subtrees hold evaluator-specific input, output, and algorithm settings
- nested evaluator `enabled` fields are the control point used by standalone evaluator scripts, but do not replace the root current-scope orchestration hierarchy

## Current Acceptance Boundaries

An implementation belongs to the current spec only if all of the following are true:

- it exists in runtime code
- it has a stable contract
- it is represented in config
- it is testable in the current repository

## Out of Scope

The following belong to future spec, not current spec:

- `round2_ssl`
- `round3_supcon`
- full EMA-aware resume
- same-world-size continuity guarantees
- token-weight artifact persistence
- advanced representation benchmark outputs such as mandatory `NMI` and `ARI`
