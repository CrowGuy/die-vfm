# DIE-VFM Testing Spec

## Purpose

This document defines what the test suite should validate for the current repository scope. It is written against the current M1 and Round1 implementation boundary.

## Testing Principles

- Tests validate contracts, not aspirational design
- Current-spec behavior must have at least one direct test path
- Future-spec behavior must not be implied as already covered
- Fast deterministic tests are preferred when possible

## Current Capabilities

The current repository treats the following as formal, testable capabilities:

- config composition and builder resolution
- dataset sample and batch contract
- model output and pooler contract
- `bootstrap` smoke runtime path
- embedding artifact export and load
- standalone evaluator execution:
  - `linear_probe`
  - `knn`
  - `centroid`
  - `retrieval`
- `round1_frozen` orchestration for:
  - `linear_probe`
  - `knn`
  - `retrieval`
- M1 checkpoint/resume utilities (bootstrap scope)

## Test Layers

## 1. Contract Tests

These tests validate fundamental data and model contracts.

Required coverage:

- dataset sample shape and field contract
- batch collation contract
- backbone output contract
- pooler output contract
- model output contract
- embedding shard payload validation
- embedding manifest validation

Primary goal:

- detect contract breakage early

## 2. Builder and Config Tests

These tests validate that the configured system can be composed.

Required coverage:

- Hydra base config composition
- dataset builder behavior
- backbone builder behavior
- pooler builder behavior
- model builder behavior

Primary goal:

- catch config drift and unsupported combinations early

## 3. Bootstrap Smoke Tests

These tests validate the current smoke entrypoint behavior.

Required coverage:

- `scripts/run.py` in `bootstrap` mode
- dataloader smoke path
- model forward smoke path
- run directory creation
- smoke artifact creation
- checkpoint set creation

Primary goal:

- confirm that a new environment can execute the minimum runtime path

## 4. Embedding Artifact Tests

These tests validate embedding export and load behavior.

Required coverage:

- export one split successfully
- load one split successfully
- image id alignment
- metadata alignment
- label handling
- empty dataloader rejection
- incompatible shard payload rejection

Primary goal:

- keep artifact schema stable and safe for evaluator reuse

## 5. Evaluator Unit and End-to-End Tests

These tests validate artifact-driven evaluation.

Current evaluator scope:

- linear probe
- knn
- centroid
- retrieval

Current positioning:

- `linear_probe`, `knn`, and `retrieval` should be covered both as standalone
  evaluators and as the evaluator set orchestrated by `round1_frozen`
- `centroid` should be covered as current standalone evaluator support; it is
  not currently part of `round1_frozen` orchestration coverage

Required coverage:

- evaluator core logic
- runner config resolution
- output writing
- metrics and summary generation
- predictions artifact generation where enabled

Primary goal:

- ensure evaluators remain independent from model runtime

## 6. Checkpoint Tests

These tests validate current M1 checkpoint behavior in bootstrap scope.

Required coverage:

- atomic save creates target checkpoint files
- payload validation rejects malformed checkpoints
- `warm_start` restores model weights
- `full_resume` restores trainer state
- bootstrap auto-resume path
- checkpoint naming and file placement

Primary goal:

- keep current resume semantics reliable within current scope

## Current Non-Goals for Testing

The current suite does not need to guarantee the following yet:

- SSL teacher EMA continuity
- exact RNG continuity
- distributed multi-world-size resume behavior
- full config compatibility enforcement on resume
- token-weight artifact persistence

These belong to future-spec testing.

## Acceptance and Coverage Matrix

### 1. Config Composition and Builders

Current capability:

- Hydra config composes under the current root config
- current builder paths resolve supported dataset, backbone, pooler, and model objects

Acceptance criteria:

- `configs/config.yaml` composes successfully under Hydra
- the explicit `debug_model_smoke` preset applies its intended root overrides
- dataset builders return current dataset adapters and dataloaders
- pooler and model builders resolve current supported runtime objects

Existing tests:

- `tests/test_config.py`
- `tests/test_dummy_dataset.py`
- `tests/test_cifar10_dataset.py`
- `tests/test_pooler_builder.py`
- `tests/models/test_backbone_builder.py`
- `tests/models/test_model_builder.py`

Coverage status:

- covered

### 2. Dataset and Batch Contract

Current capability:

- datasets provide the current per-sample contract and dataloaders collate the current batch contract

Acceptance criteria:

- samples expose `image`, `label`, `image_id`, and `meta`
- collated batches preserve shape and row alignment for images, labels, ids, and metadata
- supported dataset adapters reject unsupported configuration where required

Existing tests:

- `tests/test_dummy_dataset.py`
- `tests/test_cifar10_dataset.py`

Coverage status:

- covered

### 3. Model and Pooler Contract

Current capability:

- current supported backbone and poolers produce outputs consistent with the model contract

Acceptance criteria:

- `dummy` backbone produces the expected feature contract
- `mean` and `attn_pooler_v1` satisfy current output and validation behavior
- the top-level model forward path returns an embedding-centered output without requiring downstream code to assume `cls_token`

Existing tests:

- `tests/models/test_dummy_backbone.py`
- `tests/models/test_mean_pooler.py`
- `tests/test_attn_pooler_v1.py`
- `tests/models/test_model_forward_smoke.py`

Coverage status:

- covered

### 4. Bootstrap Smoke Runtime

Current capability:

- `scripts/run.py` in `bootstrap` mode exercises the minimum runnable environment path

Acceptance criteria:

- bootstrap creates the run directory structure and config snapshot
- bootstrap runs dataloader smoke and model forward smoke successfully
- bootstrap writes smoke metadata artifacts
- bootstrap writes the current checkpoint set
- bootstrap resume entry behavior works for current `warm_start` and `full_resume` scope

Existing tests:

- `tests/test_run_dir.py`
- `tests/test_train_bootstrap.py`

Coverage status:

- covered

### 5. Embedding Artifact Export and Load

Current capability:

- embedding artifacts can be exported, validated, and reloaded under the current single-shard M1 contract

Acceptance criteria:

- one split can be exported successfully with manifest plus shard payload
- one split can be loaded successfully from disk
- image ids, labels, and metadata remain aligned with embeddings
- empty export input is rejected
- malformed manifests and shard payloads are rejected
- the export CLI works for current disabled and default train/val export paths
- placeholder artifact config fields remain no-op in M1:
  - `save_labels=false` does not suppress labels when labels are present
  - `save_metadata=false` does not suppress metadata output
  - `artifact_version` override does not change manifest runtime version
  - `shard_size` override does not change current single-shard export behavior

Existing tests:

- `tests/test_embedding_artifact.py`
- `tests/test_embedding_exporter.py`
- `tests/test_embedding_loader.py`
- `tests/test_export_embeddings_script.py`

Coverage status:

- covered

### 6. Standalone Evaluators

Current capability:

- current standalone evaluator entrypoints and runners operate from embedding artifacts

Acceptance criteria:

- `linear_probe`, `knn`, `centroid`, and `retrieval` all support standalone execution from saved artifacts
- standalone evaluator scripts use nested `evaluation.<name>.enabled` as their enable gate
- evaluator runners resolve config, compute metrics, and write current output files
- optional predictions or history outputs are written only when enabled

Existing tests:

- `tests/test_linear_probe.py`
- `tests/test_linear_probe_io.py`
- `tests/test_linear_probe_metrics.py`
- `tests/test_linear_probe_trainer.py`
- `tests/test_linear_probe_runner.py`
- `tests/test_run_linear_probe_script.py`
- `tests/test_knn_evaluator.py`
- `tests/test_knn_runner.py`
- `tests/test_run_knn_script.py`
- `tests/test_centroid_evaluator.py`
- `tests/test_centroid_runner.py`
- `tests/test_run_centroid_script.py`
- `tests/test_retrieval_evaluator.py`
- `tests/test_retrieval_runner.py`
- `tests/test_run_retrieval.py`

Coverage status:

- covered

### 7. `round1_frozen` Orchestration

Current capability:

- `round1_frozen` exports embeddings, orchestrates the current Round1 evaluator set, and writes run summaries as a single-shot flow

Acceptance criteria:

- the runtime writes train and val embeddings for one run execution
- the current Round1 evaluator set (`linear_probe`, `knn`, `retrieval`) is executed when enabled
- root `evaluation.run_*` flags disable the corresponding Round1 evaluator path
- run summary is written with evaluator execution metadata
- Round1 contract does not rely on `train.num_epochs` or `train.resume.*`
- Round1 contract does not rely on training-style checkpoint continuation semantics

Existing tests:

- `tests/test_round1_runner.py`

Coverage status:

- covered

### 8. M1 Checkpoint and Resume Utilities (Bootstrap Scope)

Current capability:

- M1 checkpoint save/load, `warm_start`, and `full_resume` utilities work for bootstrap scope

Acceptance criteria:

- save writes `latest.pt`, per-epoch checkpoints, and `best.pt` when selected
- save keeps `best.pt` stable when `is_best=false`
- save does not leave temporary `*.tmp` checkpoint files behind
- malformed or unsupported checkpoint payloads are rejected
- malformed payload rejection includes checkpoint field type validation (`epoch`, `global_step`, `model_state_dict`, optional state dict fields, `trainer_state`, `metadata`)
- `warm_start` restores model weights only
- `full_resume` restores model, trainer state, optimizer, scheduler, and grad scaler state when present
- `full_resume` rejects missing optimizer, scheduler, or grad scaler state when those objects are explicitly requested
- bootstrap auto-resume resolves and uses `latest.pt` correctly
- resume path resolution prioritizes explicit `checkpoint_path` over `auto_resume_latest`

Existing tests:

- `tests/test_checkpoint_manager.py`
- `tests/test_train_bootstrap.py`

Coverage status:

- covered

## Current Gap List

### P1 Gaps
- no uncovered P1 gap is currently tracked for current-scope capabilities

### No Current P0 Gaps

- no uncovered current-scope capability is currently known to be completely
  untested

## Future Testing Expansion

When `round2_ssl` or `round3_supcon` enter current scope, the suite should expand to include:

- optimizer continuity tests
- scheduler continuity tests
- EMA restoration tests
- compatibility rejection tests
- longer end-to-end resume scenarios

Those additions should happen when the code enters current spec, not before.
