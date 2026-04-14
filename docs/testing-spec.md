# DIE-VFM Testing Spec

## Purpose

This document defines what the test suite should validate for the current repository scope. It is written against the current M1 and Round1 implementation boundary.

## Testing Principles

- Tests validate contracts, not aspirational design
- Current-spec behavior must have at least one direct test path
- Future-spec behavior must not be implied as already covered
- Fast deterministic tests are preferred when possible

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

- `scripts/train.py` in `bootstrap` mode
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

Required coverage:

- evaluator core logic
- runner config resolution
- output writing
- metrics and summary generation
- predictions artifact generation where enabled

Primary goal:

- ensure evaluators remain independent from model runtime

## 6. Checkpoint Tests

These tests validate current M1 checkpoint behavior.

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

## Acceptance Matrix

Each current formal subsystem should map to at least one test category:

- dataset contract -> contract tests
- model contract -> contract tests
- config composition -> builder/config tests
- bootstrap mode -> smoke tests
- embedding artifact -> artifact tests
- evaluator outputs -> evaluator tests
- M1 resume -> checkpoint tests

## Future Testing Expansion

When `round2_ssl` or `round3_supcon` enter current scope, the suite should expand to include:

- optimizer continuity tests
- scheduler continuity tests
- EMA restoration tests
- compatibility rejection tests
- longer end-to-end resume scenarios

Those additions should happen when the code enters current spec, not before.
