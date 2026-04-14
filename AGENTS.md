# DIE-VFM Repository AGENTS.md

## Purpose

This file defines repository-specific execution boundaries for Codex and other coding agents working in this repo.

The goal is to keep implementation aligned with:

- `docs/current-spec.md`
- `docs/future-spec.md`
- `docs/testing-spec.md`
- `docs/checkpoint-resume-spec.md`

This repository currently prioritizes M1 and Round1 correctness over feature expansion.

## Current Development Phase

This repository is currently in:

- `Current Spec (M1 / Round1)` stabilization

Agents must treat the following as current formal scope:

- `bootstrap`
- `round1_frozen`
- artifact-driven evaluation
- current embedding artifact contract
- current M1 checkpoint/resume contract

Agents must treat the following as future scope unless explicitly requested and approved:

- `round2_ssl`
- `round3_supcon`
- full EMA-aware resume
- distributed resume guarantees
- promotion of `dinov2` into current formal support

## Canonical Sources of Truth

When there is drift, use this precedence order:

1. Runtime code
2. `configs/`
3. `docs/current-spec.md`
4. Other docs
5. Legacy or drifting schema helpers

At the current stage, `die_vfm/config/schema.py` is not automatically the canonical truth if it disagrees with runtime behavior.

## Implementation Priorities

Work should follow this order unless the user explicitly redirects:

1. Fix spec drift
2. Fix config and schema drift
3. Fix or add tests for current-spec behavior
4. Stabilize current runtime behavior
5. Only then expand functionality

Do not jump to future-scope implementation while current-scope contracts remain inconsistent.

## Scope Boundaries

### Allowed default work

Without extra approval, agents should work on:

- current-spec documentation alignment
- config cleanup
- schema cleanup
- current test fixes
- current runtime correctness fixes
- current checkpoint and artifact contract stabilization

### Work that requires explicit user confirmation

Pause and confirm before:

- promoting a future-spec feature into current spec
- introducing new dependencies
- changing public contracts in a breaking way
- changing artifact schema version
- changing checkpoint payload format incompatibly
- removing existing configs or tests that still describe supported behavior
- implementing `round2_ssl` or `round3_supcon`

## Documentation Rules

When a current formal contract changes, agents must update all affected layers together:

- runtime code
- config
- tests
- docs

`README.md` must remain a concise project entrypoint.
Detailed contracts belong in `docs/`.

Do not describe a feature as supported in `README.md` unless:

- runtime code exists
- config support exists
- tests exist
- docs are aligned

## Testing Rules

For current-scope changes, agents should prefer validating through the categories defined in `docs/testing-spec.md`:

- contract tests
- builder/config tests
- bootstrap smoke tests
- embedding artifact tests
- evaluator tests
- checkpoint tests

If tests cannot be run, say so explicitly and do not imply validation that did not happen.

## Architecture Boundaries

Agents should preserve these current architectural boundaries:

- evaluators consume embedding artifacts, not dataloaders
- `patch_tokens` remain the primary representation
- embedding artifacts are first-class outputs
- `bootstrap` is a smoke/runtime bring-up path
- `round1_frozen` is the current experiment runner

Do not blur these boundaries by introducing shortcuts that bypass artifacts or hide contract changes.

## Promotion Rule

A feature may move from future spec into current spec only when all of the following are true:

- runtime code is implemented
- config contract is implemented
- tests are implemented
- docs are updated
- the user has chosen to promote it

Until then, future features must be described as planned or experimental, not supported.

## Current Roadmap

Execution should follow `docs/implementation-roadmap.md`.

If a requested task conflicts with the roadmap, agents should still follow the user request, but should state the conflict clearly.
