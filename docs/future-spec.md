# DIE-VFM Future Spec (Round2+)

## Purpose

This document defines the intended evolution path beyond the current M1 and Round1 repository scope. It is a roadmap-level contract, not a statement of completed functionality.

This document must not be used by itself to justify current support claims, tests, or README language.

## Future Positioning

Die VFM is intended to evolve from an artifact-centric visual embedding platform into a multi-stage representation training platform for die-level visual foundation models.

Target use cases include:

- classification
- retrieval
- clustering
- anomaly-oriented representation analysis
- few-shot adaptation

## Future Runtime Modes

The next formal trainer modes are expected to include:

- `round2_ssl`
- `round3_supcon`

Possible later expansion:

- `round4_scaleup`

## Round2 SSL Direction

`round2_ssl` is intended for domain adaptation through self-supervised learning.

Expected characteristics:

- long-running training
- optimizer-driven updates
- scheduler continuity
- AMP continuity
- EMA teacher continuity
- strong checkpoint/resume requirements

Expected parameter-update boundary:

- `round2_ssl` is the first stage where model weights are expected to be updated on domain data
- selective freeze policies are expected here (for example: freeze parts of backbone while updating selected backbone blocks, pooler, projector, or SSL heads)
- freeze policy should be explicit in config and reflected in checkpoint metadata

## Round3 SupCon Direction

`round3_supcon` is intended for supervised contrastive fine-tuning on labeled subsets.

Expected characteristics:

- projection head as trainable state
- optional classifier head
- optimizer and scheduler persistence
- stronger compatibility checks across resumes

Expected parameter-update boundary:

- `round3_supcon` continues gradient-based adaptation with labels
- selective layer freezing and trainable-subset updates are expected controls in this stage as well
- this stage may tighten which modules are trainable compared with `round2_ssl`, depending on label volume and overfitting risk

## Round-Boundary Rule

To avoid semantic drift between rounds:

- `round1_frozen` remains non-training orchestration for embedding export and offline evaluation
- `round1_frozen` remains single-shot and does not carry epoch/resume semantics
- any workflow that updates model weights belongs to `round2_ssl` or later
- epoch/resume and checkpoint-continuation semantics for adaptation stages belong
  to `round2_ssl` or later
- selective-layer freeze plus trainable-subset updates must not be claimed as current Round1 capability until promoted through runtime, config, tests, and docs

## Backbone Roadmap

Target future formal backbones include:

- `dinov2` as a fully wired production-capable option

Longer-term backbone expansion may include additional token-centric models, but only after builder wiring, config support, documentation, and tests are aligned.

## Pooler Roadmap

The current mainline future direction remains attention pooling.

Future candidate poolers may include:

- `cls_pooler`
- multi-head attention pooling
- query-based pooling
- MIL-style top-k pooling

These remain future options until they are promoted by code, config, tests, and documentation together.

## Artifact Roadmap

Embedding artifacts may evolve to include richer outputs such as:

- token weights
- token-level summaries
- stronger lineage metadata
- tighter checkpoint linkage

Any artifact extension must preserve versioned compatibility and should not silently break existing loaders.

## Evaluator Roadmap

Future evaluator scope is expected to grow toward a fuller representation benchmark suite.

Candidate future outputs include:

- `NMI`
- `ARI`
- hard-case reports
- nearest-neighbor inspection artifacts
- richer retrieval diagnostics
- anomaly-oriented evaluation outputs

These are future benchmark goals rather than current mandatory outputs.

## Future Checkpoint / Resume Direction

The future target is a true full-resume contract for training-centric rounds.

Target restored state:

- model
- optimizer
- scheduler
- scaler
- trainer state
- RNG state
- EMA teacher
- config snapshot
- compatibility check results
- optional sampler or data state

Official future guarantee target:

- same-world-size full resume on a single node

Different-world-size resume should remain best-effort unless separately specified.

## Future Config Policy

The repository should converge toward:

- typed config schema
- synchronized naming across docs, configs, and runtime
- fail-fast compatibility validation

Future design should eliminate current drift between `schema.py`, runtime usage, and human-facing docs.

## Promotion Rule

A feature should move from future spec into current spec only when all of the following are true:

- runtime code exists
- config contract exists
- tests exist
- README and docs are aligned

Until then, the feature should be described as planned, experimental, or future-facing rather than supported.

If there is conflict between this document and `docs/current-spec.md`, current implementation work must follow `docs/current-spec.md`.
