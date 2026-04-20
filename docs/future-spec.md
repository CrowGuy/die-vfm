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

Current formal backbone support already includes:

- `dummy`
- promoted `dinov2` (within current promotion scope:
  `bootstrap` and `round1_frozen`)

Longer-term backbone expansion may include additional token-centric models, but only after builder wiring, config support, documentation, and tests are aligned.

### `dinov2` Promotion Scope

The `dinov2` promotion that has landed in current scope was intentionally
narrow:

- promote only `dinov2`
- promote only for current runtime modes:
  - `bootstrap`
  - `round1_frozen`
- do not use this promotion to introduce Round2+ training semantics
- do not use this promotion to open a multi-backbone support matrix

Landed outcome of this promotion:

- `dinov2` can be selected through current config composition
- `bootstrap` can build the model and complete forward/runtime smoke with
  `dinov2`
- `round1_frozen` can run frozen embedding export and artifact-driven
  evaluation with `dinov2`

Landed promotion layers:

- backbone builder support
- runtime-usable config contract
- runtime validation for current supported paths
- dedicated tests
- current-spec and README wording updates

### `dinov2` Loading Semantics Checklist

This section defines the agreed loading semantics that now form the promoted
current-scope contract for `dinov2`.

Config surface:

- `model.backbone.allow_network: bool`
- `model.backbone.local_repo_path: str | null`
- `model.backbone.local_checkpoint_path: str | null`

Semantics:

- `pretrained` controls weight source only.
- architecture source resolution is independent from `pretrained`.
- `allow_network` controls whether missing local sources may be resolved via
  network-backed hub behavior.

Architecture source resolution:

1. If `local_repo_path` is provided, runtime should use it as the architecture
   source.
2. Else if `allow_network=true`, runtime may resolve architecture code through
   `torch.hub`.
3. Else fail-fast.

Weights source resolution:

1. If `pretrained=false`, runtime should skip pretrained checkpoint loading.
2. If `pretrained=true` and `local_checkpoint_path` is provided, runtime should
   load weights from that local checkpoint after validating path existence and
   file type.
3. If `pretrained=true` and `local_checkpoint_path` is not provided:
   - allow network-backed weight resolution only when `allow_network=true`
   - fail-fast when `allow_network=false`

Fail-fast wording policy:

- Errors should explicitly report:
  - active mode and relevant config keys
  - the missing local source (`repo` vs `checkpoint`)
  - one concrete remediation path
- Recommended wording patterns:
  - `DINOv2 architecture source is unavailable: set model.backbone.local_repo_path or enable model.backbone.allow_network=true.`
  - `DINOv2 pretrained offline load requires model.backbone.local_checkpoint_path when model.backbone.allow_network=false.`
  - `Configured DINOv2 local checkpoint does not exist: path=<...>.`
  - `Configured DINOv2 local checkpoint must point to a file: path=<...>.`
  - `Configured DINOv2 local repo does not exist: path=<...>.`
  - `Configured DINOv2 local repo must point to a directory: path=<...>.`
  - `DINOv2 local_checkpoint_path is only valid when model.backbone.pretrained=true.`

Offline deployment checklist:

- prepare local DINOv2 repo and checkpoint assets on a networked machine
- transfer prepared assets to offline runtime hosts
- run with `allow_network=false` and explicit local paths
- require fail-fast instead of implicit network fallback when local assets are
  missing

Out of scope for this promotion:

- `round2_ssl` or `round3_supcon`
- optimizer/scheduler/resume semantics for `dinov2`
- multi-backbone benchmark matrix support
- additional backbone promotions in the same track

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
