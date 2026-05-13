# DIE-VFM Round2 SSL First Pilot Readiness Checklist

## Purpose

This checklist is the final preflight gate before launching the first formal
`round2_ssl` pilot.

It is intentionally operational:
- confirm the experimental Round2 runtime is stable enough for first pilot use
- prevent avoidable launch-time mistakes
- make the pilot setup reproducible

This document does **not** promote `round2_ssl` into current formal support.
Round2 remains future scope and experimental until explicitly promoted.

Related references:
- [Future Spec](future-spec.md)
- [Round2 Experiment Planning Draft](round2-experiment-planning-draft.md)
- [Testing Spec](testing-spec.md)

## Readiness Decision

Use this checklist as a `go / no-go` gate.

- `GO`: every required item is complete
- `NO-GO`: any required item is unknown, red, or only assumed

Recommended rule:
- do not start the first formal pilot with unresolved runtime ambiguity
- if a required item fails, fix it first or explicitly downgrade the pilot goal

## 1. Scope Alignment

Required:
- confirm this run is explicitly treated as `round2_ssl` experimental work
- confirm nobody is describing Round2 as promoted current support
- confirm the pilot goal is representation adaptation, not Round1 replacement

Required pilot objective:
- improve `visually same` clustering
- especially improve `cross-source` closeness across `lot / machine / time`

Required success metrics:
- pair benchmark `same_vs_different_cosine_auc_like`
- `cross-source` slice AUC-like
- `hard_same_far`
- `hard_different_close`

Required guardrails:
- no material regression on `kNN`
- no material regression on `retrieval`
- no material regression on `linear_probe`

## 2. Runtime / Config Readiness

Required:
- `train.mode=round2_ssl`
- `train.update_mode` explicitly set for this pilot
- `train.precision_mode` explicitly set for this pilot
- `round2.distributed.strategy=ddp`
- `round2.loss.token_loss_enabled` explicitly set
- `round2.evaluation.cadence=end_only`

Required supported update modes:
- `projector_pooler_only`
- `last_n_blocks`
- `full_backbone`

Required interpretation:
- runtime default update mode is `full_backbone`
- first planned experiment order remains:
  - `projector_pooler_only`
  - `last_n_blocks`
  - `full_backbone`

Required precision policy:
- debug bring-up: `fp32`
- pilot / production: `bf16`

Required distributed/runtime guarantees already expected to hold:
- DDP checkpoint save/load uses the same model-state contract
- post-train artifact ownership is `rank 0 only`
- final metrics are broadcast back to non-rank-zero processes
- sampler and dataloader are bound to the same dataset instance in DDP mode

## 3. Resume Semantics Readiness

Required:
- decide before launch whether the pilot is:
  - fresh run
  - `warm_start`
  - `full_resume`

Required contract:
- `warm_start` may initialize from another run
- `full_resume` must stay within the same run lineage

Required if using resume:
- `train.resume.enabled=true`
- `train.resume.mode` explicitly set
- checkpoint path resolution is known in advance
- expected checkpoint file exists before launch

Required if using `full_resume`:
- checkpoint comes from the same `round2/checkpoints` lineage
- team understands that optimizer / scheduler / trainer state will continue

Required if using `warm_start`:
- team understands that only model weights continue
- epoch / global step restart from the new run

## 4. Data Readiness

Required:
- Round2 pilot corpus is separate from Round1 evaluation anchors
- pair benchmark `did` values are excluded from the Round2 training corpus
- Round1 val/query images are excluded from the Round2 training corpus
- source bucket key is fixed as:
  - `fine_label + lot + machine + time_bucket`
- same-source cap is fixed at `6`
- pilot corpus target size is fixed at approximately `500K`

Required interpretation:
- Round2 training corpus is `train-only`
- tail coverage is treated as corpus-level preservation, not batch balancing
- normal/abnormal ratio is not constrained in v1

Recommended spot checks before launch:
- sampled corpus row count matches expectation
- exclusion counts are recorded
- source bucket columns exist and are non-empty after default filling
- no accidental overlap remains with pair benchmark `did`

## 5. Evaluation Asset Readiness

Required:
- Round1 baseline reference is frozen and known
- pair benchmark assets exist:
  - `pair_candidates.csv`
  - `annotations.csv`
- slicing configuration is decided before launch

Required runtime understanding:
- artifact-driven evaluators require exported embedding manifests
- pair benchmark now fails fast if configured `embedding_splits` point to
  missing embedding manifests

Required before enabling pair benchmark:
- `round2.evaluation.run_pair_benchmark=true` only when the requested
  `embedding_splits` are intentionally available

Required before enabling artifact evaluators:
- `val` export exists if you expect:
  - `linear_probe`
  - `kNN`
  - `retrieval`

Required before enabling slicing:
- pair benchmark must already be enabled and valid

## 6. Method Configuration Readiness

Required Round2 v1 method assumptions:
- `teacher-student` SSL
- EMA teacher enabled
- global projected cosine loss is the main loss

Current token-loss contract:
- token-level capability exists
- token loss is disabled by default
- if token loss is disabled, token projector is removed from the trainable set

Required augmentation policy:
- `RandomHorizontalFlip`
- `RandomVerticalFlip`

Required pre-launch clarity:
- whether this pilot is global-loss-only
- whether token loss is intentionally off
- whether EMA policy is `fixed` or `schedule`

## 7. Environment / Execution Readiness

Required:
- execution environment is chosen
- expected GPU count is known
- expected device string is known
- filesystem paths for outputs are writable

Recommended first-pilot launch pattern:
1. one small debug run
2. one reduced pilot smoke
3. first formal pilot

Required if using multi-GPU:
- DDP environment variables are controlled by the launcher
- per-rank device mapping is understood
- a short real multi-GPU smoke has already been run at least once

Recommended environment note:
- tests and single-process smoke are necessary but not sufficient
- a real 2-GPU or 8-GPU bring-up is still the final confidence check

## 8. Test / Validation Readiness

Required baseline test groups already expected to pass:
- `tests/test_config.py`
- `tests/test_round2_ssl_utils.py`
- `tests/test_round2_data_prep.py`
- `tests/test_round2_runner.py`
- `tests/test_round2_runtime.py`

Recommended pre-launch validation:
- rerun the Round2-focused test subset in the target environment
- run one real `round2_ssl` smoke with the chosen precision mode
- if using resume, run one resume smoke in the same environment

Required interpretation:
- no unverified environment assumptions
- no “it passed locally so cluster launch should be fine” reasoning

## 9. Pilot Command Freeze Record

Before launch, record all of the following:
- pilot name
- run name
- git commit SHA
- backbone
- update mode
- precision mode
- token loss enabled/disabled
- EMA policy
- scheduler name
- number of epochs
- batch size
- worker count
- distributed launcher command
- dataset manifest / corpus path
- pair benchmark paths

Recommended rule:
- treat the first formal pilot as a freeze record candidate
- do not rely on terminal scrollback as the only source of truth

## 10. Go / No-Go Questions

Answer `yes` to all required questions before launch:

1. Do we know exactly which Round2 objective this pilot is testing?
2. Do we know which update mode this pilot uses?
3. Do we know whether token loss is on or off?
4. Do we know whether this run is fresh, warm-start, or full-resume?
5. If resume is enabled, is the resume lineage valid for the chosen mode?
6. Is the Round2 pilot corpus already built and benchmark-excluded?
7. Are pair benchmark inputs present if pair benchmark is enabled?
8. Are requested embedding splits actually available for every enabled evaluator?
9. Has at least one smoke run been completed in the intended environment?
10. Is the exact launch command frozen somewhere durable?

If any answer is `no`, the pilot is not ready.

## Suggested Launch Sequence

Recommended sequence:

1. Run a minimal debug smoke
2. Run a small-scale multi-GPU smoke
3. Freeze the first formal pilot command
4. Launch the first formal pilot
5. Record:
   - summary paths
   - checkpoint paths
   - embedding artifact paths
   - pair benchmark output paths
6. Compare results against Round1 baseline and pair benchmark targets

## First Pilot Sign-Off

Fill this section before launch:

- Pilot owner:
- Date:
- Commit SHA:
- Run name:
- Dataset / corpus id:
- Update mode:
- Precision mode:
- Resume mode:
- Pair benchmark enabled:
- Slicing enabled:
- Multi-GPU smoke completed:
- Final decision: `GO` / `NO-GO`
- Notes:
