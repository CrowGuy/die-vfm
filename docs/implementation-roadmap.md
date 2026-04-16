# DIE-VFM Implementation Roadmap

## Purpose

This roadmap converts the current open work into an execution order that can be implemented directly. It is designed for the repository's current phase: `Current Spec (M1 / Round1)` stabilization.

## Guiding Principle

The repository should not expand into future-spec work until current-spec contracts, configs, docs, and tests are aligned.

## Phase 0: Governance and Source-of-Truth Alignment

Goal:

- make the repo safe for iterative implementation

Tasks:

1. Establish repo-level execution rules in `AGENTS.md`
2. Keep `README.md` as an entrypoint, not a full design dump
3. Treat `docs/current-spec.md` as the current formal product contract
4. Treat `docs/future-spec.md` as roadmap, not current support
5. Require doc/config/test updates whenever a current contract changes

Completion criteria:

- repo has a clear execution boundary for agents
- current and future scope are explicitly separated

Current status:

- repo-level governance files are in place
- future and current scope are split into separate docs
- README has been reduced to entrypoint scope
- continued work should now use these governance rules as default operating constraints

## Phase 1: Config and Schema Drift Cleanup

Goal:

- eliminate misleading config and schema mismatches in current scope, and make
  the helper role of `schema.py` explicit

Tasks:

1. Reconcile `die_vfm/config/schema.py` with actual runtime config naming
2. Fix `max_epochs` vs `num_epochs`
3. Review current config field names across:
   - `configs/config.yaml`
   - experiment configs
   - evaluator configs
   - runtime readers
4. Keep `schema.py` documented as a non-canonical helper mirror rather than a
   formal source of truth
5. Update any docs that still describe outdated config names

Completion criteria:

- current config names match runtime behavior
- schema helpers no longer contradict current runtime
- the repository is explicit that `schema.py` is a helper mirror, not the
  canonical config contract

Current status:

- `schema.py` has been repositioned as a non-canonical helper mirror under
  option C
- `max_epochs` vs `num_epochs` drift has been removed from the current config
  helper layer
- `debug_model_smoke.yaml` now behaves as a real global preset rather than a
  descriptive leftover
- Hydra compose order in `configs/config.yaml` now allows experiment presets to
  override root defaults as intended
- standalone evaluator scripts now consistently use nested `enabled` flags,
  while `round1_frozen` uses root `evaluation.run_*` orchestration flags
- artifact export config now explicitly separates runtime-effective fields from
  current placeholders
- core validation for Phase 1 has been covered by passing tests including:
  - `tests/test_config.py`
  - `tests/test_round1_runner.py`
  - `tests/test_run_knn_script.py`
  - `tests/test_run_centroid_script.py`
  - `tests/test_run_linear_probe_script.py`
  - `tests/test_run_retrieval.py`

## Phase 2: Test Drift Cleanup
Goal:

- ensure tests describe the current repository rather than older assumptions

Tasks:

1. Audit tests against `docs/testing-spec.md`
2. Fix tests that still validate outdated config names or contracts
3. Confirm each current formal subsystem maps to at least one test category
4. Identify missing tests versus broken tests

Completion criteria:

- tests no longer encode known stale assumptions
- there is a visible gap list for missing current-scope coverage

Current decision note:

- `centroid` remains current standalone evaluator support
- current `round1_frozen` orchestration is limited to `linear_probe`, `knn`,
  and `retrieval`
- docs and tests should preserve this distinction rather than expanding
  `round1_frozen` only to match wording

Current status:

- stale assumptions in `tests/test_config.py` have been reduced by separating
  base config composition from explicit `debug_model_smoke` preset behavior
- `tests/test_round1_runner.py` has been cleaned up to avoid placeholder
  artifact config assumptions and now covers `round1_frozen` evaluator
  disable-path behavior for root `evaluation.run_*` flags
- `tests/test_checkpoint_manager.py` now covers current-scope
  `lr_scheduler` and `grad_scaler` full-resume behavior, including missing-state
  rejection
- `tests/test_export_embeddings_script.py` now provides script-level coverage
  for disabled export and successful train/val artifact export
- current spec, testing docs, and tests now agree that `centroid` is current
  standalone evaluator support rather than part of the current
  `round1_frozen` orchestration path
- Phase 2 validation has been exercised by passing tests including:
  - `tests/test_config.py`
  - `tests/test_round1_runner.py`
  - `tests/test_checkpoint_manager.py`
  - `tests/test_export_embeddings_script.py`

## Phase 3: Acceptance Criteria and Test Matrix Hardening

Status:

- completed

Goal:

- turn current spec into a directly verifiable engineering contract

Tasks:

1. Add or refine acceptance criteria for:
   - `bootstrap`
   - `round1_frozen`
   - embedding export/load
   - evaluator outputs
   - M1 checkpoint/resume
2. Expand `docs/testing-spec.md` if new current-scope expectations are clarified
3. Map each acceptance criterion to a concrete test or identified test gap

Completion criteria:

- current spec has a direct verification path
- each current major capability has an explicit acceptance target

Current status:

- `docs/testing-spec.md` now includes a complete acceptance and coverage
  matrix for all current M1 / Round1 capabilities
- current capabilities, acceptance criteria, test mappings, and coverage states
  are explicitly documented in one place
- config and builder coverage has been hardened via direct builder tests and
  root config assertions
- embedding artifact coverage has been hardened with script-level checks for
  `export_splits`, `include_test_split`, and placeholder no-op fields
- standalone evaluator coverage has been hardened with subprocess CLI
  end-to-end tests for `knn`, `centroid`, and `retrieval`

## Phase 4: Round1 Runtime Stabilization

Status:

- completed

Goal:

- complete Round1 semantic cleanup so `round1_frozen` is unambiguously a
  single-shot inference/evaluation runner

Tasks:

1. Freeze the Round1 contract as single-shot:
   no `train.num_epochs`, no `train.resume.*`, no Round1 checkpoint
   continuation semantics
2. Update spec-facing docs (`current-spec`, `future-spec`, `testing-spec`,
   `README`) to reflect the Round1 single-shot boundary and Round2+ training
   boundary
3. Clean up Round1 config surface so training-only controls are not presented
   as Round1 runtime controls
4. Refactor Round1 runtime orchestration to remove epoch/resume/checkpoint
   continuation assumptions and keep only inference/evaluation orchestration
5. Rewrite Round1 acceptance tests to validate single-shot behavior and
   fail-fast boundary checks

Completion criteria:

- `round1_frozen` is consistently treated as a single-shot non-training runner
  across docs, config expectations, runtime behavior, and tests
- Round1 no longer exposes epoch/resume semantics to users
- Round1 outputs (artifacts and summaries) are coherent and diagnosable without
  training-style continuation semantics

Current status:

- Round1 runtime has been refactored into a dedicated single-shot runner
  (`Round1FrozenRunner`) and no longer uses epoch/resume/checkpoint-continuation
  semantics
- Round1 acceptance coverage now maps to `tests/test_round1_runner.py` and
  validates run-level artifact output, evaluator enable/disable behavior, and
  fail-fast boundaries
- entrypoint naming drift has been reduced by moving the runtime entry script
  from `scripts/train.py` to `scripts/run.py`, with README/testing references
  updated accordingly
- current-spec docs, testing spec, and runtime behavior now align on the
  Round1 single-shot contract

## Phase 5: Checkpoint / Resume Current-Scope Stabilization

Status:

- completed

Goal:

- make the M1 checkpoint contract explicit and reliable for bootstrap scope
  while keeping training-centric resume semantics in future rounds

Tasks:

1. Validate bootstrap checkpoint payload schema against
   `docs/checkpoint-resume-spec.md`
2. Fix drift between documented and actual bootstrap resume behavior
3. Improve current-scope tests for:
   - atomic save
   - warm start
   - full resume
   - latest/best/epoch naming
4. Keep Round1 single-shot semantics and future training-resume semantics
   clearly separated until Round2/Round3 implementation lands

Completion criteria:

- bootstrap checkpoint behavior is documented, implemented, and tested
  consistently without reintroducing Round1 semantic drift

Current status:

- bootstrap checkpoint payload validation is now strict and explicit, including
  type validation for core fields and optional state-dict fields
- save semantics coverage now validates latest/epoch/best behavior, best-file
  stability when `is_best=false`, and no leftover temporary files
- resume path behavior now has explicit coverage for priority rules
  (`checkpoint_path` over `auto_resume_latest`) and malformed payload rejection
- bootstrap runtime logging assertions now cover save-path logs and
  resume-related logs in subprocess acceptance tests
- wording drift has been reduced by using runtime-oriented bootstrap completion
  logs instead of training-oriented wording

## Phase 6: `dinov2` Decision Point

Status:

- completed

Goal:

- remove the ambiguous middle state around `dinov2`

Tasks:

1. Decide whether `dinov2` stays future-only for now or is promoted into current scope
2. If staying future-only:
   - keep docs explicit that it is not current formal support
3. If promoting to current:
   - wire builder support
   - validate config integration
   - add tests
   - update docs and README

Completion criteria:

- `dinov2` is no longer ambiguously described

Current status:

- `dinov2` has been explicitly kept as future-facing rather than promoted into
  current formal support
- current docs already align on this boundary:
  - `docs/current-spec.md` lists `dinov2` as present in codebase but not current
    formal support
  - `docs/future-spec.md` keeps `dinov2` under future backbone roadmap
  - `README.md` states `dinov2` is not yet fully wired as a current formal
    capability
- promotion work (builder wiring + runtime validation + tests) is intentionally
  deferred until there is an explicit future promotion track

## Phase 7: Minimum Trustworthy End-to-End Path

Goal:

- maintain at least one complete current-scope path that the team can trust

Tasks:

1. Ensure there is at least one e2e route covering:
   - config compose
   - model build
   - embedding export
   - evaluator run
   - checkpoint write
2. Prefer a fast dummy-backed path for repeatable validation

Completion criteria:

- the repository has one complete, repeatable, trustworthy current-scope e2e path

## Phase 8: Future-Spec Preparation

Goal:

- prepare for future work without prematurely claiming it

Tasks:

1. Refine future-spec details only after current scope is stable
2. Keep `round2_ssl`, `round3_supcon`, and richer resume work in future docs until code and tests land
3. Promote features only through the full path:
   - runtime
   - config
   - tests
   - docs

Completion criteria:

- future work has a stable launch point and does not destabilize current scope

## Priority Summary

### Completed

- Phase 0: governance alignment
- Phase 1: config/schema drift cleanup
- Phase 2: test drift cleanup
- Phase 3: acceptance/test matrix hardening
- Phase 4: Round1 runtime stabilization
- Phase 5: checkpoint/resume stabilization
- Phase 6: `dinov2` decision

### P0

- none

### P1

- Phase 7: minimum trustworthy e2e path

### P2

- Phase 8: future-spec preparation

## Execution Rule

Unless the user explicitly redirects work, implementation should proceed phase by phase from top to bottom.
