# DIE-VFM Round2 Experiment Planning Draft

## Purpose

This document captures the current planning draft for `round2_ssl` after the
first `dinov2` Round1 domain baseline closeout.

This is a planning document, not a statement of implemented repository support.
It is intended to guide experiment design, data preparation, and success
criteria discussion before Round2 runtime/config/test promotion work begins.

Related references:
- [Current Spec](current-spec.md)
- [Future Spec](future-spec.md)
- [Implementation Roadmap](implementation-roadmap.md)
- [Domain Baseline Closeout](domain-baseline-closeout.md)
- [Round2 First Pilot Readiness Checklist](round2-first-pilot-readiness-checklist.md)

## Positioning

Current repository scope ends at:
- `bootstrap`
- `round1_frozen`
- artifact-driven frozen evaluation

Round2 remains future scope:
- `round2_ssl` is not yet a promoted runtime contract
- optimizer/scheduler/checkpoint/resume semantics for Round2 may exist in the
  experimental skeleton, but they are not yet promoted as current formal
  support

This draft should therefore be treated as:
- experiment planning input
- data strategy proposal
- method selection rationale
- success-criteria definition

It should not be treated as completed runtime support.

## Round1 Baseline Context

The current Round2 plan is anchored on the frozen Round1 baseline:
- baseline id: `domain_round1_dinov2_baseline_v1`
- backbone family: `dinov2`
- runtime mode: `round1_frozen`

Round1 closeout summary:
- train/reference embeddings: `109,945`
- val/query embeddings: `27,475`
- pair benchmark reviewed pairs: `874`
- pair benchmark `same_vs_different_cosine_auc_like = 0.8135`
- cross-source slice AUC-like: approximately `0.751`

Main Round1 findings:
- the baseline already contains useful visual signal
- cross-source invariance is the clearest persistent weak point
- fine-label structure does not cleanly match visual sameness

These findings define the Round2 problem statement.

## Round2 Objective

Primary objective:
- make `visually same` samples cluster more consistently in embedding space

Priority emphasis:
- visually same samples should stay close even when source changes
- source change includes `lot`, `machine`, and `time`

Operational interpretation:
- Round2 is not primarily a fine-label classification improvement project
- Round2 is primarily a representation adaptation stage for:
  - visual sameness
  - cross-source invariance
  - stronger downstream retrieval/clustering behavior
- the recommended formal-pilot execution flow is:
  - one distributed training run
  - followed by one single-process postprocess run for artifact export and offline evaluation

## Non-Goals

Round2 should not initially optimize for:
- best possible fine-label classification accuracy
- taxonomy cleanup through model behavior alone
- replacing the pair benchmark with label-only evaluators
- using noisy fine labels as the main supervision signal from day one

## Round2 Success Criteria

Primary metrics:
- pair benchmark `same_vs_different_cosine_auc_like`
- `cross_source` slice AUC-like

Hard-case diagnostics:
- fewer severe `hard_same_far` examples
- fewer severe `hard_different_close` examples

Guardrail metrics:
- `kNN` should not show material regression
- `retrieval` should not show material regression
- `linear_probe` should not show material regression

Interpretation rule:
- Round2 should be considered successful only if pair-based and cross-source
  metrics improve without obvious collapse in the existing Round1 evaluator
  suite

Suggested practical threshold for pilot interpretation:
- pair AUC-like: target `+0.02` or more versus Round1 baseline
- cross-source AUC-like: target `+0.03` or more versus Round1 baseline
- label-based evaluators: avoid regression larger than roughly `1-2` absolute
  points unless pair/cross-source gains are clearly compelling

## Core Hypotheses

### Hypothesis 1: Domain adaptation should improve visual stability

If `dinov2` is adapted on large-scale domain images with SSL, then embeddings
should better reflect domain-specific texture, background, noise, and pattern
statistics.

Expected benefit:
- stronger domain fit
- more stable representation under real manufacturing variation

Main uncertainty:
- pure SSL may improve domain fit without sufficiently improving cross-source
  sameness

### Hypothesis 2: Cross-source sameness needs explicit tightening pressure

If training only uses standard image-level SSL augmentation invariance, the
model may not automatically learn that different-source but visually same
samples should be close.

Expected implication:
- Round2 likely needs a second-stage mechanism that directly encourages
  high-confidence cross-source positive pairs to align

### Hypothesis 3: Noisy fine labels should remain secondary in early Round2

Because fine labels do not consistently map to visual sameness:
- same fine label can contain multiple visual modes
- different fine labels can still be visually similar

Expected implication:
- early Round2 should avoid making noisy fine labels the dominant training
  signal
- label-aware tightening is more appropriate later, after representation
  quality improves or cleaner subsets exist

## Data Strategy

### Data Role Separation

Round1 assets should not be reused as the sole Round2 training set.

Recommended role split:
- Round1 assets become fixed benchmark and comparison anchors
- Round2 uses a larger, separate train-only corpus built from the domain data

Use Round1 assets for:
- baseline comparison
- pair benchmark evaluation
- slicing analysis
- pseudo-positive mining seed experiments
- pilot smoke checks

Do not use Round1 assets as the only formal Round2 training corpus.

## Round2 Training Corpus Principles

The Round2 training corpus should:
- come from the same domain image universe
- exclude pair benchmark `did` values
- exclude Round1 val/query images
- control duplicate or near-duplicate inflation
- limit same-source overconcentration
- preserve meaningful tail coverage

Recommended controls:
- near-duplicate or duplicate control before main training
- same-source cap per `lot-machine-time` group
- source-aware sampling to avoid batches dominated by one source bucket
- retain long-tail classes instead of collapsing entirely to head classes

## Proposed Dataset Layers

### `round2_pilot_subset`

Purpose:
- fast iteration
- loss-function comparison
- batch-construction debugging
- mining strategy smoke tests

Suggested characteristics:
- materially larger than Round1 pilot
- still small enough for rapid reruns
- source-diverse
- de-duplicated and benchmark-excluded

### `round2_main_corpus`

Purpose:
- main SSL adaptation stage

Suggested characteristics:
- significantly larger than Round1 pilot
- benchmark-excluded
- duplicate-controlled
- source-balanced enough to expose cross-source variation during training

## Method Recommendation

### Why not start with noisy-label supervised training

The current data situation argues against label-dominant training at the start
of Round2:
- the main objective is visual sameness, not taxonomy imitation
- the label system is noisy with respect to that objective
- over-weighting fine labels too early may harden the wrong boundaries

Therefore the recommended Round2 order is:
1. representation adaptation first
2. explicit sameness tightening second
3. stronger label-aware tightening later, if needed

## Proposed Round2 Variants

### `Round2-A`: pure SSL adaptation baseline

Training idea:
- adapt `dinov2` on domain data using a DINO-style self-distillation SSL loss
  or equivalent teacher-student SSL objective

Main signal:
- unlabeled domain images
- no explicit pair supervision

Why this is the recommended first variant:
- lowest supervision risk
- most consistent with current `dinov2` family behavior
- establishes a clean Round2 baseline before adding pair-specific pressure

Expected strengths:
- stronger domain fit
- improved representation stability
- low dependence on noisy labels

Expected weakness:
- may not sufficiently improve cross-source same-sample closeness by itself

### `Round2-B`: SSL plus pseudo-positive tightening

Training idea:
- keep the same SSL objective as `Round2-A`
- add a secondary pair-aware alignment loss for high-confidence pseudo-positive
  pairs

Pseudo-positive source:
- mined from frozen Round1 embeddings
- prioritize cross-source nearest-neighbor agreement
- prefer high-confidence visually similar examples
- avoid trivial same-source near-duplicate pairs as the dominant positive type

Why this is the recommended main Round2 direction:
- directly targets the known weak point from Round1
- converts existing embedding signal into a more task-aligned training
  constraint
- remains less dependent on noisy fine labels than fully supervised training

Expected strengths:
- better cross-source alignment
- more direct improvement on pair benchmark metrics

Main risk:
- bad pseudo-positive mining can inject false attraction constraints

### `Round2-C`: source-aware training refinement

Training idea:
- keep `Round2-B` loss design
- improve batch composition and/or sampling policy so training repeatedly sees
  source variation

Possible controls:
- source-aware batch mixing
- same-source quota limits per batch
- deliberate cross-source positive exposure

Why this is valuable:
- it is a relatively low-risk way to strengthen invariance
- it targets the exact weak slice identified in Round1 analysis

Expected strengths:
- additional cross-source robustness
- reduced overfitting to local source artifacts

## Recommended Variant Order

Recommended execution order:
1. `Round2-A` as the minimum viable Round2 baseline
2. `Round2-B` as the first task-aligned improvement path
3. `Round2-C` as a refinement of `Round2-B`

Rationale:
- `Round2-A` tells us whether plain domain SSL already helps enough
- `Round2-B` tests whether explicit cross-source tightening is necessary
- `Round2-C` tests whether data presentation, not just objective design,
  remains a bottleneck

## Positive/Negative Construction Guidance

For early Round2 planning:
- `Round2-A` does not require explicit positives beyond SSL view generation
- `Round2-B` should use only high-confidence pseudo-positive pairs
- early pseudo-positives should emphasize cross-source visual agreement

Preferred early pseudo-positive candidates:
- mutual nearest neighbors across different source groups
- top-similarity neighbors that remain stable across embedding checkpoints
- reviewed pair benchmark patterns that inspire mining rules, but are not used
  directly as the only training source

Avoid:
- using all nearest neighbors as positives without filtering
- letting same-source near-duplicates dominate the positive set
- treating noisy fine-label equality as the default positive rule

## Evaluation Plan

All Round2 variants should be evaluated against the fixed Round1 benchmark
stack.

Required evaluation views:
- pair benchmark full summary
- pair benchmark slicing analysis
- `hard_same_far`
- `hard_different_close`
- `linear_probe`
- `kNN`
- `retrieval`

Evaluation comparison rule:
- every Round2 pilot result should be compared directly against the frozen
  Round1 baseline, not only against other Round2 variants

## Suggested Pilot Matrix

### Pilot 1: establish Round2 minimum baseline

Variant:
- `Round2-A`

Question:
- does plain domain SSL improve pair-based and cross-source metrics at all

Success interpretation:
- any clear uplift in pair benchmark or cross-source slice without major
  evaluator regression validates continuing Round2

### Pilot 2: test explicit cross-source tightening

Variant:
- `Round2-B`

Question:
- does pseudo-positive alignment materially outperform plain SSL on the target
  metrics

Success interpretation:
- pair AUC-like and cross-source slice improve beyond `Round2-A`

### Pilot 3: test source-aware training presentation

Variant:
- `Round2-C`

Question:
- does source-aware sampling further reduce the cross-source weakness

Success interpretation:
- cross-source slice improves again without destabilizing the rest of the
  benchmark stack

## Risks and Failure Modes

### Risk 1: pure SSL under-improves target behavior

Possible outcome:
- better domain fit
- weak improvement on pair benchmark

Implication:
- move quickly to pseudo-positive tightening

### Risk 2: pseudo-positive quality is not good enough

Possible outcome:
- false positives pull apart meaningful structure
- hard-case profile becomes unstable

Implication:
- add stricter mining thresholds
- require cross-source consistency checks
- reduce pseudo-positive volume in early pilots

### Risk 3: pair gains come with evaluator regression

Possible outcome:
- pair benchmark improves
- label-based evaluators drop materially

Implication:
- inspect whether the model is collapsing toward a narrow similarity notion
- rebalance loss weights or sampling policy

### Risk 4: duplicate inflation hides real progress

Possible outcome:
- training appears strong because the corpus contains too many trivial repeats

Implication:
- keep duplicate control explicit in data preparation
- inspect source and duplicate composition before trusting gains

## Open Planning Decisions

The following decisions still need explicit agreement before implementation:
- exact size target for `round2_pilot_subset`
- exact size target for `round2_main_corpus`
- duplicate-control rule definition
- pseudo-positive mining threshold policy
- trainable-module boundary for Round2
- whether the first Round2 pilot should update:
  - full backbone
  - selected late backbone blocks
  - pooler/projector only

## Recommended Next Steps

1. Finalize the Round2 pilot data-building rules.
2. Decide the initial trainable-module policy for `Round2-A`.
3. Define the pseudo-positive mining policy for `Round2-B`.
4. Convert this planning draft into:
   - runtime requirements
   - config contract proposal
   - test plan
5. Only then begin implementation work for promoting `round2_ssl` from future
   scope toward current support.
