# DIE-VFM Checkpoint / Resume Spec

## Purpose

This document defines the checkpoint and resume contract for the current repository scope. It intentionally describes the M1 implementation boundary rather than the richer future target.

## Current Scope

Checkpoint and resume are currently required for:

- `bootstrap`

The current implementation supports:

- checkpoint save
- atomic file replacement
- `warm_start`
- M1-style `full_resume`

## Current Checkpoint Set

Bootstrap runs currently write:

- `latest.pt`
- `best.pt` when applicable
- `epoch_xxxx.pt`

These files live under:

```text
runs/<run_name>/checkpoints/
```

## Save Semantics

Current required behavior:

- payload is written to a temporary file
- temporary file is atomically replaced into the target path
- each save updates `latest.pt`
- each save writes its own `epoch_xxxx.pt`
- `best.pt` is updated only when `is_best=True`; otherwise existing `best.pt` is preserved

This protects against partially written final checkpoint files under normal interruption scenarios.

## Current Payload Contract

The current payload shape is defined by runtime code and consists of:

```python
{
    "checkpoint_version": str,
    "epoch": int,
    "global_step": int,
    "model_state_dict": dict,
    "optimizer_state_dict": dict | None,
    "lr_scheduler_state_dict": dict | None,
    "grad_scaler_state_dict": dict | None,
    "trainer_state": dict,
    "metadata": dict,
}
```

Current validation requires:

- payload is a dict
- required top-level keys are present
- `checkpoint_version` matches supported version
- `epoch` is an int
- `global_step` is an int
- `model_state_dict` is a dict
- `optimizer_state_dict`, `lr_scheduler_state_dict`, and `grad_scaler_state_dict` are present and each is either a dict or `None`
- `trainer_state` is a dict
- `metadata` is a dict

## Current Resume Modes

### `warm_start`

Purpose:

- initialize model weights from an existing checkpoint

Behavior:

- loads checkpoint payload
- restores model state
- does not restore trainer progress
- does not require optimizer, scheduler, or scaler state

### `full_resume`

Purpose:

- continue a bootstrap-scope run using saved state

Behavior:

- loads checkpoint payload
- restores model state
- restores trainer state
- restores optimizer state when an optimizer object is provided
- restores scheduler state when a scheduler object is provided
- restores scaler state when a scaler object is provided

If an optimizer, scheduler, or scaler is requested during full resume but missing from the checkpoint, resume must fail explicitly.

## Current Path Resolution Rules

Checkpoint selection follows this order:

1. explicit `checkpoint_path`
2. `latest.pt` when `auto_resume_latest=true`
3. no resolved checkpoint

If an explicit `checkpoint_path` is provided and missing, resume fails.

## Current Logging Expectations

Current runtime should log:

- resolved checkpoint path
- resume mode
- restored epoch and global step when available
- written checkpoint paths after save

## Current Explicit Non-Guarantees

The current repository does not yet formally guarantee:

- RNG restoration
- EMA teacher restoration
- sampler or data-loader state restoration
- corrupted `latest.pt` fallback to prior periodic checkpoint
- full config compatibility checks
- `round1_frozen` checkpoint/resume semantics as part of the Round1 contract
- same-world-size continuity guarantees across future distributed modes

## Future Direction

Future rounds are expected to expand this contract to include:

- RNG state
- EMA teacher state
- compatibility enforcement
- richer metadata and lineage
- same-world-size full-resume guarantees for training-centric modes

Those capabilities are future-facing and should not be described as current formal behavior until code and tests land.
