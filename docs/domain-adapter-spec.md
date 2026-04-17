# DIE-VFM Domain Dataset Adapter Spec (Current v1.0)

## Purpose

This document defines the current formal specification for integrating domain
data into DIE-VFM while preserving current M1 / Round1 runtime boundaries.

This is a dataset-ingestion contract for current-scope runtime flows
(`bootstrap` and `round1_frozen`), not a training-stage design contract.

## Scope

In scope for this spec:

- CSV-manifest-driven domain dataset ingestion
- deterministic image loading and preprocessing
- optional dual-image channel merge behavior
- explicit label-to-int mapping contract
- adapter output alignment to the current dataset sample contract

Out of scope for this spec:

- data augmentation policy
- class-balanced batching or custom sampler policy
- pseudo-labeling
- SSL multi-view augmentation
- `round2_ssl` / `round3_supcon` training semantics

## Architecture

Domain dataset integration follows a two-layer design:

1. `DomainVisionDataset` (`torchvision.datasets.VisionDataset`-style)
   - reads and validates CSV rows
   - resolves split membership
   - loads image(s)
   - performs optional merge composition
2. `DomainDatasetAdapter` (`die_vfm.datasets.base.DatasetAdapter`)
   - maps dataset rows to the repository sample contract:
     - `image`
     - `label`
     - `image_id`
     - `meta`

This keeps `VisionDataset`-level loading concerns separate from DIE-VFM
contract enforcement.

## Manifest Contract

### Format

- single CSV manifest with a header row
- CSV should be read with `pandas.read_csv`
- empty-string normalization before validation:
  - `IMG_2="" -> None`
  - `Label="" -> None`
- manifest validation is dataset-initialization-time validation, not lazy
  sample-time validation

### Columns

Required columns:

- `DID`
- `IMG_1`
- `Source`
- `PATH`

Optional columns:

- `IMG_2`
- `Label`

### Column Semantics

- `DID`
  - globally unique within the manifest
  - primary sample identifier
  - default source for `image_id`
- `IMG_1`
  - primary image filename
  - required
- `IMG_2`
  - secondary image filename
  - optional
- `Source`
  - allowed values: `Train`, `Infer`
- `Label`
  - raw annotation
  - optional
- `PATH`
  - directory path containing image file(s)
  - must be an absolute path

### Split Mapping

- `Source=Train` -> runtime split `train`
- `Source=Infer` -> runtime split `val`

This mapping is a current runtime compatibility rule and does not imply that
`Infer` data is semantically equivalent to a traditional validation set.

### Split Label Presence Rule

- `train` must not mix labeled and unlabeled samples in the same filtered split
- `val` must not mix labeled and unlabeled samples in the same filtered split
  under the current embedding artifact contract
- inference/export use cases must use fully labeled or fully unlabeled `val`
  subsets per run

## Config Surface

The domain adapter config surface is the externally configurable behavior
boundary for data ingestion.

Current fields:

- `dataset.name`
- `dataset.manifest_path`
- `dataset.image_size`
- `dataset.merge_images`
- `dataset.single_image_source` (`img1` or `img2`)
- `dataset.require_non_empty_val` (enable inference-only non-empty `val` policy)
- `dataset.did_field`
- `dataset.img1_field`
- `dataset.img2_field`
- `dataset.source_field`
- `dataset.label_field`
- `dataset.path_field`
- `dataset.normalize.mean`
- `dataset.normalize.std`
- `dataset.label_map` (`dict[str, int]`)

## Image Loading and Composition

### Default Single-Image Behavior

- deterministic preprocessing only:
  - resize
  - tensor conversion
  - normalization
- no augmentation policy is defined in this spec
- single-image mode must still produce a 3-channel tensor output
- single-image loading is normalized into RGB-compatible output before tensor
  conversion

### Single-Image Selection Rule

If single-image mode is active:

- `single_image_source=img1` -> use `IMG_1`
- `single_image_source=img2`:
  - use `IMG_2` when present
  - fallback to `IMG_1` when `IMG_2` is missing

### Merge Mode Rule

Merge is used only when all conditions are true:

- `merge_images=true`
- `IMG_1` exists
- `IMG_2` exists

Merge behavior:

- load both images via PIL
- convert both to grayscale `L`
- compose into RGB:
  - `R = IMG_1`
  - `G = IMG_2`
  - `B = black`

If merge conditions are not met, fallback to single-image behavior.

If merge is attempted and image shapes are incompatible, fail fast.

## Label Mapping Contract

### Raw vs Runtime Label

- `Label` remains preserved in `meta.raw_label`
- runtime `label` must be `int | None`

### Label Canonicalization Rule

Before label lookup or validation:

- every non-empty `Label` must be canonicalized to string form
- canonicalization must strip leading and trailing whitespace
- canonicalized labels must be lowercased
- label comparison is case-insensitive
- numeric-like labels must collapse to the same canonical value
- numeric-like labels include:
  - numeric manifest values
  - string labels parseable as numeric values
- if a numeric-like label is mathematically integral, its canonical form must be
  the plain integer string
- `1`
- `"1"`
- `1.0`
  - `"1.0"`
  - `"01"`
  - `"001.000"`
  - all normalize to `"1"`
- if a numeric-like label is not integral, it must normalize to a minimal
  lowercase decimal string without presentation-only trailing zeros:
  - `"1.50"` -> `"1.5"`
- non-numeric labels keep their canonicalized lowercase string form after
  trimming
- manifest label vocabulary is treated as a sorted set of canonicalized
  label strings

### Mapping Rule

- `dataset.label_map` lookup must operate on canonicalized label strings
- `dataset.label_map` keys follow the same canonicalization rule
- when `Label` is non-empty, it must exist in `dataset.label_map`
- mapped value is used as runtime `label` (int)
- when `Label` is empty, runtime `label=None`
- unknown labels fail fast

Runtime must not derive label indices implicitly from manifest order or
subset-dependent class discovery.

## Adapter Output Contract

Each sample must satisfy:

```python
{
    "image": Tensor,
    "label": int | None,
    "image_id": str,
    "meta": dict,
}
```

Recommended `meta` minimum fields:

- `did`
- `source`
- `img_1`
- `img_2` (or `None`)
- `path`
- `raw_label` (or `None`)
- `merge_images` (bool)
- `selected_image_source` (`img1`, `img2`, or `merged`)

## Dataset Metadata Contract

`get_dataset_metadata()` provides at least:

- `dataset_name`
- `split`
- `num_samples`
- `image_size`
- `num_channels`
- `has_labels`
- `manifest_path`
- `source_values`
- `label_vocabulary`
- `merge_images`
- `single_image_source`
- `require_non_empty_val`

## Runtime Integration Rules

Domain adapter integration must preserve current runtime diagnosability.

- dataset initialization performs whole-manifest validation before runtime
  iteration
- filtered `train` split with zero samples must fail fast
- `bootstrap` requires non-empty `train`
- `round1_frozen` requires at least one non-empty split among `train` and `val`
- `round1_frozen` may export embeddings from only `train`, only `val`, or both,
  depending on which filtered splits are non-empty
- in inference-only workflows, `val` must be non-empty; this policy is enabled
  via `dataset.require_non_empty_val=true`
- in training-oriented workflows, `val` may be empty

## Failure Handling

Current runtime fail-fast boundaries include:

- missing required manifest columns
- invalid `Source` value
- empty `DID`
- non-absolute `PATH`
- empty `PATH`
- empty `IMG_1`
- missing file path(s)
- missing manifest path
- manifest path that is not a file
- duplicate `DID`
- filtered `train` split is empty
- filtered `val` split is empty when `dataset.require_non_empty_val=true`
- mixed labeled and unlabeled samples within `train`
- mixed labeled and unlabeled samples within `val` under current artifact
  contract
- merge-shape mismatch when merge is active
- invalid config guard values:
  - `single_image_source` not in `{img1, img2}`
  - invalid `image_size` length
  - invalid `normalize.mean`/`normalize.std` length
- non-empty `Label` not found in `label_map`
- invalid `label_map` config shape:
  - non-mapping input
  - empty key after canonicalization
  - non-int or bool value
  - conflicting canonicalized keys mapped to different values

## Dependency Decision

This spec assumes `pandas` is a formal runtime dependency for manifest parsing.

## Promotion Note

This document defines ingestion behavior only. Any future move into
training-centric semantics (augmentation policy, balanced sampling, or
adaptation-stage logic) must be promoted separately through runtime, config,
tests, and docs.
