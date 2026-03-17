# Die VFM

Die-level visual foundation model platform.

This repository skeleton provides:

- Config composition via Hydra
- Run directory creation
- Logging
- Training entry point bootstrap
- Basic tests

---

## PR-2 Scope

PR-2 introduces the **dataset adapter interface and dataloader pipeline**.

New functionality includes:

- Dataset adapter base interface
- Dummy dataset implementation
- Dataset builder
- Dataloader builder
- Dataset metadata logging and artifact
- Train entrypoint dataloader smoke test
- Dataset unit tests

## Repository Structure
```text
die_vfm/
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── config.yaml
│   ├── dataset/
│   │   └── dummy.yaml
│   ├── experiment/
│   │   └── round1_frozen.yaml
│   ├── model/
│   │   ├── backbone/
│   │   │   └── dinov2.yaml
│   │   └── pooler/
│   │       ├── attn_pooler_v1.yaml
│   │       └── mean_pooler.yaml
├── scripts/
│   └── train.py
├── die_vfm/
│   ├── datasets/
│   │   ├── base.py
│   │   ├── builder.py
│   │   └── dummy_dataset.py
│   ├── utils/
│   ├── trainer/
│   ├── models/
│   ├── evaluator/
│   └── artifacts/
└── tests/
    ├── test_config.py
    ├── test_dummy_dataset.py
    ├── test_train_bootstrap.py
```

## Dataset Sample Contract

Each dataset adapter must return samples with the following structure:

```python
{
    "image": Tensor,        # shape: [C, H, W]
    "label": int | None,
    "image_id": str,
    "meta": dict
}
```
Example:
```python
{
    "image": tensor(3,224,224),
    "label": 0,
    "image_id": "train_00000",
    "meta": {
        "split": "train",
        "index": 0,
        "source": "dummy"
    }
}
```
---
## Batch Contract

The dataloader collates dataset samples into batches:

```python
{
    "image": Tensor,        # shape: [B, C, H, W]
    "label": Tensor | None, # shape: [B]
    "image_id": list[str],
    "meta": list[dict],
}
```
---
## Dataloader Smoke Test
PR-2 introduces a dataloader smoke test in `train.py`.

Run:

```bash
python scripts/train.py system.num_workers=0 train.run_dataloader_smoke_test=true
```
Expected output:
```text
Dataset metadata: {...}
Dataloader smoke test passed.
Batch image shape: (4, 3, 224, 224)
Batch label shape: (4,)
Batch image ids: ['train_00000', ...]
Training bootstrap completed successfully.
```
---
## Run Artifacts

During training bootstrap the following artifacts are generated:
```text
runs/<run_name>/
├── config.yaml
├── dataset_metadata.yaml
└── logs/
└── run.log
```
`dataset_metadata.yaml` contains dataset-level information such as:
- dataset_name
- split
- num_samples
- num_classes
---

## Quick Start

### Install

```bash
pip install -e .[dev]
```
### Run
```bash
python scripts/train.py
```

### Run with overrides
```bash
python scripts/train.py run.run_name=local_debug
```

### Test
Run all tests:
```bash
pytest
```
- `tests/test_config.py` — config composition
- `tests/test_dummy_dataset.py` — dataset adapter contract
- `tests/test_train_bootstrap.py` — end-to-end bootstrap smoke test

### Dataloader smoke test

```bash
python scripts/train.py system.num_workers=0
```
---

## 3. `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*.so

# Packaging
*.egg-info/
build/
dist/

# Virtual env
.venv/
venv/

# Testing
.pytest_cache/
.mypy_cache/
.ruff_cache/
.coverage

# IDE
.vscode/
.idea/

# Hydra / outputs
outputs/
multirun/

# Project runs
runs/

# Logs
*.log
```