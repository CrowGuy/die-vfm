# Die VFM

Die-level visual foundation model platform.

## PR-1 Scope

This repository skeleton provides:

- Config composition via Hydra
- Run directory creation
- Logging
- Training entry point bootstrap
- Basic tests

```text
die_vfm/
├── pyproject.toml
├── README.md
├── .gitignore
├── configs/
│   ├── config.yaml
│   ├── experiment/
│   │   └── round1_frozen.yaml
│   ├── model/
│   │   ├── backbone/
│   │   │   └── dinov2.yaml
│   │   └── pooler/
│   │       ├── attn_pooler_v1.yaml
│   │       └── mean_pooler.yaml
│   ├── train/
│   │   └── default.yaml
│   └── eval/
│       └── default.yaml
├── scripts/
│   └── train.py
├── die_vfm/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── schema.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_utils.py
│   │   ├── run_dir.py
│   │   └── seed.py
│   ├── trainer/
│   │   ├── __init__.py
│   │   └── base_trainer.py
│   ├── models/
│   │   └── __init__.py
│   ├── evaluator/
│   │   └── __init__.py
│   └── artifacts/
│       └── __init__.py
└── tests/
    ├── test_config.py
    ├── test_run_dir.py
    └── test_train_bootstrap.py
```

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
```bash
pytest
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