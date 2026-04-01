"""CLI entrypoint for artifact-driven linear probe evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from hydra.core.hydra_config import HydraConfig

from die_vfm.evaluator.linear_probe_runner import (
    build_linear_probe_run_config,
    run_linear_probe,
)


def _require_path(value: Any, field_name: str) -> Path:
    """Validates and converts a required path-like config field."""
    if value is None:
        raise ValueError(f"{field_name} must be set.")
    if not isinstance(value, (str, Path)):
        raise TypeError(
            f"{field_name} must be a string or Path, got {type(value)!r}."
        )

    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(
            f"{field_name} does not exist: {path}"
        )
    return path


def _require_output_dir(value: Any, field_name: str) -> Path:
    """Validates and converts an output directory config field."""
    if value is None:
        raise ValueError(f"{field_name} must be set.")
    if not isinstance(value, (str, Path)):
        raise TypeError(
            f"{field_name} must be a string or Path, got {type(value)!r}."
        )
    return Path(value)


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs linear probe evaluation from embedding artifacts."""
    if not cfg.evaluation.run_linear_probe:
        print("evaluation.run_linear_probe is false; nothing to do.")
        return

    linear_probe_cfg = cfg.evaluation.linear_probe

    train_split_dir = _require_path(
        linear_probe_cfg.input.train_split_dir,
        "evaluation.linear_probe.input.train_split_dir",
    )
    val_split_dir = _require_path(
        linear_probe_cfg.input.val_split_dir,
        "evaluation.linear_probe.input.val_split_dir",
    )
    output_dir = _require_output_dir(
        linear_probe_cfg.output.output_dir,
        "evaluation.linear_probe.output.output_dir",
    )

    run_config = build_linear_probe_run_config(
        train_split_dir=train_split_dir,
        val_split_dir=val_split_dir,
        output_dir=output_dir,
        normalize_embeddings=linear_probe_cfg.input.normalize_embeddings,
        map_location=linear_probe_cfg.input.map_location,
        bias=linear_probe_cfg.model.bias,
        batch_size=linear_probe_cfg.trainer.batch_size,
        num_epochs=linear_probe_cfg.trainer.num_epochs,
        learning_rate=linear_probe_cfg.trainer.learning_rate,
        weight_decay=linear_probe_cfg.trainer.weight_decay,
        optimizer_name=linear_probe_cfg.trainer.optimizer_name,
        momentum=linear_probe_cfg.trainer.momentum,
        device=linear_probe_cfg.trainer.device,
        seed=linear_probe_cfg.trainer.seed,
        selection_metric=linear_probe_cfg.trainer.selection_metric,
        save_predictions=linear_probe_cfg.output.save_predictions,
        save_history=linear_probe_cfg.output.save_history,
    )

    result = run_linear_probe(run_config)

    print("Linear probe evaluation completed successfully.")
    print(f"Output directory: {result.output_dir}")
    print(f"Best epoch: {result.best_epoch}")
    print(
        "Validation metrics: "
        f"loss={result.val_metrics['loss']:.6f}, "
        f"accuracy={result.val_metrics['accuracy']:.6f}"
    )
    print("Written files:")
    for name, path in sorted(result.written_paths.items()):
        print(f"  - {name}: {path}")

    hydra_output_dir = Path(HydraConfig.get().runtime.output_dir)
    hydra_output_dir.mkdir(parents=True, exist_ok=True)

    hydra_config_path = hydra_output_dir / "hydra_linear_probe_config.yaml"
    with hydra_config_path.open("w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg, resolve=True))

    print(f"Resolved Hydra config saved to: {hydra_config_path}")

if __name__ == "__main__":
    main()