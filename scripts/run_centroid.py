"""CLI entrypoint for artifact-driven centroid evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from die_vfm.evaluator.centroid_runner import (
    CentroidRunConfig,
    resolve_centroid_run_config,
    run_centroid,
)


def _extract_centroid_config(config: DictConfig) -> DictConfig:
    """Extracts the evaluation.centroid subtree from Hydra config."""
    if "evaluation" not in config:
        raise KeyError("Missing required Hydra config section: evaluation.")
    if "centroid" not in config.evaluation:
        raise KeyError(
            "Missing required Hydra config section: evaluation.centroid."
        )
    return config.evaluation.centroid


def _to_plain_object(config: DictConfig) -> dict[str, Any]:
    """Converts an OmegaConf config to a plain Python dict."""
    plain_config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(plain_config, dict):
        raise TypeError(
            "Expected evaluation.centroid to resolve to a dict, got "
            f"{type(plain_config)!r}."
        )
    return plain_config


def _format_metrics(metrics: dict[str, float]) -> str:
    """Formats a compact metric summary for stdout."""
    preferred_keys = (
        "accuracy",
        "top1_accuracy",
        "top5_accuracy",
    )
    parts = []
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6f}")
    if not parts:
        parts = [f"{key}={value:.6f}" for key, value in sorted(metrics.items())]
    return ", ".join(parts)


def _print_run_summary(result_output_dir: Path, metrics: dict[str, float]) -> None:
    """Prints a compact success summary."""
    print("Centroid evaluation completed.")
    print(f"Output directory: {result_output_dir}")
    print(f"Validation metrics: {_format_metrics(metrics)}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Runs centroid evaluation from Hydra config."""
    centroid_config = _extract_centroid_config(config)
    typed_config: CentroidRunConfig = resolve_centroid_run_config(
        _to_plain_object(centroid_config)
    )
    result = run_centroid(typed_config)
    _print_run_summary(
        result_output_dir=result.output_dir,
        metrics=result.val_metrics,
    )


if __name__ == "__main__":
    main()