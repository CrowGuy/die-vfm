"""CLI entrypoint for artifact-driven retrieval evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf

from die_vfm.evaluator.retrieval_runner import RetrievalRunConfig
from die_vfm.evaluator.retrieval_runner import resolve_retrieval_run_config
from die_vfm.evaluator.retrieval_runner import run_retrieval


def _extract_retrieval_config(config: DictConfig) -> DictConfig:
    """Extracts the evaluation.retrieval subtree from Hydra config."""
    if "evaluation" not in config:
        raise KeyError("Missing required Hydra config section: evaluation.")
    if "retrieval" not in config.evaluation:
        raise KeyError(
            "Missing required Hydra config section: evaluation.retrieval."
        )
    return config.evaluation.retrieval


def _is_enabled(retrieval_config: DictConfig) -> bool:
    """Returns whether retrieval evaluation is enabled."""
    return bool(retrieval_config.get("enabled", True))


def _validate_retrieval_config(retrieval_config: DictConfig) -> None:
    """Validates required retrieval config fields.

    Args:
      retrieval_config: The evaluation.retrieval config subtree.

    Raises:
      ValueError: If required config fields are missing.
    """
    input_config = retrieval_config.get("input")
    if input_config is None:
        raise ValueError(
            "Missing required config section: evaluation.retrieval.input."
        )

    output_config = retrieval_config.get("output")
    if output_config is None:
        raise ValueError(
            "Missing required config section: evaluation.retrieval.output."
        )

    train_split_dir = input_config.get("train_split_dir")
    if not train_split_dir:
        raise ValueError(
            "Missing required config: "
            "evaluation.retrieval.input.train_split_dir."
        )

    val_split_dir = input_config.get("val_split_dir")
    if not val_split_dir:
        raise ValueError(
            "Missing required config: "
            "evaluation.retrieval.input.val_split_dir."
        )

    output_dir = output_config.get("output_dir")
    if not output_dir:
        raise ValueError(
            "Missing required config: evaluation.retrieval.output.output_dir."
        )


def _to_plain_object(config: DictConfig) -> dict[str, Any]:
    """Converts an OmegaConf config to a plain Python dict."""
    plain_config = OmegaConf.to_container(
        config,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(plain_config, dict):
        raise TypeError(
            "Expected evaluation.retrieval to resolve to a dict, got "
            f"{type(plain_config)!r}."
        )
    return plain_config


def _format_metrics(metrics: dict[str, float]) -> str:
    """Formats a compact metric summary for stdout."""
    preferred_keys = (
        "recall_at_1",
        "recall_at_5",
        "map_at_1",
        "map_at_5",
    )
    parts = []
    for key in preferred_keys:
        if key in metrics:
            parts.append(f"{key}={metrics[key]:.6f}")

    if not parts:
        parts = [
            f"{key}={value:.6f}"
            for key, value in sorted(metrics.items())
        ]
    return ", ".join(parts)


def _print_run_summary(
    result_output_dir: Path,
    metrics: dict[str, float],
) -> None:
    """Prints a compact success summary."""
    print("Retrieval evaluation completed.")
    print(f"Output directory: {result_output_dir}")
    print(f"Query metrics: {_format_metrics(metrics)}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(config: DictConfig) -> None:
    """Runs retrieval evaluation from Hydra config."""
    retrieval_config = _extract_retrieval_config(config)

    if not _is_enabled(retrieval_config):
        print("Retrieval evaluation is disabled. Skipping.")
        return

    _validate_retrieval_config(retrieval_config)
    typed_config: RetrievalRunConfig = resolve_retrieval_run_config(
        _to_plain_object(retrieval_config)
    )
    result = run_retrieval(typed_config)
    _print_run_summary(
        result_output_dir=result.output_dir,
        metrics=result.val_metrics,
    )


if __name__ == "__main__":
    main()