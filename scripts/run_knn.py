"""CLI entrypoint for artifact-driven kNN evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from die_vfm.evaluator.knn_runner import resolve_knn_run_config, run_knn


def _get_knn_config(cfg: DictConfig) -> DictConfig:
    """Returns the kNN evaluation config subtree.

    Args:
      cfg: Root Hydra config.

    Returns:
      The evaluation.knn config subtree.

    Raises:
      ValueError: If evaluation.knn is missing.
    """
    evaluation_cfg = cfg.get("evaluation")
    if evaluation_cfg is None:
        raise ValueError("Missing required config section: evaluation.")

    knn_cfg = evaluation_cfg.get("knn")
    if knn_cfg is None:
        raise ValueError("Missing required config section: evaluation.knn.")

    return knn_cfg


def _is_enabled(knn_cfg: DictConfig) -> bool:
    """Returns whether the kNN evaluator is enabled."""
    return bool(knn_cfg.get("enabled", True))


def _validate_knn_config(knn_cfg: DictConfig) -> None:
    """Validates required fields for the kNN script.

    Args:
      knn_cfg: The evaluation.knn config subtree.

    Raises:
      ValueError: If required config fields are missing.
    """
    input_cfg = knn_cfg.get("input")
    if input_cfg is None:
        raise ValueError("Missing required config section: evaluation.knn.input.")

    output_cfg = knn_cfg.get("output")
    if output_cfg is None:
        raise ValueError("Missing required config section: evaluation.knn.output.")

    train_split_dir = input_cfg.get("train_split_dir")
    if not train_split_dir:
        raise ValueError(
            "Missing required config: evaluation.knn.input.train_split_dir."
        )

    val_split_dir = input_cfg.get("val_split_dir")
    if not val_split_dir:
        raise ValueError(
            "Missing required config: evaluation.knn.input.val_split_dir."
        )

    output_dir = output_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("Missing required config: evaluation.knn.output.output_dir.")


def _to_plain_config(knn_cfg: DictConfig) -> dict[str, Any]:
    """Converts Hydra config to a plain Python dict."""
    plain_cfg = OmegaConf.to_container(knn_cfg, resolve=True)
    if not isinstance(plain_cfg, dict):
        raise TypeError(
            "evaluation.knn must resolve to a mapping, "
            f"got {type(plain_cfg)!r}."
        )
    return plain_cfg


def _print_run_summary(output_dir: Path, metrics: dict[str, float]) -> None:
    """Prints a compact summary for the completed run."""
    print("kNN evaluation completed.")
    print(f"Output directory: {output_dir}")

    accuracy = metrics.get("accuracy")
    if accuracy is not None:
        print(f"Validation accuracy: {accuracy:.6f}")

    top1_accuracy = metrics.get("top1_accuracy")
    if top1_accuracy is not None:
        print(f"Validation top-1 accuracy: {top1_accuracy:.6f}")

    top5_accuracy = metrics.get("top5_accuracy")
    if top5_accuracy is not None:
        print(f"Validation top-5 accuracy: {top5_accuracy:.6f}")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Runs artifact-driven kNN evaluation from Hydra config."""
    knn_cfg = _get_knn_config(cfg)

    if not _is_enabled(knn_cfg):
        print("kNN evaluation is disabled. Skipping.")
        return

    _validate_knn_config(knn_cfg)

    plain_knn_cfg = _to_plain_config(knn_cfg)
    run_config = resolve_knn_run_config(plain_knn_cfg)
    result = run_knn(run_config)

    _print_run_summary(
        output_dir=result.output_dir,
        metrics=result.val_metrics,
    )


if __name__ == "__main__":
    main()
