"""Runner for artifact-driven centroid evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from die_vfm.evaluator.centroid_evaluator import (
    CentroidEvaluationOutput,
    CentroidEvaluatorConfig,
    evaluate_centroid,
)
from die_vfm.evaluator.io import LinearProbeDataBundle, load_linear_probe_bundle
from die_vfm.evaluator.result_writer import write_centroid_outputs


@dataclass(frozen=True)
class CentroidInputConfig:
    """Input artifact configuration for a centroid run."""

    train_split_dir: str | Path
    val_split_dir: str | Path
    normalize_embeddings: bool = False
    map_location: str = "cpu"


@dataclass(frozen=True)
class CentroidOutputConfig:
    """Output configuration for a centroid run."""

    output_dir: str | Path
    save_predictions: bool = True


@dataclass(frozen=True)
class CentroidRunConfig:
    """Top-level config for a centroid evaluator run."""

    input: CentroidInputConfig
    output: CentroidOutputConfig
    evaluator: CentroidEvaluatorConfig


@dataclass(frozen=True)
class CentroidRunResult:
    """Final result for one centroid evaluator run."""

    output_dir: Path
    written_paths: dict[str, Path]
    bundle: LinearProbeDataBundle
    evaluation_output: CentroidEvaluationOutput

    @property
    def val_metrics(self) -> dict[str, float]:
        """Returns validation metrics for the query split."""
        return self.evaluation_output.metrics


def run_centroid(config: CentroidRunConfig) -> CentroidRunResult:
    """Runs artifact-driven centroid evaluation end to end.

    Args:
      config: Top-level run configuration.

    Returns:
      Final run result including loaded bundle, evaluation outputs, and written
      output paths.
    """
    bundle = load_linear_probe_bundle(
        train_split_dir=config.input.train_split_dir,
        val_split_dir=config.input.val_split_dir,
        normalize_embeddings=config.input.normalize_embeddings,
        map_location=config.input.map_location,
    )
    evaluation_output = evaluate_centroid(
        bundle=bundle,
        config=config.evaluator,
    )

    output_dir = Path(config.output.output_dir)
    written_paths = write_centroid_outputs(
        output_dir=output_dir,
        result=evaluation_output,
        bundle=bundle,
        config=_build_serializable_run_config(config),
        save_predictions=config.output.save_predictions,
    )
    return CentroidRunResult(
        output_dir=output_dir,
        written_paths=written_paths,
        bundle=bundle,
        evaluation_output=evaluation_output,
    )


def build_centroid_run_config(
    train_split_dir: str | Path,
    val_split_dir: str | Path,
    output_dir: str | Path,
    normalize_embeddings: bool = False,
    map_location: str = "cpu",
    save_predictions: bool = True,
    metric: str = "cosine",
    batch_size: int = 1024,
    device: str = "cpu",
    topk: tuple[int, ...] = (1, 5),
) -> CentroidRunConfig:
    """Builds a CentroidRunConfig from explicit arguments."""
    return CentroidRunConfig(
        input=CentroidInputConfig(
            train_split_dir=train_split_dir,
            val_split_dir=val_split_dir,
            normalize_embeddings=normalize_embeddings,
            map_location=map_location,
        ),
        output=CentroidOutputConfig(
            output_dir=output_dir,
            save_predictions=save_predictions,
        ),
        evaluator=CentroidEvaluatorConfig(
            metric=metric,
            batch_size=batch_size,
            device=device,
            topk=topk,
        ),
    )


def resolve_centroid_run_config(config: Any) -> CentroidRunConfig:
    """Converts a loose config object into a typed CentroidRunConfig.

    This helper is intentionally permissive so the runner can be called from:
    - typed dataclass configs
    - plain dicts
    - Hydra / OmegaConf-like objects after light extraction

    Args:
      config: Raw config object.

    Returns:
      A typed CentroidRunConfig.

    Raises:
      TypeError: If the config structure cannot be interpreted.
      KeyError: If required fields are missing.
    """
    if isinstance(config, CentroidRunConfig):
        return config

    data = _to_mapping(config)
    input_cfg = _to_mapping(data["input"])
    output_cfg = _to_mapping(data["output"])
    evaluator_cfg = _to_mapping(data["evaluator"])

    topk_value = evaluator_cfg.get("topk", (1, 5))
    if isinstance(topk_value, list):
        topk_value = tuple(topk_value)
    elif not isinstance(topk_value, tuple):
        topk_value = tuple(topk_value)

    return CentroidRunConfig(
        input=CentroidInputConfig(
            train_split_dir=input_cfg["train_split_dir"],
            val_split_dir=input_cfg["val_split_dir"],
            normalize_embeddings=input_cfg.get("normalize_embeddings", False),
            map_location=input_cfg.get("map_location", "cpu"),
        ),
        output=CentroidOutputConfig(
            output_dir=output_cfg["output_dir"],
            save_predictions=output_cfg.get("save_predictions", True),
        ),
        evaluator=CentroidEvaluatorConfig(
            metric=evaluator_cfg.get("metric", "cosine"),
            batch_size=evaluator_cfg.get("batch_size", 1024),
            device=evaluator_cfg.get("device", "cpu"),
            topk=topk_value,
        ),
    )


def _build_serializable_run_config(
    config: CentroidRunConfig,
) -> dict[str, Any]:
    """Builds a serializable config payload for result writing."""
    return {
        "input": {
            "train_split_dir": str(config.input.train_split_dir),
            "val_split_dir": str(config.input.val_split_dir),
            "normalize_embeddings": config.input.normalize_embeddings,
            "map_location": config.input.map_location,
        },
        "output": {
            "output_dir": str(config.output.output_dir),
            "save_predictions": config.output.save_predictions,
        },
        "evaluator": _evaluator_config_to_dict(config.evaluator),
    }


def _evaluator_config_to_dict(
    config: CentroidEvaluatorConfig,
) -> dict[str, Any]:
    """Converts evaluator config to a plain dict."""
    if is_dataclass(config):
        return asdict(config)
    return {
        "metric": config.metric,
        "batch_size": config.batch_size,
        "device": config.device,
        "topk": config.topk,
    }


def _to_mapping(value: Any) -> dict[str, Any]:
    """Converts a config-like object to a plain mapping."""
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "items"):
        try:
            return dict(value)
        except (TypeError, ValueError):
            pass
    if hasattr(value, "__dict__"):
        return dict(vars(value))
    raise TypeError(f"Unsupported config type: {type(value)!r}.")