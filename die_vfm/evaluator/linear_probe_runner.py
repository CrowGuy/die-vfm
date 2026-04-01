"""Runner for artifact-driven linear probe evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from die_vfm.evaluator.io import LinearProbeDataBundle, load_linear_probe_bundle
from die_vfm.evaluator.linear_probe import LinearProbeClassifier, build_linear_probe
from die_vfm.evaluator.linear_probe_trainer import (
    LinearProbeTrainerConfig,
    LinearProbeTrainingResult,
    train_linear_probe,
)
from die_vfm.evaluator.result_writer import write_linear_probe_outputs


@dataclass(frozen=True)
class LinearProbeInputConfig:
    """Input artifact configuration for a linear probe run."""

    train_split_dir: str | Path
    val_split_dir: str | Path
    normalize_embeddings: bool = False
    map_location: str = "cpu"


@dataclass(frozen=True)
class LinearProbeOutputConfig:
    """Output configuration for a linear probe run."""

    output_dir: str | Path
    save_predictions: bool = True
    save_history: bool = True


@dataclass(frozen=True)
class LinearProbeModelConfig:
    """Model configuration for a linear probe run."""

    bias: bool = True


@dataclass(frozen=True)
class LinearProbeRunConfig:
    """Top-level config for a linear probe evaluator run."""

    input: LinearProbeInputConfig
    output: LinearProbeOutputConfig
    model: LinearProbeModelConfig
    trainer: LinearProbeTrainerConfig


@dataclass(frozen=True)
class LinearProbeRunResult:
    """Final result for one linear probe evaluator run."""

    output_dir: Path
    written_paths: dict[str, Path]
    bundle: LinearProbeDataBundle
    model: LinearProbeClassifier
    training_result: LinearProbeTrainingResult

    @property
    def best_epoch(self) -> int:
        """Returns the best epoch selected during training."""
        return self.training_result.best_epoch

    @property
    def train_metrics(self) -> dict[str, float]:
        """Returns train metrics associated with the selected checkpoint."""
        return self.training_result.train_metrics

    @property
    def val_metrics(self) -> dict[str, float]:
        """Returns validation metrics associated with the selected checkpoint."""
        return self.training_result.val_metrics


def run_linear_probe(
    config: LinearProbeRunConfig,
) -> LinearProbeRunResult:
    """Runs artifact-driven linear probe evaluation end to end.

    Args:
        config: Top-level run configuration.

    Returns:
        Final run result including loaded bundle, trained model, and output paths.
    """
    bundle = load_linear_probe_bundle(
        train_split_dir=config.input.train_split_dir,
        val_split_dir=config.input.val_split_dir,
        normalize_embeddings=config.input.normalize_embeddings,
        map_location=config.input.map_location,
    )

    model = build_linear_probe(
        input_dim=bundle.embedding_dim,
        num_classes=bundle.num_classes,
        bias=config.model.bias,
    )

    training_result = train_linear_probe(
        model=model,
        bundle=bundle,
        config=config.trainer,
    )

    output_dir = Path(config.output.output_dir)
    written_paths = write_linear_probe_outputs(
        output_dir=output_dir,
        result=training_result,
        bundle=bundle,
        config=_build_serializable_run_config(config),
        save_predictions=config.output.save_predictions,
        save_history=config.output.save_history,
    )

    return LinearProbeRunResult(
        output_dir=output_dir,
        written_paths=written_paths,
        bundle=bundle,
        model=model,
        training_result=training_result,
    )


def build_linear_probe_run_config(
    train_split_dir: str | Path,
    val_split_dir: str | Path,
    output_dir: str | Path,
    normalize_embeddings: bool = False,
    map_location: str = "cpu",
    bias: bool = True,
    batch_size: int = 256,
    num_epochs: int = 50,
    learning_rate: float = 1e-2,
    weight_decay: float = 0.0,
    optimizer_name: str = "sgd",
    momentum: float = 0.9,
    device: str = "cpu",
    seed: int = 0,
    selection_metric: str = "val_accuracy",
    save_predictions: bool = True,
    save_history: bool = True,
) -> LinearProbeRunConfig:
    """Builds a LinearProbeRunConfig from explicit arguments."""
    return LinearProbeRunConfig(
        input=LinearProbeInputConfig(
            train_split_dir=train_split_dir,
            val_split_dir=val_split_dir,
            normalize_embeddings=normalize_embeddings,
            map_location=map_location,
        ),
        output=LinearProbeOutputConfig(
            output_dir=output_dir,
            save_predictions=save_predictions,
            save_history=save_history,
        ),
        model=LinearProbeModelConfig(
            bias=bias,
        ),
        trainer=LinearProbeTrainerConfig(
            batch_size=batch_size,
            num_epochs=num_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            momentum=momentum,
            device=device,
            seed=seed,
            selection_metric=selection_metric,
        ),
    )


def resolve_linear_probe_run_config(config: Any) -> LinearProbeRunConfig:
    """Converts a loose config object into a typed LinearProbeRunConfig.

    This helper is intentionally permissive so the runner can be called from:
    - typed dataclass configs
    - plain dicts
    - Hydra / OmegaConf-like objects after light extraction

    Args:
        config: Raw config object.

    Returns:
        A typed LinearProbeRunConfig.

    Raises:
        TypeError: If the config structure cannot be interpreted.
        KeyError: If required fields are missing.
    """
    if isinstance(config, LinearProbeRunConfig):
        return config

    data = _to_mapping(config)

    input_cfg = _to_mapping(data["input"])
    output_cfg = _to_mapping(data["output"])
    model_cfg = _to_mapping(data["model"])
    trainer_cfg = _to_mapping(data["trainer"])

    return LinearProbeRunConfig(
        input=LinearProbeInputConfig(
            train_split_dir=input_cfg["train_split_dir"],
            val_split_dir=input_cfg["val_split_dir"],
            normalize_embeddings=input_cfg.get("normalize_embeddings", False),
            map_location=input_cfg.get("map_location", "cpu"),
        ),
        output=LinearProbeOutputConfig(
            output_dir=output_cfg["output_dir"],
            save_predictions=output_cfg.get("save_predictions", True),
            save_history=output_cfg.get("save_history", True),
        ),
        model=LinearProbeModelConfig(
            bias=model_cfg.get("bias", True),
        ),
        trainer=LinearProbeTrainerConfig(
            batch_size=trainer_cfg.get("batch_size", 256),
            num_epochs=trainer_cfg.get("num_epochs", 50),
            learning_rate=trainer_cfg.get("learning_rate", 1e-2),
            weight_decay=trainer_cfg.get("weight_decay", 0.0),
            optimizer_name=trainer_cfg.get("optimizer_name", "sgd"),
            momentum=trainer_cfg.get("momentum", 0.9),
            device=trainer_cfg.get("device", "cpu"),
            seed=trainer_cfg.get("seed", 0),
            selection_metric=trainer_cfg.get("selection_metric", "val_accuracy"),
        ),
    )


def _build_serializable_run_config(
    config: LinearProbeRunConfig,
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
            "save_history": config.output.save_history,
        },
        "model": {
            "bias": config.model.bias,
        },
        "trainer": _trainer_config_to_dict(config.trainer),
    }


def _trainer_config_to_dict(config: LinearProbeTrainerConfig) -> dict[str, Any]:
    """Converts trainer config to a plain dict."""
    if is_dataclass(config):
        return asdict(config)
    return {
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "optimizer_name": config.optimizer_name,
        "momentum": config.momentum,
        "device": config.device,
        "seed": config.seed,
        "selection_metric": config.selection_metric,
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