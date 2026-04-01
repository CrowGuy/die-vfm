"""Filesystem writers for evaluator outputs."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from die_vfm.evaluator.io import LinearProbeDataBundle
from die_vfm.evaluator.linear_probe_trainer import LinearProbeTrainingResult


def write_linear_probe_outputs(
    output_dir: str | Path,
    result: LinearProbeTrainingResult,
    bundle: LinearProbeDataBundle,
    config: Any,
    save_predictions: bool = True,
    save_history: bool = True,
) -> dict[str, Path]:
    """Writes linear probe evaluation outputs to disk.

    Args:
        output_dir: Output directory for evaluator artifacts.
        result: Training result returned by train_linear_probe().
        bundle: Prepared train/val bundle used for evaluation.
        config: Config object or dict used for this evaluator run.
        save_predictions: Whether to write predictions.pt.
        save_history: Whether to write history.yaml.

    Returns:
        Mapping from artifact logical names to written paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    written_paths: dict[str, Path] = {}

    metrics_payload = build_linear_probe_metrics_payload(
        result=result,
        bundle=bundle,
    )
    metrics_path = output_path / "metrics.yaml"
    _write_yaml(metrics_path, metrics_payload)
    written_paths["metrics"] = metrics_path

    summary_payload = build_linear_probe_summary_payload(
        output_dir=output_path,
        result=result,
        bundle=bundle,
    )
    summary_path = output_path / "summary.yaml"
    _write_yaml(summary_path, summary_payload)
    written_paths["summary"] = summary_path

    config_payload = _to_serializable_config(config)
    config_path = output_path / "config.yaml"
    _write_yaml(config_path, config_payload)
    written_paths["config"] = config_path

    if save_history:
        history_payload = build_linear_probe_history_payload(result=result)
        history_path = output_path / "history.yaml"
        _write_yaml(history_path, history_payload)
        written_paths["history"] = history_path

    if save_predictions:
        predictions_payload = build_linear_probe_predictions_payload(
            result=result,
            bundle=bundle,
        )
        predictions_path = output_path / "predictions.pt"
        torch.save(predictions_payload, predictions_path)
        written_paths["predictions"] = predictions_path

    return written_paths


def build_linear_probe_metrics_payload(
    result: LinearProbeTrainingResult,
    bundle: LinearProbeDataBundle,
) -> dict[str, Any]:
    """Builds the stable metrics.yaml payload."""
    return {
        "evaluator_type": "linear_probe",
        "evaluator_version": "v1",
        "input": {
            "train_split": bundle.train.split_name,
            "val_split": bundle.val.split_name,
            "train_num_samples": bundle.train.num_samples,
            "val_num_samples": bundle.val.num_samples,
            "embedding_dim": bundle.embedding_dim,
            "num_classes": bundle.num_classes,
            "class_ids": list(bundle.class_ids),
        },
        "best_epoch": result.best_epoch,
        "train": {
            "loss": float(result.train_metrics["loss"]),
            "accuracy": float(result.train_metrics["accuracy"]),
        },
        "val": {
            "loss": float(result.val_metrics["loss"]),
            "accuracy": float(result.val_metrics["accuracy"]),
        },
    }


def build_linear_probe_history_payload(
    result: LinearProbeTrainingResult,
) -> dict[str, Any]:
    """Builds the history.yaml payload."""
    return {
        "epochs": [
            {
                "epoch": int(record["epoch"]),
                "train_loss": float(record["train_loss"]),
                "train_accuracy": float(record["train_accuracy"]),
                "val_loss": float(record["val_loss"]),
                "val_accuracy": float(record["val_accuracy"]),
            }
            for record in result.history
        ]
    }


def build_linear_probe_predictions_payload(
    result: LinearProbeTrainingResult,
    bundle: LinearProbeDataBundle,
) -> dict[str, Any]:
    """Builds the predictions.pt payload."""
    val_output = result.val_output

    if len(val_output.image_ids) != bundle.val.num_samples:
        raise ValueError(
            "Prediction image_ids length must match validation sample count. "
            f"Got image_ids={len(val_output.image_ids)}, "
            f"val_num_samples={bundle.val.num_samples}."
        )

    return {
        "split": bundle.val.split_name,
        "image_ids": list(val_output.image_ids),
        "labels": val_output.labels.clone(),
        "pred_labels": val_output.predictions.clone(),
        "logits": val_output.logits.clone(),
        "class_ids": torch.tensor(bundle.class_ids, dtype=torch.long),
    }


def build_linear_probe_summary_payload(
    output_dir: str | Path,
    result: LinearProbeTrainingResult,
    bundle: LinearProbeDataBundle,
) -> dict[str, Any]:
    """Builds the summary.yaml payload."""
    return {
        "status": "success",
        "evaluator": "linear_probe",
        "best_epoch": result.best_epoch,
        "train_split": bundle.train.split_name,
        "val_split": bundle.val.split_name,
        "train_num_samples": bundle.train.num_samples,
        "val_num_samples": bundle.val.num_samples,
        "embedding_dim": bundle.embedding_dim,
        "num_classes": bundle.num_classes,
        "val_accuracy": float(result.val_metrics["accuracy"]),
        "val_loss": float(result.val_metrics["loss"]),
        "output_dir": str(Path(output_dir)),
    }


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    """Writes a YAML payload to disk."""
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            payload,
            file,
            sort_keys=False,
            allow_unicode=True,
        )


def _to_serializable_config(config: Any) -> dict[str, Any]:
    """Converts a config object into a YAML-serializable dict."""
    if config is None:
        return {}

    if isinstance(config, dict):
        return _to_builtin_types(config)

    if is_dataclass(config):
        return _to_builtin_types(asdict(config))

    if hasattr(config, "items"):
        try:
            return _to_builtin_types(dict(config))
        except (TypeError, ValueError):
            pass

    if hasattr(config, "__dict__"):
        return _to_builtin_types(vars(config))

    raise TypeError(
        "config must be serializable from dict, dataclass, or __dict__. "
        f"Got type={type(config)!r}."
    )


def _to_builtin_types(value: Any) -> Any:
    """Recursively converts values into YAML-friendly builtins."""
    if isinstance(value, dict):
        return {str(key): _to_builtin_types(subvalue) for key, subvalue in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_builtin_types(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.device):
        return str(value)

    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value.item()
        return value.tolist()

    return value