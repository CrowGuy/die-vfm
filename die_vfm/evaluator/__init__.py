"""Evaluator utilities for artifact-driven model evaluation."""

from die_vfm.evaluator.io import load_linear_probe_bundle
from die_vfm.evaluator.linear_probe import LinearProbeClassifier, build_linear_probe
from die_vfm.evaluator.linear_probe_runner import run_linear_probe
from die_vfm.evaluator.linear_probe_trainer import train_linear_probe
from die_vfm.evaluator.result_writer import write_linear_probe_outputs

from die_vfm.evaluator.knn_evaluator import (
    KnnEvaluationOutput,
    KnnEvaluatorConfig,
    evaluate_knn,
)
from die_vfm.evaluator.knn_runner import (
    KnnInputConfig,
    KnnOutputConfig,
    KnnRunConfig,
    KnnRunResult,
    build_knn_run_config,
    resolve_knn_run_config,
    run_knn,
)

__all__ = [
    "LinearProbeClassifier",
    "build_linear_probe",
    "load_linear_probe_bundle",
    "run_linear_probe",
    "train_linear_probe",
    "write_linear_probe_outputs",
    "KnnEvaluationOutput",
    "KnnEvaluatorConfig",
    "evaluate_knn",
    "KnnInputConfig",
    "KnnOutputConfig",
    "KnnRunConfig",
    "KnnRunResult",
    "build_knn_run_config",
    "resolve_knn_run_config",
    "run_knn",
]