from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import yaml


def _run_round2_runtime(
    tmp_path: Path,
    *,
    run_name: str,
    precision_mode: str,
    extra_overrides: list[str] | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "experiment=round2_ssl",
        "system.device=cpu",
        "system.num_workers=0",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "dataset.train_size=8",
        "dataset.val_size=4",
        "dataset.image_size=[32,32]",
        "dataset.num_channels=3",
        "dataset.num_classes=4",
        "model.backbone.image_size=32",
        "model.backbone.patch_size=16",
        "model.backbone.in_channels=3",
        "model.backbone.embed_dim=16",
        "train.num_epochs=1",
        f"train.precision_mode={precision_mode}",
        "train.update_mode=projector_pooler_only",
        "dataloader.batch_size=2",
        "evaluation.run_linear_probe=false",
        "evaluation.run_knn=true",
        "evaluation.run_retrieval=false",
        "evaluation.knn.evaluator.k=3",
        "evaluation.knn.evaluator.topk=[1]",
        "round2.postprocess.mode=in_process",
        "round2.evaluation.run_pair_benchmark=false",
        "round2.evaluation.run_slicing_analysis=false",
        *(extra_overrides or []),
    ]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=os.environ.copy(),
    )


def test_round2_runtime_debug_fp32_smoke(tmp_path: Path) -> None:
    result = _run_round2_runtime(
        tmp_path,
        run_name="pytest-round2-fp32",
        precision_mode="fp32",
    )
    assert result.returncode == 0, result.stderr

    round2_dir = tmp_path / "pytest-round2-fp32" / "round2"
    assert (round2_dir / "checkpoints" / "latest.pt").exists()
    assert (round2_dir / "embeddings" / "train" / "manifest.yaml").exists()
    assert (round2_dir / "embeddings" / "val" / "manifest.yaml").exists()
    assert (round2_dir / "evaluation" / "knn" / "summary.yaml").exists()

    summary_path = round2_dir / "round2_summary.yaml"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)
    assert summary["phase"] == "round2_ssl"
    assert summary["execution"]["precision_mode"] == "fp32"


def test_round2_runtime_pilot_bf16_smoke(tmp_path: Path) -> None:
    result = _run_round2_runtime(
        tmp_path,
        run_name="pytest-round2-bf16",
        precision_mode="bf16",
    )
    assert result.returncode == 0, result.stderr

    round2_dir = tmp_path / "pytest-round2-bf16" / "round2"
    summary_path = round2_dir / "round2_summary.yaml"
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = yaml.safe_load(handle)
    assert summary["execution"]["precision_mode"] == "bf16"
    assert "knn.top1_accuracy" in summary["metrics"]


def test_round2_runtime_full_resume_smoke(tmp_path: Path) -> None:
    first = _run_round2_runtime(
        tmp_path,
        run_name="pytest-round2-resume",
        precision_mode="fp32",
    )
    assert first.returncode == 0, first.stderr

    run_dir = tmp_path / "pytest-round2-resume" / "round2"
    first_summary_path = run_dir / "round2_summary.yaml"
    with first_summary_path.open("r", encoding="utf-8") as handle:
        first_summary = yaml.safe_load(handle)

    second = _run_round2_runtime(
        tmp_path,
        run_name="pytest-round2-resume",
        precision_mode="fp32",
        extra_overrides=[
            "train.num_epochs=2",
            "train.resume.enabled=true",
            "train.resume.mode=full_resume",
            "train.resume.auto_resume_latest=true",
        ],
    )
    assert second.returncode == 0, second.stderr

    with first_summary_path.open("r", encoding="utf-8") as handle:
        second_summary = yaml.safe_load(handle)

    assert second_summary["trainer_state"]["epoch"] == 2
    assert (
        second_summary["trainer_state"]["global_step"]
        > first_summary["trainer_state"]["global_step"]
    )
    assert second_summary["execution"]["resume_enabled"] is True
    assert second_summary["execution"]["resume_mode"] == "full_resume"
    assert second_summary["execution"]["restored_from_checkpoint"] is True
