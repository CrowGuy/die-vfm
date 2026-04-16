from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml


def _run_runtime(tmp_path: Path, run_name: str, overrides: list[str]) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.device=cpu",
        "system.num_workers=0",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        *overrides,
    ]
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def test_minimum_trustworthy_e2e_path_bootstrap_then_round1_knn_only(
    tmp_path: Path,
) -> None:
    run_name = "pytest-minimum-e2e-path"

    # Stage A: bootstrap should cover config/model/checkpoint write smoke.
    bootstrap_result = _run_runtime(
        tmp_path=tmp_path,
        run_name=run_name,
        overrides=[
            "train.mode=bootstrap",
            "train.run_dataloader_smoke_test=true",
            "train.run_model_forward_smoke_test=true",
        ],
    )
    assert bootstrap_result.returncode == 0, (
        f"bootstrap failed with stderr:\n{bootstrap_result.stderr}\n"
        f"stdout:\n{bootstrap_result.stdout}"
    )

    run_dir = tmp_path / run_name
    checkpoint_dir = run_dir / "checkpoints"
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "dataset_metadata.yaml").exists()
    assert (run_dir / "model_smoke.yaml").exists()
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "epoch_0000.pt").exists()

    # Stage B: round1 should cover embedding export + evaluator orchestration.
    round1_result = _run_runtime(
        tmp_path=tmp_path,
        run_name=run_name,
        overrides=[
            "train.mode=round1_frozen",
            "train.freeze_backbone=true",
            "train.freeze_pooler=true",
            "evaluation.run_linear_probe=false",
            "evaluation.run_knn=true",
            "evaluation.run_retrieval=false",
            "evaluation.run_centroid=false",
            "evaluation.knn.evaluator.k=3",
        ],
    )
    assert round1_result.returncode == 0, (
        f"round1 failed with stderr:\n{round1_result.stderr}\n"
        f"stdout:\n{round1_result.stdout}"
    )

    round1_dir = run_dir / "round1"
    train_split_dir = round1_dir / "embeddings" / "train"
    val_split_dir = round1_dir / "embeddings" / "val"
    knn_output_dir = round1_dir / "evaluation" / "knn"
    summary_path = round1_dir / "round1_summary.yaml"

    assert (train_split_dir / "manifest.yaml").exists()
    assert (train_split_dir / "part-00000.pt").exists()
    assert (val_split_dir / "manifest.yaml").exists()
    assert (val_split_dir / "part-00000.pt").exists()
    assert knn_output_dir.exists()
    assert (knn_output_dir / "metrics.yaml").exists()
    assert (knn_output_dir / "summary.yaml").exists()
    assert not (round1_dir / "evaluation" / "linear_probe").exists()
    assert not (round1_dir / "evaluation" / "retrieval").exists()
    assert summary_path.exists()

    with summary_path.open("r", encoding="utf-8") as file:
        summary_payload = yaml.safe_load(file)

    assert summary_payload["phase"] == "round1_frozen"
    assert summary_payload["runtime_semantics"]["mode"] == "single_shot_inference_evaluation"
    assert summary_payload["execution"]["enabled_evaluators"] == ["knn"]
    assert "knn" in summary_payload["artifacts"]["evaluation_dirs"]
    assert "linear_probe" not in summary_payload["artifacts"]["evaluation_dirs"]
    assert "retrieval" not in summary_payload["artifacts"]["evaluation_dirs"]
    assert any(metric_name.startswith("knn.") for metric_name in summary_payload["metrics"])
