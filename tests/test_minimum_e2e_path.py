from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import yaml
from PIL import Image


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


def _write_domain_rgb_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (32, 32), color=color)
    image.save(path)


def _write_domain_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = ["DID", "IMG_1", "IMG_2", "Source", "Label", "PATH"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def test_round1_frozen_domain_cli_smoke_train_only_export(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "domain_images"
    image_dir.mkdir()
    _write_domain_rgb_image(image_dir / "train_a.png", color=(255, 0, 0))
    _write_domain_rgb_image(image_dir / "train_b.png", color=(0, 255, 0))

    manifest_path = tmp_path / "domain_manifest.csv"
    _write_domain_manifest(manifest_path, [
        {
            "DID": "train_1",
            "IMG_1": "train_a.png",
            "IMG_2": "",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "train_2",
            "IMG_1": "train_b.png",
            "IMG_2": "",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    run_name = "pytest-round1-domain-cli-smoke"
    round1_result = _run_runtime(
        tmp_path=tmp_path,
        run_name=run_name,
        overrides=[
            "train.mode=round1_frozen",
            "model/backbone=dummy",
            "model/pooler=mean",
            "dataset=domain",
            f"dataset.manifest_path={manifest_path}",
            "+dataset.label_map.ok=1",
            "dataloader.batch_size=2",
            "evaluation.run_linear_probe=false",
            "evaluation.run_knn=false",
            "evaluation.run_retrieval=false",
            "evaluation.run_centroid=false",
        ],
    )
    assert round1_result.returncode == 0, (
        f"round1 domain cli smoke failed with stderr:\n{round1_result.stderr}\n"
        f"stdout:\n{round1_result.stdout}"
    )

    run_dir = tmp_path / run_name
    round1_dir = run_dir / "round1"
    summary_path = round1_dir / "round1_summary.yaml"

    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert summary_path.exists()
    assert (round1_dir / "embeddings" / "train" / "manifest.yaml").exists()
    assert (round1_dir / "embeddings" / "train" / "part-00000.pt").exists()
    assert not (round1_dir / "embeddings" / "val" / "manifest.yaml").exists()

    with summary_path.open("r", encoding="utf-8") as file:
        summary_payload = yaml.safe_load(file)

    assert summary_payload["phase"] == "round1_frozen"
    assert summary_payload["execution"]["enabled_evaluators"] == []
    assert summary_payload["execution"]["requested_evaluators"] == []
    assert summary_payload["manifests"]["train"]["available"] is True
    assert summary_payload["manifests"]["val"]["available"] is False
    assert summary_payload["metrics"] == {}

    log_path = run_dir / "logs" / "run.log"
    assert log_path.exists()


def test_domain_inference_export_experiment_runtime_smoke(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "domain_images"
    image_dir.mkdir()
    _write_domain_rgb_image(image_dir / "infer_a.png", color=(0, 0, 255))

    manifest_path = tmp_path / "domain_manifest_infer_only.csv"
    _write_domain_manifest(manifest_path, [
        {
            "DID": "infer_1",
            "IMG_1": "infer_a.png",
            "IMG_2": "",
            "Source": "Infer",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    run_name = "pytest-domain-inference-export-smoke"
    result = _run_runtime(
        tmp_path=tmp_path,
        run_name=run_name,
        overrides=[
            "experiment=domain_inference_export",
            "dataset=domain",
            f"dataset.manifest_path={manifest_path}",
            "dataset.require_non_empty_val=true",
            "dataloader.batch_size=1",
        ],
    )
    assert result.returncode == 0, (
        "domain_inference_export runtime smoke failed with stderr:\n"
        f"{result.stderr}\nstdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    round1_dir = run_dir / "round1"
    summary_path = round1_dir / "round1_summary.yaml"

    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()
    assert summary_path.exists()
    assert not (round1_dir / "embeddings" / "train" / "manifest.yaml").exists()
    assert (round1_dir / "embeddings" / "val" / "manifest.yaml").exists()
    assert (round1_dir / "embeddings" / "val" / "part-00000.pt").exists()

    with summary_path.open("r", encoding="utf-8") as file:
        summary_payload = yaml.safe_load(file)

    assert summary_payload["phase"] == "round1_frozen"
    assert summary_payload["execution"]["enabled_evaluators"] == []
    assert summary_payload["execution"]["requested_evaluators"] == []
    assert summary_payload["manifests"]["train"]["available"] is False
    assert summary_payload["manifests"]["val"]["available"] is True
    assert summary_payload["metrics"] == {}

    log_path = run_dir / "logs" / "run.log"
    assert log_path.exists()


def test_domain_inference_export_rejects_mixed_label_val_split(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "domain_images"
    image_dir.mkdir()
    _write_domain_rgb_image(image_dir / "infer_a.png", color=(0, 0, 255))
    _write_domain_rgb_image(image_dir / "infer_b.png", color=(0, 255, 255))

    manifest_path = tmp_path / "domain_manifest_mixed_val.csv"
    _write_domain_manifest(manifest_path, [
        {
            "DID": "infer_labeled",
            "IMG_1": "infer_a.png",
            "IMG_2": "",
            "Source": "Infer",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "infer_unlabeled",
            "IMG_1": "infer_b.png",
            "IMG_2": "",
            "Source": "Infer",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    run_name = "pytest-domain-inference-export-mixed-val"
    result = _run_runtime(
        tmp_path=tmp_path,
        run_name=run_name,
        overrides=[
            "experiment=domain_inference_export",
            "dataset=domain",
            f"dataset.manifest_path={manifest_path}",
            "dataset.require_non_empty_val=true",
            "+dataset.label_map.ok=1",
            "dataloader.batch_size=1",
        ],
    )

    assert result.returncode != 0
    assert "must not mix labeled and unlabeled samples" in (
        result.stderr + result.stdout
    )


def test_domain_inference_export_rejects_empty_val_when_required(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "domain_images"
    image_dir.mkdir()
    _write_domain_rgb_image(image_dir / "train_only.png", color=(255, 255, 0))

    manifest_path = tmp_path / "domain_manifest_train_only.csv"
    _write_domain_manifest(manifest_path, [
        {
            "DID": "train_1",
            "IMG_1": "train_only.png",
            "IMG_2": "",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    run_name = "pytest-domain-inference-export-empty-val"
    result = _run_runtime(
        tmp_path=tmp_path,
        run_name=run_name,
        overrides=[
            "experiment=domain_inference_export",
            "dataset=domain",
            f"dataset.manifest_path={manifest_path}",
            "dataset.require_non_empty_val=true",
            "+dataset.label_map.ok=1",
            "dataloader.batch_size=1",
        ],
    )

    assert result.returncode != 0
    assert "Filtered val split is empty under inference-only policy" in (
        result.stderr + result.stdout
    )
