"""Tests for scripts.export_embeddings."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml
from omegaconf import OmegaConf

from die_vfm.artifacts.embedding_artifact import EMBEDDING_ARTIFACT_VERSION
import scripts.export_embeddings as export_embeddings_script


def _run_script(repo_root: Path, overrides: list[str]) -> subprocess.CompletedProcess[str]:
    """Runs the export_embeddings script with Hydra overrides."""
    script_path = repo_root / "scripts" / "export_embeddings.py"

    command = [sys.executable, str(script_path), *overrides]
    return subprocess.run(
        command,
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_export_embeddings_script_noops_when_disabled(tmp_path: Path) -> None:
    """Exits cleanly and skips artifact export when disabled."""
    repo_root = Path(__file__).resolve().parents[1]
    run_name = "pytest-export-disabled"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            f"run.output_root={tmp_path}",
            f"run.run_name={run_name}",
            "artifact.embedding.enabled=false",
        ],
    )

    assert completed.returncode == 0, completed.stderr

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "export_embeddings.log"
    config_path = run_dir / "config.yaml"

    assert run_dir.exists()
    assert log_path.exists()
    assert config_path.exists()
    assert not (run_dir / "embeddings").exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Embedding artifact export disabled; skipping." in log_text
    assert "No embedding splits were exported." in log_text


def test_export_embeddings_script_writes_train_and_val_artifacts(
    tmp_path: Path,
) -> None:
    """Runs the script end to end and writes train/val embedding artifacts."""
    repo_root = Path(__file__).resolve().parents[1]
    run_name = "pytest-export-success"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            f"run.output_root={tmp_path}",
            f"run.run_name={run_name}",
            "system.device=cpu",
            "system.num_workers=0",
            "model/backbone=dummy",
            "model/pooler=mean",
            "dataset=dummy",
        ],
    )

    assert completed.returncode == 0, completed.stderr

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "export_embeddings.log"
    config_path = run_dir / "config.yaml"
    train_dir = run_dir / "embeddings" / "train"
    val_dir = run_dir / "embeddings" / "val"

    assert run_dir.exists()
    assert log_path.exists()
    assert config_path.exists()
    assert train_dir.exists()
    assert val_dir.exists()

    train_manifest_path = train_dir / "manifest.yaml"
    val_manifest_path = val_dir / "manifest.yaml"
    train_shard_path = train_dir / "part-00000.pt"
    val_shard_path = val_dir / "part-00000.pt"

    assert train_manifest_path.exists()
    assert val_manifest_path.exists()
    assert train_shard_path.exists()
    assert val_shard_path.exists()

    with train_manifest_path.open("r", encoding="utf-8") as file:
        train_manifest = yaml.safe_load(file)
    with val_manifest_path.open("r", encoding="utf-8") as file:
        val_manifest = yaml.safe_load(file)

    train_payload = torch.load(train_shard_path)
    val_payload = torch.load(val_shard_path)

    assert train_manifest["split"] == "train"
    assert train_manifest["num_shards"] == 1
    assert train_manifest["num_samples"] > 0
    assert train_manifest["embedding_dim"] > 0

    assert val_manifest["split"] == "val"
    assert val_manifest["num_shards"] == 1
    assert val_manifest["num_samples"] > 0
    assert val_manifest["embedding_dim"] == train_manifest["embedding_dim"]

    assert train_payload["embeddings"].shape[0] == train_manifest["num_samples"]
    assert val_payload["embeddings"].shape[0] == val_manifest["num_samples"]
    assert len(train_payload["image_ids"]) == train_manifest["num_samples"]
    assert len(val_payload["image_ids"]) == val_manifest["num_samples"]
    assert len(train_payload["metadata"]) == train_manifest["num_samples"]
    assert len(val_payload["metadata"]) == val_manifest["num_samples"]
    assert train_payload["labels"] is not None
    assert val_payload["labels"] is not None

    log_text = log_path.read_text(encoding="utf-8")
    assert "Starting embedding export." in log_text
    assert "Export completed successfully for splits: train, val" in log_text
    assert "Embedding export job completed successfully." in log_text


def test_export_embeddings_main_respects_export_splits_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Entrypoint should export only the explicitly requested splits."""
    cfg = OmegaConf.create(
        {
            "run": {
                "output_root": str(tmp_path),
                "run_name": "pytest-export-splits",
                "save_config_snapshot": False,
            },
            "system": {
                "seed": 123,
                "device": "cpu",
            },
            "artifact": {
                "embedding": {
                    "enabled": True,
                    "output_subdir": "embeddings",
                    "export_splits": ["val"],
                    "include_test_split": False,
                }
            },
            "model": {},
            "dataset": {},
            "dataloader": {},
        }
    )

    calls: list[tuple[str, Path]] = []

    class _FakeModel:
        def to(self, device: object) -> "_FakeModel":
            return self

    def _fake_build_model(_: object) -> _FakeModel:
        return _FakeModel()

    def _fake_build_dataloader(_: object, split: str) -> str:
        return f"loader:{split}"

    def _fake_export_split_embeddings(
        *,
        model: object,
        dataloader: object,
        output_dir: Path,
        split: str,
        device: object,
    ) -> SimpleNamespace:
        del model, dataloader, device
        calls.append((split, Path(output_dir)))
        return SimpleNamespace(num_samples=1, embedding_dim=2, has_labels=True)

    monkeypatch.setattr(export_embeddings_script, "setup_logging", lambda _: None)
    monkeypatch.setattr(
        export_embeddings_script,
        "save_config_snapshot",
        lambda cfg, run_dir: None,
    )
    monkeypatch.setattr(export_embeddings_script, "build_model", _fake_build_model)
    monkeypatch.setattr(
        export_embeddings_script,
        "build_dataloader",
        _fake_build_dataloader,
    )
    monkeypatch.setattr(
        export_embeddings_script,
        "export_split_embeddings",
        _fake_export_split_embeddings,
    )

    export_embeddings_script.main.__wrapped__(cfg)

    assert calls == [
        (
            "val",
            tmp_path / "pytest-export-splits" / "embeddings" / "val",
        )
    ]


def test_export_embeddings_main_appends_test_split_when_enabled(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Entrypoint should append `test` when include_test_split is enabled."""
    cfg = OmegaConf.create(
        {
            "run": {
                "output_root": str(tmp_path),
                "run_name": "pytest-export-include-test",
                "save_config_snapshot": False,
            },
            "system": {
                "seed": 123,
                "device": "cpu",
            },
            "artifact": {
                "embedding": {
                    "enabled": True,
                    "output_subdir": "embeddings",
                    "export_splits": ["train", "val"],
                    "include_test_split": True,
                }
            },
            "model": {},
            "dataset": {},
            "dataloader": {},
        }
    )

    calls: list[str] = []

    class _FakeModel:
        def to(self, device: object) -> "_FakeModel":
            return self

    def _fake_build_model(_: object) -> _FakeModel:
        return _FakeModel()

    def _fake_build_dataloader(_: object, split: str) -> str:
        calls.append(f"dataloader:{split}")
        return f"loader:{split}"

    def _fake_export_split_embeddings(
        *,
        model: object,
        dataloader: object,
        output_dir: Path,
        split: str,
        device: object,
    ) -> SimpleNamespace:
        del model, dataloader, output_dir, device
        calls.append(f"export:{split}")
        return SimpleNamespace(num_samples=1, embedding_dim=2, has_labels=True)

    monkeypatch.setattr(export_embeddings_script, "setup_logging", lambda _: None)
    monkeypatch.setattr(
        export_embeddings_script,
        "save_config_snapshot",
        lambda cfg, run_dir: None,
    )
    monkeypatch.setattr(export_embeddings_script, "build_model", _fake_build_model)
    monkeypatch.setattr(
        export_embeddings_script,
        "build_dataloader",
        _fake_build_dataloader,
    )
    monkeypatch.setattr(
        export_embeddings_script,
        "export_split_embeddings",
        _fake_export_split_embeddings,
    )

    export_embeddings_script.main.__wrapped__(cfg)

    assert calls == [
        "dataloader:train",
        "export:train",
        "dataloader:val",
        "export:val",
        "dataloader:test",
        "export:test",
    ]


def test_export_embeddings_script_placeholder_fields_are_noop(
    tmp_path: Path,
) -> None:
    """Placeholder artifact config fields should not alter current M1 behavior."""
    repo_root = Path(__file__).resolve().parents[1]
    run_name = "pytest-export-placeholder-noop"

    completed = _run_script(
        repo_root=repo_root,
        overrides=[
            f"run.output_root={tmp_path}",
            f"run.run_name={run_name}",
            "system.device=cpu",
            "system.num_workers=0",
            "model/backbone=dummy",
            "model/pooler=mean",
            "dataset=dummy",
            "artifact.embedding.save_labels=false",
            "artifact.embedding.save_metadata=false",
            "artifact.embedding.artifact_version=v999",
            "artifact.embedding.shard_size=2",
        ],
    )

    assert completed.returncode == 0, completed.stderr

    run_dir = tmp_path / run_name
    config_path = run_dir / "config.yaml"
    train_dir = run_dir / "embeddings" / "train"
    train_manifest_path = train_dir / "manifest.yaml"
    train_shard_path = train_dir / "part-00000.pt"

    assert config_path.exists()
    assert train_manifest_path.exists()
    assert train_shard_path.exists()

    with config_path.open("r", encoding="utf-8") as file:
        config_payload = yaml.safe_load(file)
    with train_manifest_path.open("r", encoding="utf-8") as file:
        manifest_payload = yaml.safe_load(file)

    shard_payload = torch.load(train_shard_path)

    assert config_payload["artifact"]["embedding"]["save_labels"] is False
    assert config_payload["artifact"]["embedding"]["save_metadata"] is False
    assert config_payload["artifact"]["embedding"]["artifact_version"] == "v999"
    assert config_payload["artifact"]["embedding"]["shard_size"] == 2

    assert manifest_payload["artifact_version"] == EMBEDDING_ARTIFACT_VERSION
    assert manifest_payload["num_shards"] == 1
    assert manifest_payload["shards"] == [
        {"file_name": "part-00000.pt", "num_samples": manifest_payload["num_samples"]}
    ]
    assert shard_payload["labels"] is not None
    assert len(shard_payload["metadata"]) == manifest_payload["num_samples"]
