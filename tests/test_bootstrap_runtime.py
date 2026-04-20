from __future__ import annotations

import csv
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image
import torch


def _write_domain_rgb_image(
    path: Path,
    *,
    color: tuple[int, int, int],
) -> None:
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path)


def _write_domain_manifest(
    path: Path,
    rows: list[dict[str, str]],
) -> None:
    fieldnames = ["DID", "IMG_1", "IMG_2", "Source", "Label", "PATH"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_local_dinov2_hub_repo(path: Path) -> None:
    """Creates a minimal local torch.hub repo for offline dinov2 smoke."""
    path.mkdir(parents=True, exist_ok=True)
    hubconf_path = path / "hubconf.py"
    hubconf_path.write_text(
        "\n".join([
            "import torch",
            "from torch import nn",
            "",
            "class _LocalDINO(nn.Module):",
            "    def __init__(self, feature_dim: int):",
            "        super().__init__()",
            "        self.feature_dim = int(feature_dim)",
            "        self.weight = nn.Parameter(torch.ones(1))",
            "",
            "    def forward_features(self, image):",
            "        batch_size, _, height, width = image.shape",
            "        patch_size = 14",
            "        num_patches = (height // patch_size) * (width // patch_size)",
            "        return {",
            "            'x_norm_patchtokens': torch.zeros(",
            "                (batch_size, num_patches, self.feature_dim),",
            "                device=image.device,",
            "                dtype=image.dtype,",
            "            ),",
            "            'x_norm_clstoken': torch.zeros(",
            "                (batch_size, self.feature_dim),",
            "                device=image.device,",
            "                dtype=image.dtype,",
            "            ),",
            "        }",
            "",
            "def dinov2_vits14(pretrained: bool = False):",
            "    return _LocalDINO(feature_dim=384)",
            "",
            "def dinov2_vitb14(pretrained: bool = False):",
            "    return _LocalDINO(feature_dim=768)",
            "",
            "def dinov2_vitl14(pretrained: bool = False):",
            "    return _LocalDINO(feature_dim=1024)",
            "",
            "def dinov2_vitg14(pretrained: bool = False):",
            "    return _LocalDINO(feature_dim=1536)",
            "",
        ]),
        encoding="utf-8",
    )


def test_bootstrap_runtime_runs_dataloader_and_model_smoke_test(
    tmp_path: Path,
) -> None:
    """Tests that the runtime entrypoint completes bootstrap successfully."""
    run_name = "pytest-bootstrap-runtime"

    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.run_model_forward_smoke_test=true",
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"run.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()

    log_path = run_dir / "logs" / "run.log"
    assert log_path.exists()

    log_text = log_path.read_text(encoding="utf-8")
    assert "Starting runtime entrypoint." in log_text
    assert "Starting bootstrap runtime." in log_text
    assert "Dataloader smoke test passed." in log_text
    assert "Batch image shape: (4, 3, 224, 224)" in log_text
    assert "Batch label shape: (4,)" in log_text

    assert "Built model: DieVFMModel" in log_text
    assert "Backbone: DummyBackbone" in log_text
    assert "Pooler: MeanPooler" in log_text
    assert "Model forward completed." in log_text
    assert "Embedding shape: (4, 192)" in log_text

    assert "Saved latest checkpoint:" in log_text
    assert "Saved epoch checkpoint:" in log_text
    assert "Saved best checkpoint:" in log_text
    assert "Bootstrap runtime completed successfully." in log_text
    assert "Dataset metadata:" in log_text

    dataset_metadata_path = run_dir / "dataset_metadata.yaml"
    assert dataset_metadata_path.exists()

    dataset_metadata_text = dataset_metadata_path.read_text(encoding="utf-8")
    assert "dataset_name: dummy" in dataset_metadata_text
    assert "split: train" in dataset_metadata_text
    assert "num_samples: 16" in dataset_metadata_text

    model_smoke_path = run_dir / "model_smoke.yaml"
    assert model_smoke_path.exists()

    model_smoke_text = model_smoke_path.read_text(encoding="utf-8")
    assert "name: DieVFMModel" in model_smoke_text
    assert "embedding_dim: 192" in model_smoke_text
    assert "backbone: DummyBackbone" in model_smoke_text
    assert "pooler: MeanPooler" in model_smoke_text
    assert "- 4" in model_smoke_text
    assert "- 192" in model_smoke_text


def test_bootstrap_runtime_domain_dataset_cli_smoke(
    tmp_path: Path,
) -> None:
    """Tests that bootstrap can run end-to-end with dataset=domain."""
    image_dir = tmp_path / "domain_images"
    image_dir.mkdir()
    _write_domain_rgb_image(image_dir / "a.png", color=(255, 0, 0))
    _write_domain_rgb_image(image_dir / "b.png", color=(0, 255, 0))

    manifest_path = tmp_path / "domain_manifest.csv"
    _write_domain_manifest(manifest_path, [
        {
            "DID": "train_1",
            "IMG_1": "a.png",
            "IMG_2": "",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "train_2",
            "IMG_1": "b.png",
            "IMG_2": "",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    run_name = "pytest-bootstrap-domain-runtime"
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=domain",
        f"dataset.manifest_path={manifest_path}",
        "+dataset.label_map.ok=1",
        "dataloader.batch_size=2",
        "train.run_dataloader_smoke_test=true",
        "train.run_model_forward_smoke_test=true",
    ]

    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, (
        f"run.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    assert run_dir.exists()
    assert (run_dir / "config.yaml").exists()

    log_path = run_dir / "logs" / "run.log"
    assert log_path.exists()
    log_text = log_path.read_text(encoding="utf-8")

    assert "Starting bootstrap runtime." in log_text
    assert "Dataloader smoke test passed." in log_text
    assert "Batch image shape: (2, 3, 224, 224)" in log_text
    assert "Batch label shape: (2,)" in log_text
    assert "Saved latest checkpoint:" in log_text
    assert "Saved epoch checkpoint:" in log_text
    assert "Saved best checkpoint:" in log_text
    assert "Bootstrap runtime completed successfully." in log_text
    assert "Dataset metadata:" in log_text

    dataset_metadata_path = run_dir / "dataset_metadata.yaml"
    assert dataset_metadata_path.exists()
    dataset_metadata_text = dataset_metadata_path.read_text(encoding="utf-8")
    assert "dataset_name: domain" in dataset_metadata_text
    assert "split: train" in dataset_metadata_text
    assert "num_samples: 2" in dataset_metadata_text
    assert "has_labels: true" in dataset_metadata_text
    assert "merge_images: false" in dataset_metadata_text
    assert "single_image_source: img1" in dataset_metadata_text

    checkpoint_dir = run_dir / "checkpoints"
    assert checkpoint_dir.exists()
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "epoch_0000.pt").exists()

    model_smoke_path = run_dir / "model_smoke.yaml"
    assert model_smoke_path.exists()
    model_smoke_text = model_smoke_path.read_text(encoding="utf-8")
    assert "name: DieVFMModel" in model_smoke_text
    assert "embedding_dim: 192" in model_smoke_text
    assert "- 2" in model_smoke_text
    assert "- 192" in model_smoke_text


def test_bootstrap_runtime_dinov2_cli_smoke_with_fake_hub(
    tmp_path: Path,
) -> None:
    """Tests bootstrap runtime with model/backbone=dinov2 under fake hub."""
    run_name = "pytest-bootstrap-dinov2-runtime"
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dinov2",
        "model.backbone.pretrained=false",
        "model.backbone.freeze=false",
        "train.freeze_backbone=false",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.run_model_forward_smoke_test=true",
    ]

    env = os.environ.copy()
    env["DIE_VFM_DINOV2_FAKE_HUB"] = "1"
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"run.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "run.log"
    log_text = log_path.read_text(encoding="utf-8")

    assert "Backbone: DINOv2Backbone" in log_text
    assert "Embedding shape: (4, 768)" in log_text
    assert "Bootstrap runtime completed successfully." in log_text

    model_smoke_path = run_dir / "model_smoke.yaml"
    model_smoke_text = model_smoke_path.read_text(encoding="utf-8")
    assert "embedding_dim: 768" in model_smoke_text
    assert "backbone: DINOv2Backbone" in model_smoke_text


def test_bootstrap_runtime_rejects_conflicting_dinov2_freeze_policy(
    tmp_path: Path,
) -> None:
    """Fails fast when train/model freeze flags conflict for dinov2."""
    run_name = "pytest-bootstrap-dinov2-conflict"
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dinov2",
        "model.backbone.pretrained=false",
        "model.backbone.freeze=false",
        "train.freeze_backbone=true",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
    ]

    env = os.environ.copy()
    env["DIE_VFM_DINOV2_FAKE_HUB"] = "1"
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode != 0
    assert "Conflicting freeze policy for dinov2" in (result.stdout + result.stderr)


def test_bootstrap_runtime_dinov2_offline_local_asset_cli_smoke(
    tmp_path: Path,
) -> None:
    """Tests offline dinov2 bootstrap path with local hub repo asset."""
    local_repo_path = tmp_path / "local_dinov2_repo"
    _write_local_dinov2_hub_repo(local_repo_path)

    run_name = "pytest-bootstrap-dinov2-offline-local-asset"
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dinov2",
        "model.backbone.pretrained=false",
        "model.backbone.freeze=false",
        "model.backbone.allow_network=false",
        f"model.backbone.local_repo_path={local_repo_path}",
        "train.freeze_backbone=false",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.run_model_forward_smoke_test=true",
    ]

    env = os.environ.copy()
    env.pop("DIE_VFM_DINOV2_FAKE_HUB", None)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"run.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "run.log"
    log_text = log_path.read_text(encoding="utf-8")

    assert "Backbone: DINOv2Backbone" in log_text
    assert "Embedding shape: (4, 768)" in log_text
    assert "Bootstrap runtime completed successfully." in log_text

    model_smoke_path = run_dir / "model_smoke.yaml"
    model_smoke_text = model_smoke_path.read_text(encoding="utf-8")
    assert "embedding_dim: 768" in model_smoke_text
    assert "backbone: DINOv2Backbone" in model_smoke_text


def test_bootstrap_runtime_dinov2_offline_local_checkpoint_cli_smoke(
    tmp_path: Path,
) -> None:
    """Tests offline pretrained dinov2 bootstrap path with local checkpoint."""
    local_repo_path = tmp_path / "local_dinov2_repo"
    _write_local_dinov2_hub_repo(local_repo_path)

    checkpoint_path = tmp_path / "local_dinov2.ckpt"
    torch.save({"weight": torch.ones(1)}, checkpoint_path)

    run_name = "pytest-bootstrap-dinov2-offline-checkpoint"
    command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dinov2",
        "model.backbone.pretrained=true",
        "model.backbone.freeze=false",
        "model.backbone.allow_network=false",
        f"model.backbone.local_repo_path={local_repo_path}",
        f"model.backbone.local_checkpoint_path={checkpoint_path}",
        "train.freeze_backbone=false",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.run_model_forward_smoke_test=true",
    ]

    env = os.environ.copy()
    env.pop("DIE_VFM_DINOV2_FAKE_HUB", None)
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0, (
        f"run.py failed with stderr:\n{result.stderr}\n"
        f"stdout:\n{result.stdout}"
    )

    run_dir = tmp_path / run_name
    log_path = run_dir / "logs" / "run.log"
    log_text = log_path.read_text(encoding="utf-8")

    assert "Backbone: DINOv2Backbone" in log_text
    assert "Embedding shape: (4, 768)" in log_text
    assert "Bootstrap runtime completed successfully." in log_text


def test_bootstrap_runtime_writes_checkpoint_set(tmp_path: Path) -> None:
  """Tests that bootstrap writes latest/best/epoch checkpoints."""
  run_name = "pytest-bootstrap-checkpoint-save"
  command = [
      sys.executable,
      "scripts/run.py",
      f"run.output_root={tmp_path}",
      f"run.run_name={run_name}",
      "system.num_workers=0",
      "system.device=cpu",
      "model/backbone=dummy",
      "model/pooler=mean",
      "dataset=dummy",
      "train.run_dataloader_smoke_test=true",
  ]

  result = subprocess.run(
      command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert result.returncode == 0, (
      f"run.py failed with stderr:\n{result.stderr}\n"
      f"stdout:\n{result.stdout}"
  )

  checkpoint_dir = tmp_path / run_name / "checkpoints"
  assert checkpoint_dir.exists()
  assert (checkpoint_dir / "latest.pt").exists()
  assert (checkpoint_dir / "best.pt").exists()
  assert (checkpoint_dir / "epoch_0000.pt").exists()

  log_path = tmp_path / run_name / "logs" / "run.log"
  log_text = log_path.read_text(encoding="utf-8")
  assert "Saved latest checkpoint:" in log_text
  assert "Saved epoch checkpoint:" in log_text
  assert "Saved best checkpoint:" in log_text
  assert "Bootstrap runtime completed successfully." in log_text


def test_bootstrap_runtime_auto_resume_latest_full_resume(
    tmp_path: Path,
) -> None:
  """Tests that bootstrap can auto-resume from latest.pt."""
  run_name = "pytest-bootstrap-auto-resume-full"

  first_command = [
      sys.executable,
      "scripts/run.py",
      f"run.output_root={tmp_path}",
      f"run.run_name={run_name}",
      "system.num_workers=0",
      "system.device=cpu",
      "model/backbone=dummy",
      "model/pooler=mean",
      "dataset=dummy",
      "train.run_dataloader_smoke_test=true",
  ]
  first_result = subprocess.run(
      first_command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert first_result.returncode == 0, (
      f"first run failed with stderr:\n{first_result.stderr}\n"
      f"stdout:\n{first_result.stdout}"
  )

  second_command = [
      sys.executable,
      "scripts/run.py",
      f"run.output_root={tmp_path}",
      f"run.run_name={run_name}",
      "system.num_workers=0",
      "system.device=cpu",
      "model/backbone=dummy",
      "model/pooler=mean",
      "dataset=dummy",
      "train.run_dataloader_smoke_test=true",
      "train.resume.enabled=true",
      "train.resume.mode=full_resume",
      "train.resume.auto_resume_latest=true",
      "train.resume.checkpoint_path=null",
  ]
  second_result = subprocess.run(
      second_command,
      check=False,
      capture_output=True,
      text=True,
  )

  assert second_result.returncode == 0, (
      f"second run failed with stderr:\n{second_result.stderr}\n"
      f"stdout:\n{second_result.stdout}"
  )

  log_path = tmp_path / run_name / "logs" / "run.log"
  log_text = log_path.read_text(encoding="utf-8")

  assert "Resolved resume checkpoint:" in log_text
  assert "Resume mode: full_resume" in log_text
  assert "Full resume completed." in log_text
  assert "Saved latest checkpoint:" in log_text
  assert "Saved epoch checkpoint:" in log_text
  assert "Bootstrap runtime completed successfully." in log_text

  checkpoint_dir = tmp_path / run_name / "checkpoints"
  assert (checkpoint_dir / "latest.pt").exists()
  assert (checkpoint_dir / "epoch_0000.pt").exists()


def test_bootstrap_runtime_explicit_warm_start_checkpoint(
        tmp_path: Path,
    ) -> None:
    """Tests that bootstrap can warm start from an explicit checkpoint."""
    source_run_name = "pytest-bootstrap-warm-start-source"
    target_run_name = "pytest-bootstrap-warm-start-target"

    source_command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={source_run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
    ]
    source_result = subprocess.run(
        source_command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert source_result.returncode == 0, (
        f"source run failed with stderr:\n{source_result.stderr}\n"
        f"stdout:\n{source_result.stdout}"
    )

    checkpoint_path = (
        tmp_path / source_run_name / "checkpoints" / "latest.pt"
    )
    assert checkpoint_path.exists()

    target_command = [
        sys.executable,
        "scripts/run.py",
        f"run.output_root={tmp_path}",
        f"run.run_name={target_run_name}",
        "system.num_workers=0",
        "system.device=cpu",
        "model/backbone=dummy",
        "model/pooler=mean",
        "dataset=dummy",
        "train.run_dataloader_smoke_test=true",
        "train.resume.enabled=true",
        "train.resume.mode=warm_start",
        f"train.resume.checkpoint_path={checkpoint_path}",
        "train.resume.auto_resume_latest=false",
    ]
    target_result = subprocess.run(
        target_command,
        check=False,
        capture_output=True,
        text=True,
    )

    assert target_result.returncode == 0, (
        f"target run failed with stderr:\n{target_result.stderr}\n"
        f"stdout:\n{target_result.stdout}"
    )

    log_path = tmp_path / target_run_name / "logs" / "run.log"
    log_text = log_path.read_text(encoding="utf-8")

    assert "Resolved resume checkpoint:" in log_text
    assert "Resume mode: warm_start" in log_text
    assert "Warm start completed from checkpoint." in log_text
    assert "Saved latest checkpoint:" in log_text
    assert "Saved epoch checkpoint:" in log_text
    assert "Bootstrap runtime completed successfully." in log_text

    checkpoint_dir = tmp_path / target_run_name / "checkpoints"
    assert (checkpoint_dir / "latest.pt").exists()
    assert (checkpoint_dir / "best.pt").exists()
    assert (checkpoint_dir / "epoch_0000.pt").exists()
