from __future__ import annotations

from pathlib import Path

from hydra import compose
from hydra import initialize_config_dir


def _compose_config(*, overrides: list[str] | None = None):
    """Composes the Hydra config for config-surface tests."""
    config_dir = str(Path("configs").resolve())

    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(config_name="config", overrides=overrides or [])


def test_base_config_loads() -> None:
    """Tests that the base Hydra config composes and exposes current sections."""
    cfg = _compose_config()

    assert cfg.project.name == "die_vfm"
    assert cfg.run.output_root == "runs"
    assert cfg.system.seed == 42

    assert isinstance(cfg.experiment.name, str)
    assert cfg.experiment.name
    assert cfg.model.backbone.name == "dummy"
    assert cfg.model.pooler.name == "mean"

    assert cfg.train.mode in {"bootstrap", "round1_frozen"}
    assert cfg.train.num_epochs == 1
    assert cfg.train.log_every_n_steps == 10
    assert cfg.train.run_dataloader_smoke_test is True
    assert cfg.train.run_model_forward_smoke_test is True

    assert cfg.dataset.name == "dummy"
    assert list(cfg.dataset.image_size) == [224, 224]
    assert cfg.dataset.num_channels == 3
    assert cfg.dataset.num_classes == 5
    assert cfg.dataset.train_size == 16
    assert cfg.dataset.val_size == 8
    assert cfg.dataset.label_offset == 0
    assert cfg.dataset.split_seed.train == 101
    assert cfg.dataset.split_seed.val == 202

    assert cfg.dataloader.batch_size == 4
    assert cfg.dataloader.drop_last is False
    assert cfg.dataloader.pin_memory is False
    assert cfg.dataloader.persistent_workers is False

    assert cfg.evaluation.run_linear_probe is False
    assert cfg.evaluation.run_knn is False
    assert cfg.evaluation.run_centroid is False
    assert cfg.evaluation.run_retrieval is False


def test_debug_model_smoke_preset_overrides_expected_root_fields() -> None:
    """Tests the explicit debug_model_smoke preset rather than default selection."""
    cfg = _compose_config(overrides=["experiment=debug_model_smoke"])

    assert cfg.experiment.name == "debug_model_smoke"
    assert cfg.train.mode == "bootstrap"
    assert cfg.system.device == "cpu"
    assert cfg.system.num_workers == 0
    assert cfg.dataloader.batch_size == 4
    assert cfg.model.return_debug_outputs is True


def test_domain_dataset_preset_composes_expected_config_surface() -> None:
    """Tests that the domain dataset preset composes the expected config surface."""
    cfg = _compose_config(overrides=["dataset=domain"])

    assert cfg.dataset.name == "domain"
    assert cfg.dataset.manifest_path == "./data/domain/manifest.csv"
    assert list(cfg.dataset.image_size) == [224, 224]
    assert cfg.dataset.merge_images is False
    assert cfg.dataset.single_image_source == "img1"
    assert cfg.dataset.require_non_empty_val is False

    assert cfg.dataset.did_field == "DID"
    assert cfg.dataset.img1_field == "IMG_1"
    assert cfg.dataset.img2_field == "IMG_2"
    assert cfg.dataset.source_field == "Source"
    assert cfg.dataset.label_field == "Label"
    assert cfg.dataset.path_field == "PATH"

    assert list(cfg.dataset.normalize.mean) == [0.485, 0.456, 0.406]
    assert list(cfg.dataset.normalize.std) == [0.229, 0.224, 0.225]
    assert dict(cfg.dataset.label_map) == {}


def test_domain_inference_export_preset_composes_expected_fields() -> None:
    """Tests the domain inference-only preset contract."""
    cfg = _compose_config(overrides=["experiment=domain_inference_export"])

    assert cfg.experiment.name == "domain_inference_export"
    assert cfg.train.mode == "round1_frozen"
    assert cfg.evaluation.run_linear_probe is False
    assert cfg.evaluation.run_knn is False
    assert cfg.evaluation.run_centroid is False
    assert cfg.evaluation.run_retrieval is False


def test_dinov2_backbone_preset_composes_expected_fields() -> None:
    """Tests that model/backbone=dinov2 composes expected config fields."""
    cfg = _compose_config(overrides=["model/backbone=dinov2"])

    assert cfg.model.backbone.name == "dinov2"
    assert cfg.model.backbone.variant == "vit_base"
    assert cfg.model.backbone.pretrained is True
    assert cfg.model.backbone.freeze is False
    assert cfg.model.backbone.return_cls_token is True
    assert cfg.model.backbone.allow_network is True
    assert cfg.model.backbone.local_repo_path is None
    assert cfg.model.backbone.local_checkpoint_path is None
