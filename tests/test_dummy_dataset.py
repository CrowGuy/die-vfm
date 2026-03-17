from __future__ import annotations

from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from die_vfm.datasets.builder import build_dataloader, build_dataset
from die_vfm.datasets.dummy_dataset import DummyDatasetAdapter


def _compose_config() -> DictConfig:
    """Composes the project config for tests."""
    with hydra.initialize(version_base=None, config_path="../configs"):
        cfg = hydra.compose(config_name="config")
    return cfg


def test_dummy_dataset_adapter_contract() -> None:
    """Tests the dummy dataset adapter sample contract."""
    cfg = _compose_config()

    dataset = build_dataset(cfg, split="train")
    assert isinstance(dataset, DummyDatasetAdapter)
    assert len(dataset) == 16

    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert sample["image"].shape == (3, 224, 224)
    assert isinstance(sample["label"], int)
    assert sample["label"] == 0
    assert sample["image_id"] == "train_00000"
    assert sample["meta"]["split"] == "train"
    assert sample["meta"]["index"] == 0
    assert sample["meta"]["source"] == "dummy"

    metadata = dataset.get_dataset_metadata()
    assert metadata["dataset_name"] == "dummy"
    assert metadata["split"] == "train"
    assert metadata["num_samples"] == 16
    assert metadata["num_classes"] == 5


def test_dataloader_builder_collates_batch() -> None:
    """Tests the dataloader builder and batch collation."""
    cfg = _compose_config()

    dataloader = build_dataloader(cfg, split="train")
    batch = next(iter(dataloader))

    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape == (4, 3, 224, 224)

    assert isinstance(batch["label"], torch.Tensor)
    assert batch["label"] is not None
    assert batch["label"].shape == (4,)
    assert batch["label"].dtype == torch.long

    assert isinstance(batch["image_id"], list)
    assert len(batch["image_id"]) == 4
    assert batch["image_id"][0].startswith("train_")

    assert isinstance(batch["meta"], list)
    assert len(batch["meta"]) == 4
    assert batch["meta"][0]["split"] == "train"