from __future__ import annotations

from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from die_vfm.datasets.builder import build_dataloader, build_dataset
from die_vfm.datasets.cifar10_dataset import Cifar10DatasetAdapter


class _FakeTorchvisionCIFAR10:
    """Minimal fake CIFAR10 dataset for adapter tests."""

    classes = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    last_init_args: dict[str, Any] | None = None

    def __init__(
        self,
        root: str,
        train: bool,
        transform: Any,
        download: bool,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.download = download
        self.samples = [
            (self._make_image(fill_value=0), 0),
            (self._make_image(fill_value=64), 3),
            (self._make_image(fill_value=128), 7),
        ]
        _FakeTorchvisionCIFAR10.last_init_args = {
            "root": root,
            "train": train,
            "transform": transform,
            "download": download,
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        image, label = self.samples[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    @staticmethod
    def _make_image(fill_value: int) -> Image.Image:
        array = np.full((32, 32, 3), fill_value, dtype=np.uint8)
        return Image.fromarray(array, mode="RGB")


def _make_cfg() -> Any:
    return OmegaConf.create({
        "dataset": {
            "name": "cifar10",
            "root": "./data/cifar10",
            "image_size": [224, 224],
            "download": False,
            "normalize": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "dataloader": {
            "batch_size": 2,
            "drop_last": False,
            "pin_memory": False,
            "persistent_workers": False,
        },
        "system": {
            "num_workers": 0,
        },
    })


def test_cifar10_dataset_adapter_sample_contract(monkeypatch) -> None:
    monkeypatch.setattr(
        "die_vfm.datasets.cifar10_dataset.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    dataset = Cifar10DatasetAdapter(
        root="./data/cifar10",
        split="train",
        image_size=[224, 224],
        download=False,
        normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225],
    )

    sample = dataset[1]

    assert set(sample.keys()) == {"image", "label", "image_id", "meta"}
    assert isinstance(sample["image"], torch.Tensor)
    assert tuple(sample["image"].shape) == (3, 224, 224)
    assert sample["label"] == 3
    assert sample["image_id"] == "cifar10_train_00001"

    assert sample["meta"]["split"] == "train"
    assert sample["meta"]["index"] == 1
    assert sample["meta"]["source"] == "cifar10"
    assert sample["meta"]["class_name"] == "cat"
    assert sample["meta"]["raw_label"] == 3


def test_cifar10_dataset_adapter_from_config(monkeypatch) -> None:
    monkeypatch.setattr(
        "die_vfm.datasets.cifar10_dataset.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    cfg = _make_cfg()
    dataset = Cifar10DatasetAdapter.from_config(cfg.dataset, split="train")

    assert isinstance(dataset, Cifar10DatasetAdapter)
    assert len(dataset) == 3

    init_args = _FakeTorchvisionCIFAR10.last_init_args
    assert init_args is not None
    assert init_args["root"] == "./data/cifar10"
    assert init_args["train"] is True
    assert init_args["download"] is False


def test_cifar10_dataset_adapter_val_split_maps_to_test(monkeypatch) -> None:
    monkeypatch.setattr(
        "die_vfm.datasets.cifar10_dataset.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    dataset = Cifar10DatasetAdapter.from_config(_make_cfg().dataset, split="val")

    init_args = _FakeTorchvisionCIFAR10.last_init_args
    assert init_args is not None
    assert init_args["train"] is False
    assert dataset.split == "val"


def test_build_dataset_returns_cifar10_adapter(monkeypatch) -> None:
    monkeypatch.setattr(
        "die_vfm.datasets.cifar10_dataset.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    cfg = _make_cfg()
    dataset = build_dataset(cfg, split="train")

    assert isinstance(dataset, Cifar10DatasetAdapter)
    assert len(dataset) == 3


def test_build_dataloader_collates_cifar10_batch(monkeypatch) -> None:
    monkeypatch.setattr(
        "die_vfm.datasets.cifar10_dataset.datasets.CIFAR10",
        _FakeTorchvisionCIFAR10,
    )

    cfg = _make_cfg()
    dataloader = build_dataloader(cfg, split="train")
    batch = next(iter(dataloader))

    assert set(batch.keys()) == {"image", "label", "image_id", "meta"}
    assert isinstance(batch["image"], torch.Tensor)
    assert tuple(batch["image"].shape) == (2, 3, 224, 224)
    assert isinstance(batch["label"], torch.Tensor)
    assert tuple(batch["label"].shape) == (2,)
    assert isinstance(batch["image_id"], list)
    assert len(batch["image_id"]) == 2
    assert isinstance(batch["meta"], list)
    assert len(batch["meta"]) == 2
    assert all(isinstance(item, dict) for item in batch["meta"])


def test_cifar10_dataset_adapter_rejects_unsupported_split() -> None:
    try:
        Cifar10DatasetAdapter(
            root="./data/cifar10",
            split="test",
            image_size=[224, 224],
            download=False,
            normalize_mean=[0.485, 0.456, 0.406],
            normalize_std=[0.229, 0.224, 0.225],
        )
    except ValueError as exc:
        assert "Unsupported split" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unsupported split.")