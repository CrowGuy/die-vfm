from __future__ import annotations

from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from die_vfm.datasets.base import DatasetAdapter, DatasetSample
from die_vfm.datasets.cifar10_dataset import Cifar10DatasetAdapter
from die_vfm.datasets.dummy_dataset import DummyDatasetAdapter


def build_dataset(cfg: Any, split: str) -> DatasetAdapter:
    """Builds a dataset adapter from config.

    Args:
      cfg: Global project config.
      split: Dataset split name, such as "train" or "val".

    Returns:
      A dataset adapter instance.

    Raises:
      ValueError: If the dataset name is unsupported.
    """
    dataset_name = str(cfg.dataset.name)

    if dataset_name == "dummy":
        return DummyDatasetAdapter.from_config(cfg.dataset, split=split)

    if dataset_name == "cifar10":
        return Cifar10DatasetAdapter.from_config(cfg.dataset, split=split)

    raise ValueError(f"Unsupported dataset name: {dataset_name}")


def collate_dataset_samples(
    samples: List[DatasetSample],
) -> Dict[str, Any]:
    """Collates dataset samples into a batch dictionary.

    Args:
      samples: A list of dataset samples.

    Returns:
      A batch dictionary with the following keys:
      - image: torch.Tensor with shape [B, C, H, W]
      - label: torch.Tensor with shape [B] or None
      - image_id: list[str]
      - meta: list[dict]

    Raises:
      ValueError: If the input sample list is empty.
    """
    if not samples:
        raise ValueError("Cannot collate an empty sample list.")

    images = torch.stack([sample["image"] for sample in samples], dim=0)
    labels: List[Optional[int]] = [sample["label"] for sample in samples]

    if all(label is not None for label in labels):
        batch_labels: Optional[torch.Tensor] = torch.tensor(
            labels,
            dtype=torch.long,
        )
    else:
        batch_labels = None

    batch = {
        "image": images,
        "label": batch_labels,
        "image_id": [sample["image_id"] for sample in samples],
        "meta": [sample["meta"] for sample in samples],
    }
    return batch


def build_dataloader(cfg: Any, split: str) -> DataLoader:
    """Builds a PyTorch dataloader for the given split.

    Args:
      cfg: Global project config.
      split: Dataset split name, such as "train" or "val".

    Returns:
      A PyTorch dataloader instance.
    """
    dataset = build_dataset(cfg, split=split)
    shuffle = split == "train"
    num_workers = int(cfg.system.num_workers)
    persistent_workers = bool(cfg.dataloader.persistent_workers)
    if num_workers == 0:
        persistent_workers = False

    return DataLoader(
        dataset=dataset,
        batch_size=int(cfg.dataloader.batch_size),
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=bool(cfg.dataloader.drop_last),
        pin_memory=bool(cfg.dataloader.pin_memory),
        persistent_workers=persistent_workers,
        collate_fn=collate_dataset_samples,
    )