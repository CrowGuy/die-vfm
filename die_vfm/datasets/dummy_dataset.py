from __future__ import annotations

from typing import Any, Dict, Sequence

import torch

from die_vfm.datasets.base import DatasetAdapter, DatasetSample


class DummyDatasetAdapter(DatasetAdapter):
    """Deterministic dummy dataset adapter for bootstrap and smoke tests."""

    def __init__(
        self,
        split: str,
        image_size: Sequence[int],
        num_channels: int,
        num_classes: int,
        dataset_size: int,
        label_offset: int,
        split_seed: int,
    ) -> None:
        """Initializes the dummy dataset adapter.

        Args:
          split: Dataset split name. Usually "train" or "val".
          image_size: Image spatial size as (height, width).
          num_channels: Number of image channels.
          num_classes: Number of classes used to generate labels.
          dataset_size: Number of samples in the dataset split.
          label_offset: Offset added to generated labels.
          split_seed: Base random seed for deterministic sample generation.
        """
        if len(image_size) != 2:
            raise ValueError(
                "image_size must contain exactly two elements: (height, width)"
            )
        if dataset_size <= 0:
            raise ValueError(
                f"dataset_size must be positive, got {dataset_size}"
            )
        if num_channels <= 0:
            raise ValueError(
                f"num_channels must be positive, got {num_channels}"
            )
        if num_classes <= 0:
            raise ValueError(
                f"num_classes must be positive, got {num_classes}"
            )

        self._split = split
        self._image_height = int(image_size[0])
        self._image_width = int(image_size[1])
        self._num_channels = int(num_channels)
        self._num_classes = int(num_classes)
        self._dataset_size = int(dataset_size)
        self._label_offset = int(label_offset)
        self._split_seed = int(split_seed)

    @property
    def split(self) -> str:
        """Returns the dataset split name."""
        return self._split

    def __len__(self) -> int:
        """Returns the number of samples in the dataset split."""
        return self._dataset_size

    def __getitem__(self, index: int) -> DatasetSample:
        """Returns a deterministic dummy sample for the given index."""
        if index < 0 or index >= len(self):
            raise IndexError(
                f"index out of range: index={index}, len={len(self)}"
            )

        generator = torch.Generator().manual_seed(self._split_seed + index)
        image = torch.randn(
            self._num_channels,
            self._image_height,
            self._image_width,
            generator=generator,
        )
        label = self._label_offset + (index % self._num_classes)

        sample: DatasetSample = {
            "image": image,
            "label": label,
            "image_id": f"{self._split}_{index:05d}",
            "meta": {
                "split": self._split,
                "index": index,
                "source": "dummy",
            },
        }
        self.validate_sample(sample)
        return sample

    @classmethod
    def from_config(cls, cfg: Any, split: str) -> "DummyDatasetAdapter":
        """Builds a dummy dataset adapter instance from config."""
        split_to_size = {
            "train": int(cfg.train_size),
            "val": int(cfg.val_size),
        }
        if split not in split_to_size:
            raise ValueError(f"Unsupported split for dummy dataset: {split}")

        split_seed_cfg = cfg.split_seed
        split_to_seed = {
            "train": int(split_seed_cfg.train),
            "val": int(split_seed_cfg.val),
        }

        return cls(
            split=split,
            image_size=cfg.image_size,
            num_channels=int(cfg.num_channels),
            num_classes=int(cfg.num_classes),
            dataset_size=split_to_size[split],
            label_offset=int(cfg.label_offset),
            split_seed=split_to_seed[split],
        )

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Returns dataset-level metadata."""
        return {
            "dataset_name": "dummy",
            "split": self._split,
            "num_samples": self._dataset_size,
            "num_channels": self._num_channels,
            "image_size": [self._image_height, self._image_width],
            "num_classes": self._num_classes,
            "label_offset": self._label_offset,
            "split_seed": self._split_seed,
        }