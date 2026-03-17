from __future__ import annotations

import abc
from typing import Any, Dict, Optional, TypedDict

import torch
from torch.utils.data import Dataset


class DatasetSample(TypedDict):
    """Typed dictionary for a single dataset sample."""

    image: torch.Tensor
    label: Optional[int]
    image_id: str
    meta: Dict[str, Any]


class DatasetAdapter(Dataset[DatasetSample], metaclass=abc.ABCMeta):
    """Abstract dataset adapter interface.

    All concrete dataset adapters must return samples with the following
    contract:

      - image: torch.Tensor with shape [C, H, W]
      - label: int or None
      - image_id: str
      - meta: dict
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""

    @abc.abstractmethod
    def __getitem__(self, index: int) -> DatasetSample:
        """Returns one dataset sample by index."""

    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Any, split: str) -> "DatasetAdapter":
        """Builds a dataset adapter instance from config."""

    @abc.abstractmethod
    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Returns dataset-level metadata."""

    @staticmethod
    def validate_sample(sample: DatasetSample) -> None:
        """Validates a dataset sample.

        Args:
          sample: A sample returned by a dataset adapter.

        Raises:
          KeyError: If a required key is missing.
          TypeError: If a field type is invalid.
          ValueError: If a tensor shape is invalid.
        """
        required_keys = ("image", "label", "image_id", "meta")
        for key in required_keys:
            if key not in sample:
                raise KeyError(f"Missing required sample key: {key}")

        image = sample["image"]
        label = sample["label"]
        image_id = sample["image_id"]
        meta = sample["meta"]

        if not isinstance(image, torch.Tensor):
            raise TypeError(
                f"sample['image'] must be a torch.Tensor, got {type(image)}"
            )
        if image.ndim != 3:
            raise ValueError(
                "sample['image'] must have shape [C, H, W], "
                f"got ndim={image.ndim}"
            )

        if label is not None and not isinstance(label, int):
            raise TypeError(
                f"sample['label'] must be int or None, got {type(label)}"
            )

        if not isinstance(image_id, str):
            raise TypeError(
                f"sample['image_id'] must be str, got {type(image_id)}"
            )

        if not isinstance(meta, dict):
            raise TypeError(
                f"sample['meta'] must be dict, got {type(meta)}"
            )