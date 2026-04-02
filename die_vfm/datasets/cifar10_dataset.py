from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from PIL import Image
from torchvision import datasets
from torchvision import transforms

from die_vfm.datasets.base import DatasetAdapter, DatasetSample


class Cifar10DatasetAdapter(DatasetAdapter):
    """CIFAR10 dataset adapter aligned to the die_vfm dataset contract."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        image_size: Sequence[int],
        download: bool,
        normalize_mean: Sequence[float],
        normalize_std: Sequence[float],
    ) -> None:
        """Initializes the CIFAR10 dataset adapter.

        Args:
          root: Root directory for the CIFAR10 dataset.
          split: Dataset split name. Supported values are "train" and "val".
          image_size: Output image size as (height, width).
          download: Whether to download the dataset if not found locally.
          normalize_mean: Per-channel normalization mean.
          normalize_std: Per-channel normalization std.
        """
        if split not in ("train", "val"):
            raise ValueError(f"Unsupported split for CIFAR10 dataset: {split}")
        if len(image_size) != 2:
            raise ValueError(
                "image_size must contain exactly two elements: (height, width)"
            )
        if len(normalize_mean) != 3:
            raise ValueError(
                "normalize_mean must contain exactly three elements for RGB"
            )
        if len(normalize_std) != 3:
            raise ValueError(
                "normalize_std must contain exactly three elements for RGB"
            )

        self._root = str(root)
        self._split = split
        self._image_height = int(image_size[0])
        self._image_width = int(image_size[1])
        self._download = bool(download)
        self._normalize_mean = [float(value) for value in normalize_mean]
        self._normalize_std = [float(value) for value in normalize_std]

        is_train_split = split == "train"
        self._transform = transforms.Compose([
            transforms.Resize((self._image_height, self._image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self._normalize_mean,
                std=self._normalize_std,
            ),
        ])
        self._dataset = datasets.CIFAR10(
            root=self._root,
            train=is_train_split,
            transform=self._transform,
            download=self._download,
        )
        self._class_names: List[str] = list(self._dataset.classes)

    @property
    def split(self) -> str:
        """Returns the dataset split name."""
        return self._split

    def __len__(self) -> int:
        """Returns the number of samples in the dataset split."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> DatasetSample:
        """Returns one CIFAR10 sample aligned to the dataset contract."""
        if index < 0 or index >= len(self):
            raise IndexError(
                f"index out of range: index={index}, len={len(self)}"
            )

        image, label = self._dataset[index]
        if not isinstance(image, type(self._transform(Image.new("RGB", (1, 1))))):
            # Defensive check: CIFAR10 with the configured transform should
            # already return a torch.Tensor after ToTensor().
            pass

        class_name = self._class_names[int(label)]
        sample: DatasetSample = {
            "image": image,
            "label": int(label),
            "image_id": f"cifar10_{self._split}_{index:05d}",
            "meta": {
                "split": self._split,
                "index": index,
                "source": "cifar10",
                "class_name": class_name,
                "raw_label": int(label),
            },
        }
        self.validate_sample(sample)
        return sample

    @classmethod
    def from_config(
        cls,
        cfg: Any,
        split: str,
    ) -> "Cifar10DatasetAdapter":
        """Builds a CIFAR10 dataset adapter instance from config."""
        return cls(
            root=cfg.root,
            split=split,
            image_size=cfg.image_size,
            download=bool(cfg.download),
            normalize_mean=cfg.normalize.mean,
            normalize_std=cfg.normalize.std,
        )

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Returns dataset-level metadata."""
        return {
            "dataset_name": "cifar10",
            "split": self._split,
            "num_samples": len(self),
            "num_channels": 3,
            "image_size": [self._image_height, self._image_width],
            "num_classes": len(self._class_names),
            "class_names": list(self._class_names),
            "root": self._root,
            "download": self._download,
            "normalize_mean": list(self._normalize_mean),
            "normalize_std": list(self._normalize_std),
        }