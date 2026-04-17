from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from decimal import InvalidOperation
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, TypedDict

import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import VisionDataset

from die_vfm.datasets.base import DatasetAdapter, DatasetSample


_SOURCE_TO_SPLIT = {
    "Train": "train",
    "Infer": "val",
}
_SUPPORTED_SPLITS = frozenset({"train", "val"})
_SUPPORTED_SINGLE_IMAGE_SOURCE = frozenset({"img1", "img2"})


def _is_missing_value(value: Any) -> bool:
    """Returns whether a manifest field should be treated as missing."""
    if value is None:
        return True
    missing = pd.isna(value)
    if isinstance(missing, bool):
        return missing
    return False


def _to_optional_string(value: Any) -> str | None:
    """Converts one manifest value into a stripped optional string."""
    if _is_missing_value(value):
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _canonicalize_numeric_label(text: str) -> str | None:
    """Canonicalizes one numeric-like label string when parseable."""
    try:
        decimal_value = Decimal(text)
    except InvalidOperation:
        return None

    if not decimal_value.is_finite():
        return None

    normalized = decimal_value.normalize()
    if normalized == normalized.to_integral_value():
        return str(int(normalized))

    normalized_text = format(normalized, "f")
    if "." in normalized_text:
        normalized_text = normalized_text.rstrip("0").rstrip(".")
    return normalized_text.lower()


def _canonicalize_label(value: Any) -> str | None:
    """Canonicalizes one raw label value into the runtime label key space."""
    text = _to_optional_string(value)
    if text is None:
        return None

    lowered = text.lower()
    numeric = _canonicalize_numeric_label(lowered)
    if numeric is not None:
        return numeric
    return lowered


def _build_label_map(label_map_cfg: Any) -> dict[str, int]:
    """Builds a canonicalized label map from config."""
    if label_map_cfg is None:
        return {}
    if not hasattr(label_map_cfg, "items"):
        raise TypeError("label_map must be a mapping from label to int.")

    canonical_map: dict[str, int] = {}
    for raw_key, raw_value in label_map_cfg.items():
        canonical_key = _canonicalize_label(raw_key)
        if canonical_key is None:
            raise ValueError("label_map contains an empty key.")
        if isinstance(raw_value, bool) or not isinstance(raw_value, int):
            raise TypeError(
                "label_map values must be int, got "
                f"{type(raw_value)} for key {raw_key!r}"
            )

        mapped_value = int(raw_value)
        existing = canonical_map.get(canonical_key)
        if existing is not None and existing != mapped_value:
            raise ValueError(
                "label_map contains conflicting canonical keys: "
                f"{canonical_key!r}"
            )
        canonical_map[canonical_key] = mapped_value
    return canonical_map


@dataclass(frozen=True)
class _ManifestRow:
    """Validated manifest row used by the domain dataset runtime."""

    did: str
    source: str
    split: str
    path: Path
    img1: str
    img2: str | None
    raw_label: str | None
    canonical_label: str | None
    label: int | None
    selected_image_source: str
    selected_image_path: Path
    merge_img1_path: Path | None
    merge_img2_path: Path | None


class DomainVisionSample(TypedDict):
    """Typed sample payload returned by DomainVisionDataset."""

    image: torch.Tensor
    label: Optional[int]
    did: str
    source: str
    path: str
    img1: str
    img2: str | None
    raw_label: str | None
    canonical_label: str | None
    selected_image_source: str


class DomainVisionDataset(VisionDataset):
    """CSV-manifest-driven vision dataset for domain runtime ingestion."""

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        image_size: Sequence[int],
        merge_images: bool,
        single_image_source: str,
        require_non_empty_val: bool,
        did_field: str,
        img1_field: str,
        img2_field: str,
        source_field: str,
        label_field: str,
        path_field: str,
        normalize_mean: Sequence[float],
        normalize_std: Sequence[float],
        label_map: Any,
    ) -> None:
        super().__init__(root="")

        if split not in _SUPPORTED_SPLITS:
            raise ValueError(f"Unsupported split for domain dataset: {split}")
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
        if single_image_source not in _SUPPORTED_SINGLE_IMAGE_SOURCE:
            raise ValueError(
                "single_image_source must be one of {'img1', 'img2'}, "
                f"got {single_image_source!r}"
            )

        self._manifest_path = Path(manifest_path).expanduser().resolve()
        self._split = split
        self._image_height = int(image_size[0])
        self._image_width = int(image_size[1])
        self._merge_images = bool(merge_images)
        self._single_image_source = single_image_source
        self._require_non_empty_val = bool(require_non_empty_val)

        self._did_field = str(did_field)
        self._img1_field = str(img1_field)
        self._img2_field = str(img2_field)
        self._source_field = str(source_field)
        self._label_field = str(label_field)
        self._path_field = str(path_field)

        self._normalize_mean = [float(value) for value in normalize_mean]
        self._normalize_std = [float(value) for value in normalize_std]
        self._transform = transforms.Compose([
            transforms.Resize((self._image_height, self._image_width)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self._normalize_mean,
                std=self._normalize_std,
            ),
        ])
        self._label_map = _build_label_map(label_map)
        all_rows = self._load_and_validate_manifest()
        self._rows = [
            row for row in all_rows if row.split == self._split
        ]

        if self._split == "train" and len(self._rows) == 0:
            raise ValueError("Filtered train split is empty.")
        if (
            self._split == "val"
            and self._require_non_empty_val
            and len(self._rows) == 0
        ):
            raise ValueError(
                "Filtered val split is empty under inference-only policy."
            )

        has_labeled = [row.label is not None for row in self._rows]
        if any(has_labeled) and not all(has_labeled):
            if self._split == "train":
                raise ValueError(
                    "Filtered train split must not mix labeled and unlabeled "
                    "samples."
                )
            if self._split == "val":
                raise ValueError(
                    "Filtered val split must not mix labeled and unlabeled "
                    "samples under current artifact contract."
                )

        self._source_values = sorted({row.source for row in self._rows})
        self._label_vocabulary = sorted({
            row.canonical_label
            for row in self._rows
            if row.canonical_label is not None
        })

    @property
    def split(self) -> str:
        """Returns the dataset split name."""
        return self._split

    def __len__(self) -> int:
        """Returns the number of samples in the split."""
        return len(self._rows)

    def __getitem__(self, index: int) -> DomainVisionSample:
        """Loads one validated domain sample by split-local index."""
        if index < 0 or index >= len(self):
            raise IndexError(
                f"index out of range: index={index}, len={len(self)}"
            )

        row = self._rows[index]
        image = self._load_image(row)
        return {
            "image": image,
            "label": row.label,
            "did": row.did,
            "source": row.source,
            "path": str(row.path),
            "img1": row.img1,
            "img2": row.img2,
            "raw_label": row.raw_label,
            "canonical_label": row.canonical_label,
            "selected_image_source": row.selected_image_source,
        }

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Returns split-level dataset metadata."""
        has_labels = bool(self._rows) and all(
            row.label is not None for row in self._rows
        )
        return {
            "dataset_name": "domain",
            "split": self._split,
            "num_samples": len(self),
            "num_channels": 3,
            "image_size": [self._image_height, self._image_width],
            "has_labels": has_labels,
            "manifest_path": str(self._manifest_path),
            "source_values": list(self._source_values),
            "label_vocabulary": list(self._label_vocabulary),
            "merge_images": self._merge_images,
            "single_image_source": self._single_image_source,
            "require_non_empty_val": self._require_non_empty_val,
        }

    def _load_and_validate_manifest(self) -> list[_ManifestRow]:
        if not self._manifest_path.exists():
            raise ValueError(
                f"Manifest path does not exist: {self._manifest_path}"
            )
        if not self._manifest_path.is_file():
            raise ValueError(
                "Manifest path must point to a CSV file, got: "
                f"{self._manifest_path}"
            )

        dataframe = pd.read_csv(self._manifest_path)
        required_columns = [
            self._did_field,
            self._img1_field,
            self._source_field,
            self._path_field,
        ]
        for column in required_columns:
            if column not in dataframe.columns:
                raise KeyError(f"Missing required manifest column: {column}")

        has_img2_column = self._img2_field in dataframe.columns
        has_label_column = self._label_field in dataframe.columns

        validated_rows: list[_ManifestRow] = []
        did_set: set[str] = set()

        for row_index, row in dataframe.iterrows():
            did = _to_optional_string(row[self._did_field])
            if did is None:
                raise ValueError(
                    f"Manifest row {row_index} has empty DID field."
                )
            if did in did_set:
                raise ValueError(f"Duplicate DID in manifest: {did}")
            did_set.add(did)

            source = _to_optional_string(row[self._source_field])
            if source not in _SOURCE_TO_SPLIT:
                raise ValueError(
                    f"Invalid Source value at row {row_index}: {source!r}"
                )
            split = _SOURCE_TO_SPLIT[source]

            path_text = _to_optional_string(row[self._path_field])
            if path_text is None:
                raise ValueError(
                    f"Manifest row {row_index} has empty PATH field."
                )
            path = Path(path_text)
            if not path.is_absolute():
                raise ValueError(
                    "PATH must be absolute, got "
                    f"{path_text!r} at row {row_index}"
                )
            if not path.is_dir():
                raise ValueError(
                    f"PATH is not a directory at row {row_index}: {path_text}"
                )

            img1 = _to_optional_string(row[self._img1_field])
            if img1 is None:
                raise ValueError(
                    f"Manifest row {row_index} has empty IMG_1 field."
                )
            img2 = None
            if has_img2_column:
                img2 = _to_optional_string(row[self._img2_field])

            raw_label = None
            if has_label_column:
                raw_label = _to_optional_string(row[self._label_field])

            canonical_label = _canonicalize_label(raw_label)
            mapped_label: int | None = None
            if canonical_label is not None:
                if canonical_label not in self._label_map:
                    raise ValueError(
                        "Manifest label not found in label_map at row "
                        f"{row_index}: {raw_label!r} -> {canonical_label!r}"
                    )
                mapped_label = self._label_map[canonical_label]

            img1_path = path / img1
            if not img1_path.is_file():
                raise ValueError(
                    f"IMG_1 file is missing at row {row_index}: {img1_path}"
                )

            (
                selected_source,
                selected_image_path,
                merge_img1_path,
                merge_img2_path,
            ) = self._resolve_image_plan(
                row_index=row_index,
                img1_path=img1_path,
                img2_name=img2,
                path=path,
            )

            validated_rows.append(_ManifestRow(
                did=did,
                source=source,
                split=split,
                path=path,
                img1=img1,
                img2=img2,
                raw_label=raw_label,
                canonical_label=canonical_label,
                label=mapped_label,
                selected_image_source=selected_source,
                selected_image_path=selected_image_path,
                merge_img1_path=merge_img1_path,
                merge_img2_path=merge_img2_path,
            ))

        return validated_rows

    def _resolve_image_plan(
        self,
        row_index: int,
        img1_path: Path,
        img2_name: str | None,
        path: Path,
    ) -> tuple[str, Path, Path | None, Path | None]:
        has_img2 = img2_name is not None

        if self._merge_images and has_img2:
            img2_path = path / img2_name
            if not img2_path.is_file():
                raise ValueError(
                    f"IMG_2 file is missing at row {row_index}: {img2_path}"
                )
            with Image.open(img1_path) as image1, Image.open(img2_path) as image2:
                if image1.size != image2.size:
                    raise ValueError(
                        "Merge images must have the same spatial size at row "
                        f"{row_index}: {img1_path} vs {img2_path}"
                    )
            return ("merged", img1_path, img1_path, img2_path)

        if self._single_image_source == "img2" and has_img2:
            img2_path = path / img2_name
            if not img2_path.is_file():
                raise ValueError(
                    f"IMG_2 file is missing at row {row_index}: {img2_path}"
                )
            return ("img2", img2_path, None, None)

        return ("img1", img1_path, None, None)

    def _load_image(self, row: _ManifestRow) -> torch.Tensor:
        if row.selected_image_source == "merged":
            if row.merge_img1_path is None or row.merge_img2_path is None:
                raise ValueError("Merged image row is missing merge image paths.")
            with Image.open(row.merge_img1_path) as image1:
                image1_l = image1.convert("L")
            with Image.open(row.merge_img2_path) as image2:
                image2_l = image2.convert("L")
            if image1_l.size != image2_l.size:
                raise ValueError(
                    "Merged images must have the same spatial size during load."
                )
            black = Image.new("L", image1_l.size, color=0)
            image = Image.merge("RGB", (image1_l, image2_l, black))
        else:
            with Image.open(row.selected_image_path) as loaded_image:
                image = loaded_image.convert("RGB")

        return self._transform(image)


class DomainDatasetAdapter(DatasetAdapter):
    """Domain dataset adapter aligned to the die_vfm dataset contract."""

    def __init__(self, dataset: DomainVisionDataset) -> None:
        self._dataset = dataset
        self._split = dataset.split

    @property
    def split(self) -> str:
        """Returns the dataset split name."""
        return self._split

    def __len__(self) -> int:
        """Returns the number of samples in the dataset split."""
        return len(self._dataset)

    def __getitem__(self, index: int) -> DatasetSample:
        """Returns one domain sample aligned to the dataset contract."""
        vision_sample = self._dataset[index]
        sample: DatasetSample = {
            "image": vision_sample["image"],
            "label": vision_sample["label"],
            "image_id": vision_sample["did"],
            "meta": {
                "split": self._split,
                "source": vision_sample["source"],
                "did": vision_sample["did"],
                "img_1": vision_sample["img1"],
                "img_2": vision_sample["img2"],
                "path": vision_sample["path"],
                "raw_label": vision_sample["raw_label"],
                "canonical_label": vision_sample["canonical_label"],
                "merge_images": (
                    vision_sample["selected_image_source"] == "merged"
                ),
                "selected_image_source": vision_sample[
                    "selected_image_source"
                ],
            },
        }
        self.validate_sample(sample)
        return sample

    @classmethod
    def from_config(cls, cfg: Any, split: str) -> "DomainDatasetAdapter":
        """Builds a domain dataset adapter instance from config."""
        dataset = DomainVisionDataset(
            manifest_path=cfg.manifest_path,
            split=split,
            image_size=cfg.image_size,
            merge_images=bool(cfg.merge_images),
            single_image_source=str(cfg.single_image_source),
            require_non_empty_val=bool(
                getattr(cfg, "require_non_empty_val", False)
            ),
            did_field=str(cfg.did_field),
            img1_field=str(cfg.img1_field),
            img2_field=str(cfg.img2_field),
            source_field=str(cfg.source_field),
            label_field=str(cfg.label_field),
            path_field=str(cfg.path_field),
            normalize_mean=cfg.normalize.mean,
            normalize_std=cfg.normalize.std,
            label_map=getattr(cfg, "label_map", {}),
        )
        return cls(dataset=dataset)

    def get_dataset_metadata(self) -> Dict[str, Any]:
        """Returns dataset-level metadata."""
        return self._dataset.get_dataset_metadata()
