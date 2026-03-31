from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import yaml
from torch.utils.data import DataLoader, Dataset

from die_vfm.artifacts.embedding_artifact import (
    EMBEDDING_MANIFEST_FILENAME,
    EmbeddingManifest,
)
from die_vfm.artifacts.embedding_exporter import export_split_embeddings
from die_vfm.artifacts.embedding_loader import (
    load_embedding_artifacts,
    load_embedding_split,
)


class _ToyDataset(Dataset):
    """Deterministic toy dataset for embedding artifact loader tests."""

    def __init__(
        self,
        split: str,
        num_samples: int,
        with_labels: bool = True,
    ) -> None:
        self.split = split
        self.num_samples = num_samples
        self.with_labels = with_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        image = torch.full((3, 8, 8), fill_value=float(index), dtype=torch.float32)
        label = index % 3 if self.with_labels else None
        return {
            "image": image,
            "label": label,
            "image_id": f"{self.split}_{index:05d}",
            "meta": {
                "split": self.split,
                "index": index,
                "source": "toy",
            },
        }


def _toy_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch], dim=0)

    raw_labels = [item["label"] for item in batch]
    if all(label is None for label in raw_labels):
        labels = None
    else:
        if any(label is None for label in raw_labels):
            raise ValueError("Mixed label availability within one batch is not allowed.")
        labels = torch.tensor(raw_labels, dtype=torch.long)

    return {
        "image": images,
        "label": labels,
        "image_id": [item["image_id"] for item in batch],
        "meta": [item["meta"] for item in batch],
    }


class _ToyModel(torch.nn.Module):
    """Tiny model that returns output.embedding with shape [B, D]."""

    def __init__(self, embedding_dim: int = 5) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, image: torch.Tensor) -> SimpleNamespace:
        batch_size = image.shape[0]
        flat_mean = image.view(batch_size, -1).mean(dim=1, keepdim=True)
        offsets = torch.arange(
            self.embedding_dim,
            device=image.device,
            dtype=image.dtype,
        ).unsqueeze(0)
        embedding = flat_mean + offsets
        return SimpleNamespace(embedding=embedding)


def _build_dataloader(
    split: str,
    num_samples: int,
    batch_size: int,
    with_labels: bool = True,
) -> DataLoader:
    dataset = _ToyDataset(
        split=split,
        num_samples=num_samples,
        with_labels=with_labels,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=_toy_collate,
    )


def _export_split(
    root_dir: Path,
    split: str,
    num_samples: int,
    embedding_dim: int,
    with_labels: bool = True,
) -> Path:
    model = _ToyModel(embedding_dim=embedding_dim)
    dataloader = _build_dataloader(
        split=split,
        num_samples=num_samples,
        batch_size=2,
        with_labels=with_labels,
    )
    split_dir = root_dir / split
    export_split_embeddings(
        model=model,
        dataloader=dataloader,
        output_dir=split_dir,
        split=split,
        device="cpu",
    )
    return split_dir


def test_load_embedding_split_round_trip_with_labels(tmp_path: Path) -> None:
    split_dir = _export_split(
        root_dir=tmp_path / "embeddings",
        split="train",
        num_samples=5,
        embedding_dim=4,
        with_labels=True,
    )

    artifact = load_embedding_split(split_dir)

    assert artifact.manifest.split == "train"
    assert artifact.manifest.num_samples == 5
    assert artifact.manifest.embedding_dim == 4
    assert artifact.manifest.has_labels is True

    assert artifact.embeddings.shape == (5, 4)
    assert artifact.labels is not None
    assert artifact.labels.shape == (5,)
    assert artifact.image_ids == [f"train_{idx:05d}" for idx in range(5)]
    assert len(artifact.metadata) == 5

    for idx in range(5):
        assert artifact.metadata[idx]["split"] == "train"
        assert artifact.metadata[idx]["index"] == idx
        assert int(artifact.labels[idx]) == idx % 3

        expected_row = torch.arange(4, dtype=artifact.embeddings.dtype) + float(idx)
        assert torch.allclose(artifact.embeddings[idx], expected_row)


def test_load_embedding_split_round_trip_without_labels(tmp_path: Path) -> None:
    split_dir = _export_split(
        root_dir=tmp_path / "embeddings",
        split="val",
        num_samples=4,
        embedding_dim=3,
        with_labels=False,
    )

    artifact = load_embedding_split(split_dir)

    assert artifact.manifest.split == "val"
    assert artifact.manifest.num_samples == 4
    assert artifact.manifest.embedding_dim == 3
    assert artifact.manifest.has_labels is False

    assert artifact.embeddings.shape == (4, 3)
    assert artifact.labels is None
    assert artifact.image_ids == [f"val_{idx:05d}" for idx in range(4)]
    assert len(artifact.metadata) == 4


def test_load_embedding_artifacts_loads_multiple_splits(tmp_path: Path) -> None:
    embeddings_root = tmp_path / "embeddings"

    _export_split(
        root_dir=embeddings_root,
        split="train",
        num_samples=6,
        embedding_dim=5,
        with_labels=True,
    )
    _export_split(
        root_dir=embeddings_root,
        split="val",
        num_samples=3,
        embedding_dim=5,
        with_labels=False,
    )

    artifacts = load_embedding_artifacts(embeddings_root)

    assert set(artifacts.keys()) == {"train", "val"}

    assert artifacts["train"].manifest.split == "train"
    assert artifacts["train"].embeddings.shape == (6, 5)
    assert artifacts["train"].labels is not None

    assert artifacts["val"].manifest.split == "val"
    assert artifacts["val"].embeddings.shape == (3, 5)
    assert artifacts["val"].labels is None


def test_load_embedding_split_raises_when_manifest_missing(tmp_path: Path) -> None:
    split_dir = tmp_path / "embeddings" / "train"
    split_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Manifest file not found"):
        load_embedding_split(split_dir)


def test_load_embedding_split_raises_when_shard_missing(tmp_path: Path) -> None:
    split_dir = _export_split(
        root_dir=tmp_path / "embeddings",
        split="train",
        num_samples=4,
        embedding_dim=3,
        with_labels=True,
    )

    (split_dir / "part-00000.pt").unlink()

    with pytest.raises(FileNotFoundError, match="Shard file not found"):
        load_embedding_split(split_dir)


def test_load_embedding_split_raises_when_manifest_num_samples_mismatch(
    tmp_path: Path,
) -> None:
    split_dir = _export_split(
        root_dir=tmp_path / "embeddings",
        split="train",
        num_samples=4,
        embedding_dim=3,
        with_labels=True,
    )

    manifest_path = split_dir / EMBEDDING_MANIFEST_FILENAME
    with manifest_path.open("r", encoding="utf-8") as f:
        manifest_dict = yaml.safe_load(f)

    manifest_dict["num_samples"] = 999

    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(manifest_dict, f, sort_keys=False)

    with pytest.raises(
        ValueError,
        match="Sum of shard num_samples must equal manifest num_samples",
    ):
        load_embedding_split(split_dir)


def test_load_embedding_split_raises_when_duplicate_image_ids(tmp_path: Path) -> None:
    split_dir = _export_split(
        root_dir=tmp_path / "embeddings",
        split="train",
        num_samples=4,
        embedding_dim=3,
        with_labels=True,
    )

    shard_path = split_dir / "part-00000.pt"
    payload = torch.load(shard_path)

    payload["image_ids"][1] = payload["image_ids"][0]
    torch.save(payload, shard_path)

    with pytest.raises(ValueError, match="Duplicate image_ids"):
        load_embedding_split(split_dir)


def test_load_embedding_split_raises_when_shard_sample_count_mismatch(
    tmp_path: Path,
) -> None:
    split_dir = _export_split(
        root_dir=tmp_path / "embeddings",
        split="train",
        num_samples=4,
        embedding_dim=3,
        with_labels=True,
    )

    manifest_path = split_dir / EMBEDDING_MANIFEST_FILENAME
    manifest = EmbeddingManifest.load_yaml(manifest_path)

    broken_manifest_dict = manifest.to_dict()
    broken_manifest_dict["shards"][0]["num_samples"] = 999

    with manifest_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(broken_manifest_dict, f, sort_keys=False)

    with pytest.raises(
        ValueError,
        match="shard num_samples|shard total|manifest num_samples",
    ):
        load_embedding_split(split_dir)