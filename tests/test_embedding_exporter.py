from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from die_vfm.artifacts.embedding_artifact import EmbeddingManifest
from die_vfm.artifacts.embedding_exporter import export_split_embeddings


class _ToyDataset(Dataset):
    """Deterministic toy dataset for embedding exporter tests."""

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


def test_export_split_embeddings_writes_manifest_and_shard(tmp_path: Path) -> None:
    model = _ToyModel(embedding_dim=5)
    dataloader = _build_dataloader(
        split="train",
        num_samples=7,
        batch_size=3,
        with_labels=True,
    )

    split_dir = tmp_path / "embeddings" / "train"
    manifest = export_split_embeddings(
        model=model,
        dataloader=dataloader,
        output_dir=split_dir,
        split="train",
        device="cpu",
    )

    assert isinstance(manifest, EmbeddingManifest)
    assert manifest.split == "train"
    assert manifest.num_samples == 7
    assert manifest.embedding_dim == 5
    assert manifest.has_labels is True
    assert manifest.num_shards == 1

    shard_path = split_dir / "part-00000.pt"
    manifest_path = split_dir / "manifest.yaml"

    assert shard_path.exists()
    assert manifest_path.exists()

    payload = torch.load(shard_path)

    assert isinstance(payload["embeddings"], torch.Tensor)
    assert payload["embeddings"].shape == (7, 5)

    assert isinstance(payload["labels"], torch.Tensor)
    assert payload["labels"].shape == (7,)
    assert payload["labels"].dtype == torch.long

    assert payload["image_ids"] == [f"train_{idx:05d}" for idx in range(7)]
    assert len(payload["metadata"]) == 7
    assert payload["metadata"][0]["split"] == "train"
    assert payload["metadata"][0]["index"] == 0
    assert payload["metadata"][-1]["index"] == 6


def test_export_split_embeddings_preserves_alignment(tmp_path: Path) -> None:
    model = _ToyModel(embedding_dim=4)
    dataloader = _build_dataloader(
        split="val",
        num_samples=5,
        batch_size=2,
        with_labels=True,
    )

    split_dir = tmp_path / "embeddings" / "val"
    export_split_embeddings(
        model=model,
        dataloader=dataloader,
        output_dir=split_dir,
        split="val",
        device="cpu",
    )

    payload = torch.load(split_dir / "part-00000.pt")

    embeddings = payload["embeddings"]
    labels = payload["labels"]
    image_ids = payload["image_ids"]
    metadata = payload["metadata"]

    assert embeddings.shape[0] == 5
    assert labels.shape[0] == 5
    assert len(image_ids) == 5
    assert len(metadata) == 5

    for idx in range(5):
        assert image_ids[idx] == f"val_{idx:05d}"
        assert metadata[idx]["split"] == "val"
        assert metadata[idx]["index"] == idx
        assert int(labels[idx]) == idx % 3

        expected_row = torch.arange(4, dtype=embeddings.dtype) + float(idx)
        assert torch.allclose(embeddings[idx], expected_row)


def test_export_split_embeddings_supports_missing_labels(tmp_path: Path) -> None:
    model = _ToyModel(embedding_dim=3)
    dataloader = _build_dataloader(
        split="val",
        num_samples=4,
        batch_size=2,
        with_labels=False,
    )

    split_dir = tmp_path / "embeddings" / "val"
    manifest = export_split_embeddings(
        model=model,
        dataloader=dataloader,
        output_dir=split_dir,
        split="val",
        device="cpu",
    )

    payload = torch.load(split_dir / "part-00000.pt")

    assert manifest.has_labels is False
    assert payload["labels"] is None
    assert payload["embeddings"].shape == (4, 3)
    assert payload["image_ids"] == [f"val_{idx:05d}" for idx in range(4)]
    assert len(payload["metadata"]) == 4


def test_export_split_embeddings_restores_model_training_mode(tmp_path: Path) -> None:
    model = _ToyModel(embedding_dim=2)
    model.train(True)

    dataloader = _build_dataloader(
        split="train",
        num_samples=3,
        batch_size=2,
        with_labels=True,
    )

    split_dir = tmp_path / "embeddings" / "train"
    export_split_embeddings(
        model=model,
        dataloader=dataloader,
        output_dir=split_dir,
        split="train",
        device="cpu",
    )

    assert model.training is True


def test_export_split_embeddings_raises_on_empty_dataloader(tmp_path: Path) -> None:
    model = _ToyModel(embedding_dim=2)
    dataloader = _build_dataloader(
        split="train",
        num_samples=0,
        batch_size=2,
        with_labels=True,
    )

    split_dir = tmp_path / "embeddings" / "train"

    try:
        export_split_embeddings(
            model=model,
            dataloader=dataloader,
            output_dir=split_dir,
            split="train",
            device="cpu",
        )
    except ValueError as exc:
        assert "produced no batches" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty dataloader.")