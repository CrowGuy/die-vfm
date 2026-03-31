from __future__ import annotations

from pathlib import Path

import pytest
import torch

from die_vfm.artifacts.embedding_artifact import (
    EMBEDDING_ARTIFACT_FORMAT,
    EMBEDDING_ARTIFACT_TYPE,
    EMBEDDING_ARTIFACT_VERSION,
    EmbeddingManifest,
    EmbeddingShardInfo,
    LoadedEmbeddingSplit,
    default_manifest_path,
    validate_embedding_shard_payload,
    validate_loaded_embedding_split,
)


def _make_manifest() -> EmbeddingManifest:
    return EmbeddingManifest(
        artifact_type=EMBEDDING_ARTIFACT_TYPE,
        artifact_version=EMBEDDING_ARTIFACT_VERSION,
        format=EMBEDDING_ARTIFACT_FORMAT,
        split="train",
        num_samples=4,
        embedding_dim=8,
        dtype="float32",
        has_labels=True,
        num_shards=1,
        shards=[
            EmbeddingShardInfo(
                file_name="part-00000.pt",
                num_samples=4,
            )
        ],
    )


def _make_valid_payload(with_labels: bool = True) -> dict:
    return {
        "embeddings": torch.randn(4, 8, dtype=torch.float32),
        "labels": torch.tensor([0, 1, 0, 1], dtype=torch.long) if with_labels else None,
        "image_ids": [f"img_{idx:05d}" for idx in range(4)],
        "metadata": [{"index": idx} for idx in range(4)],
    }


def test_embedding_shard_info_to_dict_round_trip() -> None:
    shard = EmbeddingShardInfo(file_name="part-00000.pt", num_samples=12)

    data = shard.to_dict()
    rebuilt = EmbeddingShardInfo.from_dict(data)

    assert rebuilt == shard
    assert data["file_name"] == "part-00000.pt"
    assert data["num_samples"] == 12


def test_embedding_shard_info_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="file_name"):
        EmbeddingShardInfo(file_name="", num_samples=1)

    with pytest.raises(ValueError, match="num_samples"):
        EmbeddingShardInfo(file_name="part-00000.pt", num_samples=-1)


def test_embedding_manifest_to_dict_from_dict_round_trip() -> None:
    manifest = _make_manifest()

    data = manifest.to_dict()
    rebuilt = EmbeddingManifest.from_dict(data)

    assert rebuilt == manifest
    assert data["split"] == "train"
    assert data["num_samples"] == 4
    assert data["embedding_dim"] == 8
    assert data["dtype"] == "float32"
    assert data["has_labels"] is True
    assert data["num_shards"] == 1
    assert len(data["shards"]) == 1


def test_embedding_manifest_save_and_load_yaml(tmp_path: Path) -> None:
    manifest = _make_manifest()
    path = tmp_path / "embeddings" / "train" / "manifest.yaml"

    manifest.save_yaml(path)
    loaded = EmbeddingManifest.load_yaml(path)

    assert path.exists()
    assert loaded == manifest


def test_default_manifest_path_returns_expected_path(tmp_path: Path) -> None:
    split_dir = tmp_path / "embeddings" / "val"
    expected = split_dir / "manifest.yaml"

    assert default_manifest_path(split_dir) == expected


def test_embedding_manifest_rejects_invalid_type() -> None:
    with pytest.raises(ValueError, match="artifact_type"):
        EmbeddingManifest(
            artifact_type="wrong_type",
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="train",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=1,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
        )


def test_embedding_manifest_rejects_invalid_version() -> None:
    with pytest.raises(ValueError, match="artifact_version"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version="v999",
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="train",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=1,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
        )


def test_embedding_manifest_rejects_invalid_format() -> None:
    with pytest.raises(ValueError, match="format"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format="npz",
            split="train",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=1,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
        )


def test_embedding_manifest_rejects_empty_split() -> None:
    with pytest.raises(ValueError, match="split"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=1,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
        )


def test_embedding_manifest_rejects_non_positive_embedding_dim() -> None:
    with pytest.raises(ValueError, match="embedding_dim"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="train",
            num_samples=4,
            embedding_dim=0,
            dtype="float32",
            has_labels=True,
            num_shards=1,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
        )


def test_embedding_manifest_rejects_non_positive_num_shards() -> None:
    with pytest.raises(ValueError, match="num_shards"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="train",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=0,
            shards=[],
        )


def test_embedding_manifest_rejects_shard_count_mismatch() -> None:
    with pytest.raises(ValueError, match="len\\(shards\\)=1 vs num_shards=2"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="train",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=2,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
        )


def test_embedding_manifest_rejects_shard_sample_total_mismatch() -> None:
    with pytest.raises(ValueError, match="shard total=3 vs manifest=4"):
        EmbeddingManifest(
            artifact_type=EMBEDDING_ARTIFACT_TYPE,
            artifact_version=EMBEDDING_ARTIFACT_VERSION,
            format=EMBEDDING_ARTIFACT_FORMAT,
            split="train",
            num_samples=4,
            embedding_dim=8,
            dtype="float32",
            has_labels=True,
            num_shards=1,
            shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=3)],
        )


def test_validate_embedding_shard_payload_accepts_valid_payload_with_labels() -> None:
    payload = _make_valid_payload(with_labels=True)

    validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_accepts_valid_payload_without_labels() -> None:
    payload = _make_valid_payload(with_labels=False)

    validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_rejects_missing_keys() -> None:
    payload = _make_valid_payload()
    payload.pop("metadata")

    with pytest.raises(ValueError, match="Missing shard payload keys"):
        validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_rejects_wrong_embedding_rank() -> None:
    payload = _make_valid_payload()
    payload["embeddings"] = torch.randn(4)

    with pytest.raises(ValueError, match="shape \\[N, D\\]"):
        validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_rejects_label_length_mismatch() -> None:
    payload = _make_valid_payload()
    payload["labels"] = torch.tensor([0, 1], dtype=torch.long)

    with pytest.raises(ValueError, match="labels.*match"):
        validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_rejects_image_id_length_mismatch() -> None:
    payload = _make_valid_payload()
    payload["image_ids"] = ["a", "b"]

    with pytest.raises(ValueError, match="image_ids.*match"):
        validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_rejects_duplicate_image_ids() -> None:
    payload = _make_valid_payload()
    payload["image_ids"][1] = payload["image_ids"][0]

    with pytest.raises(ValueError, match="Duplicate image_ids"):
        validate_embedding_shard_payload(payload)


def test_validate_embedding_shard_payload_rejects_metadata_length_mismatch() -> None:
    payload = _make_valid_payload()
    payload["metadata"] = [{"index": 0}]

    with pytest.raises(ValueError, match="metadata.*match"):
        validate_embedding_shard_payload(payload)


def test_loaded_embedding_split_properties() -> None:
    manifest = _make_manifest()
    artifact = LoadedEmbeddingSplit(
        manifest=manifest,
        embeddings=torch.randn(4, 8, dtype=torch.float32),
        labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        image_ids=[f"img_{idx:05d}" for idx in range(4)],
        metadata=[{"index": idx} for idx in range(4)],
    )

    assert artifact.num_samples == 4
    assert artifact.embedding_dim == 8
    assert artifact.has_labels is True


def test_validate_loaded_embedding_split_accepts_valid_artifact() -> None:
    manifest = _make_manifest()
    artifact = LoadedEmbeddingSplit(
        manifest=manifest,
        embeddings=torch.randn(4, 8, dtype=torch.float32),
        labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
        image_ids=[f"img_{idx:05d}" for idx in range(4)],
        metadata=[{"index": idx} for idx in range(4)],
    )

    validate_loaded_embedding_split(artifact)


def test_validate_loaded_embedding_split_rejects_manifest_num_samples_mismatch() -> None:
    manifest = EmbeddingManifest(
        artifact_type=EMBEDDING_ARTIFACT_TYPE,
        artifact_version=EMBEDDING_ARTIFACT_VERSION,
        format=EMBEDDING_ARTIFACT_FORMAT,
        split="train",
        num_samples=5,
        embedding_dim=8,
        dtype="float32",
        has_labels=True,
        num_shards=1,
        shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=5)],
    )

    with pytest.raises(ValueError, match="manifest.num_samples"):
        LoadedEmbeddingSplit(
            manifest=manifest,
            embeddings=torch.randn(4, 8, dtype=torch.float32),
            labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            image_ids=[f"img_{idx:05d}" for idx in range(4)],
            metadata=[{"index": idx} for idx in range(4)],
        )


def test_validate_loaded_embedding_split_rejects_manifest_embedding_dim_mismatch() -> None:
    manifest = EmbeddingManifest(
        artifact_type=EMBEDDING_ARTIFACT_TYPE,
        artifact_version=EMBEDDING_ARTIFACT_VERSION,
        format=EMBEDDING_ARTIFACT_FORMAT,
        split="train",
        num_samples=4,
        embedding_dim=16,
        dtype="float32",
        has_labels=True,
        num_shards=1,
        shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
    )

    with pytest.raises(ValueError, match="manifest.embedding_dim"):
        LoadedEmbeddingSplit(
            manifest=manifest,
            embeddings=torch.randn(4, 8, dtype=torch.float32),
            labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            image_ids=[f"img_{idx:05d}" for idx in range(4)],
            metadata=[{"index": idx} for idx in range(4)],
        )


def test_validate_loaded_embedding_split_rejects_manifest_has_labels_mismatch() -> None:
    manifest = EmbeddingManifest(
        artifact_type=EMBEDDING_ARTIFACT_TYPE,
        artifact_version=EMBEDDING_ARTIFACT_VERSION,
        format=EMBEDDING_ARTIFACT_FORMAT,
        split="train",
        num_samples=4,
        embedding_dim=8,
        dtype="float32",
        has_labels=False,
        num_shards=1,
        shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
    )

    with pytest.raises(ValueError, match="manifest.has_labels"):
        LoadedEmbeddingSplit(
            manifest=manifest,
            embeddings=torch.randn(4, 8, dtype=torch.float32),
            labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            image_ids=[f"img_{idx:05d}" for idx in range(4)],
            metadata=[{"index": idx} for idx in range(4)],
        )


def test_validate_loaded_embedding_split_rejects_manifest_dtype_mismatch() -> None:
    manifest = EmbeddingManifest(
        artifact_type=EMBEDDING_ARTIFACT_TYPE,
        artifact_version=EMBEDDING_ARTIFACT_VERSION,
        format=EMBEDDING_ARTIFACT_FORMAT,
        split="train",
        num_samples=4,
        embedding_dim=8,
        dtype="float16",
        has_labels=True,
        num_shards=1,
        shards=[EmbeddingShardInfo(file_name="part-00000.pt", num_samples=4)],
    )

    with pytest.raises(ValueError, match="manifest.dtype"):
        LoadedEmbeddingSplit(
            manifest=manifest,
            embeddings=torch.randn(4, 8, dtype=torch.float32),
            labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            image_ids=[f"img_{idx:05d}" for idx in range(4)],
            metadata=[{"index": idx} for idx in range(4)],
        )


def test_validate_loaded_embedding_split_rejects_duplicate_image_ids() -> None:
    manifest = _make_manifest()

    with pytest.raises(ValueError, match="Duplicate image_ids"):
        LoadedEmbeddingSplit(
            manifest=manifest,
            embeddings=torch.randn(4, 8, dtype=torch.float32),
            labels=torch.tensor([0, 1, 0, 1], dtype=torch.long),
            image_ids=["dup", "dup", "img_00002", "img_00003"],
            metadata=[{"index": idx} for idx in range(4)],
        )