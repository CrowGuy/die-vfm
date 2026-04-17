from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from die_vfm.datasets.builder import build_dataset
from die_vfm.datasets.domain_dataset import DomainDatasetAdapter


def _write_rgb_image(path: Path, *, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (8, 8), color=color)
    image.save(path)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, index=False)


def _make_cfg(
    *,
    manifest_path: Path,
    image_size: list[int] | None = None,
    merge_images: bool = False,
    single_image_source: str = "img1",
    require_non_empty_val: bool = False,
    label_map: dict[str, int] | None = None,
) -> Any:
    return OmegaConf.create({
        "dataset": {
            "name": "domain",
            "manifest_path": str(manifest_path),
            "image_size": image_size or [8, 8],
            "merge_images": merge_images,
            "single_image_source": single_image_source,
            "require_non_empty_val": require_non_empty_val,
            "did_field": "DID",
            "img1_field": "IMG_1",
            "img2_field": "IMG_2",
            "source_field": "Source",
            "label_field": "Label",
            "path_field": "PATH",
            "normalize": {
                "mean": [0.0, 0.0, 0.0],
                "std": [1.0, 1.0, 1.0],
            },
            "label_map": label_map or {},
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


def test_domain_dataset_single_image_and_label_canonicalization(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))
    _write_rgb_image(image_dir / "b.png", color=(0, 255, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": " 1.0 ",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "sample_2",
            "IMG_1": "b.png",
            "Source": "Train",
            "Label": "01",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        merge_images=False,
        single_image_source="img1",
        label_map={"1": 7},
    )
    dataset = DomainDatasetAdapter.from_config(cfg.dataset, split="train")

    assert len(dataset) == 2
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)
    assert tuple(sample["image"].shape) == (3, 8, 8)
    assert sample["label"] == 7
    assert sample["image_id"] == "sample_1"
    assert sample["meta"]["selected_image_source"] == "img1"
    assert sample["meta"]["canonical_label"] == "1"

    metadata = dataset.get_dataset_metadata()
    assert metadata["dataset_name"] == "domain"
    assert metadata["split"] == "train"
    assert metadata["num_samples"] == 2
    assert metadata["has_labels"] is True
    assert metadata["label_vocabulary"] == ["1"]


def test_domain_dataset_label_map_key_canonicalization_positive(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(7, 8, 9))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "1",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        label_map={"001.000": 11},
    )
    dataset = DomainDatasetAdapter.from_config(cfg.dataset, split="train")

    assert len(dataset) == 1
    assert dataset[0]["label"] == 11
    assert dataset[0]["meta"]["canonical_label"] == "1"


def test_domain_dataset_rejects_invalid_single_image_source_config(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        single_image_source="img3",
        label_map={},
    )
    with pytest.raises(ValueError, match="single_image_source must be one of"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_invalid_image_size_config(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        image_size=[8],
        label_map={},
    )
    with pytest.raises(ValueError, match="image_size must contain exactly two"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_invalid_normalize_mean_config(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map={},
    )
    cfg.dataset.normalize.mean = [0.0, 0.0]
    with pytest.raises(ValueError, match="normalize_mean must contain exactly three"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_invalid_normalize_std_config(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map={},
    )
    cfg.dataset.normalize.std = [1.0, 1.0]
    with pytest.raises(ValueError, match="normalize_std must contain exactly three"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_merge_mode_uses_dual_image_composition(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    _write_rgb_image(image_dir / "img1.png", color=(255, 255, 255))
    _write_rgb_image(image_dir / "img2.png", color=(0, 0, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "img1.png",
            "IMG_2": "img2.png",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        merge_images=True,
        label_map={"ok": 3},
    )
    dataset = DomainDatasetAdapter.from_config(cfg.dataset, split="train")
    sample = dataset[0]

    assert sample["meta"]["selected_image_source"] == "merged"
    image = sample["image"]
    assert torch.allclose(image[0], torch.ones_like(image[0]))
    assert torch.allclose(image[1], torch.zeros_like(image[1]))
    assert torch.allclose(image[2], torch.zeros_like(image[2]))


def test_domain_dataset_train_rejects_mixed_label_presence(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))
    _write_rgb_image(image_dir / "b.png", color=(0, 255, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "sample_2",
            "IMG_1": "b.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={"ok": 1})
    with pytest.raises(
        ValueError,
        match="must not mix labeled and unlabeled",
    ):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_train_split_must_not_be_empty(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "x.png", color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "infer_1",
            "IMG_1": "x.png",
            "Source": "Infer",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={"ok": 1})
    with pytest.raises(ValueError, match="Filtered train split is empty"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_val_rejects_mixed_labels_under_artifact_contract(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))
    _write_rgb_image(image_dir / "b.png", color=(0, 255, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "infer_1",
            "IMG_1": "a.png",
            "Source": "Infer",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "infer_2",
            "IMG_1": "b.png",
            "Source": "Infer",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={"ok": 2})
    with pytest.raises(
        ValueError,
        match="Filtered val split must not mix labeled and unlabeled samples",
    ):
        DomainDatasetAdapter.from_config(cfg.dataset, split="val")


def test_build_dataset_returns_domain_adapter(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "x.png", color=(9, 9, 9))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "x.png",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={"ok": 5})
    dataset = build_dataset(cfg, split="train")

    assert isinstance(dataset, DomainDatasetAdapter)
    assert len(dataset) == 1


def test_domain_dataset_rejects_duplicate_did(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(1, 2, 3))
    _write_rgb_image(image_dir / "b.png", color=(4, 5, 6))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "dup_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
        {
            "DID": "dup_1",
            "IMG_1": "b.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="Duplicate DID in manifest"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_invalid_source_value(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Validation",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="Invalid Source value"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_non_absolute_path(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "",
            "PATH": "relative/images",
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="PATH must be absolute"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_missing_img1_file(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "missing.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="IMG_1 file is missing"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_merge_shape_mismatch(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(image_dir / "a.png")
    Image.new("RGB", (10, 8), color=(0, 255, 0)).save(image_dir / "b.png")

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "IMG_2": "b.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        merge_images=True,
        label_map={},
    )
    with pytest.raises(ValueError, match="Merge images must have the same"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_label_not_in_label_map(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "unknown_label",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={"ok": 1})
    with pytest.raises(ValueError, match="Manifest label not found in label_map"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_single_image_source_img2_falls_back_to_img1(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "IMG_2": "",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        single_image_source="img2",
        label_map={},
    )
    dataset = DomainDatasetAdapter.from_config(cfg.dataset, split="train")
    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 8, 8)
    assert sample["meta"]["selected_image_source"] == "img1"


def test_domain_dataset_rejects_missing_manifest_path(tmp_path: Path) -> None:
    manifest_path = tmp_path / "missing_manifest.csv"
    cfg = _make_cfg(manifest_path=manifest_path, label_map={})

    with pytest.raises(ValueError, match="Manifest path does not exist"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_manifest_path_that_is_not_file(
    tmp_path: Path,
) -> None:
    manifest_dir = tmp_path / "manifest_dir"
    manifest_dir.mkdir()
    cfg = _make_cfg(manifest_path=manifest_dir, label_map={})

    with pytest.raises(ValueError, match="Manifest path must point to a CSV file"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_manifest_missing_required_column(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={"ok": 1})
    with pytest.raises(KeyError, match="Missing required manifest column: Source"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_manifest_row_with_empty_did(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="has empty DID field"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_manifest_row_with_empty_path(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "Source": "Train",
            "Label": "",
            "PATH": "",
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="has empty PATH field"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_manifest_row_with_empty_img1(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="has empty IMG_1 field"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_path_that_is_not_directory(
    tmp_path: Path,
) -> None:
    image_file = tmp_path / "image_as_path.png"
    _write_rgb_image(image_file, color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "image_as_path.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_file.resolve()),
        },
    ])

    cfg = _make_cfg(manifest_path=manifest_path, label_map={})
    with pytest.raises(ValueError, match="PATH is not a directory"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_missing_img2_file_in_merge_mode(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "IMG_2": "missing_img2.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        merge_images=True,
        label_map={},
    )
    with pytest.raises(ValueError, match="IMG_2 file is missing"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_missing_img2_file_when_img2_is_selected(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "a.png", color=(255, 0, 0))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "sample_1",
            "IMG_1": "a.png",
            "IMG_2": "missing_img2.png",
            "Source": "Train",
            "Label": "",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        single_image_source="img2",
        label_map={},
    )
    with pytest.raises(ValueError, match="IMG_2 file is missing"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_empty_val_when_inference_policy_is_enabled(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _write_rgb_image(image_dir / "train.png", color=(1, 2, 3))

    manifest_path = tmp_path / "manifest.csv"
    _write_manifest(manifest_path, [
        {
            "DID": "train_1",
            "IMG_1": "train.png",
            "Source": "Train",
            "Label": "ok",
            "PATH": str(image_dir.resolve()),
        },
    ])

    cfg = _make_cfg(
        manifest_path=manifest_path,
        require_non_empty_val=True,
        label_map={"ok": 1},
    )
    with pytest.raises(
        ValueError,
        match="Filtered val split is empty under inference-only policy",
    ):
        DomainDatasetAdapter.from_config(cfg.dataset, split="val")


def test_domain_dataset_rejects_non_mapping_label_map(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map=["ok"],
    )

    with pytest.raises(TypeError, match="label_map must be a mapping"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_empty_label_map_key(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map={"": 1},
    )

    with pytest.raises(ValueError, match="label_map contains an empty key"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_non_int_label_map_value(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map={"ok": "1"},
    )

    with pytest.raises(TypeError, match="label_map values must be int"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_bool_label_map_value(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map={"ok": True},
    )

    with pytest.raises(TypeError, match="label_map values must be int"):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")


def test_domain_dataset_rejects_conflicting_canonical_label_map_keys(
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        manifest_path=tmp_path / "unused_manifest.csv",
        label_map={"1": 1, "01": 2},
    )

    with pytest.raises(
        ValueError,
        match="label_map contains conflicting canonical keys",
    ):
        DomainDatasetAdapter.from_config(cfg.dataset, split="train")
