from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a round1 domain manifest CSV and dataset config YAML "
            "from sampled train/val pilot CSVs."
        )
    )
    parser.add_argument("--train-csv", required=True, help="round1 pilot train CSV")
    parser.add_argument("--val-csv", required=True, help="round1 pilot val CSV")
    parser.add_argument(
        "--manifest-output",
        required=True,
        help="Output path for round1_domain_manifest.csv",
    )
    parser.add_argument(
        "--dataset-config-output",
        required=True,
        help="Output path for configs/dataset/domain_round1_pilot.yaml",
    )
    parser.add_argument(
        "--did-col",
        default="did",
        help="Column name for sample identifier",
    )
    parser.add_argument(
        "--image-id-col",
        default="image_id",
        help="Column name for image filename or image id",
    )
    parser.add_argument(
        "--image-path-col",
        default="image_path",
        help="Column name for image directory or image file path",
    )
    parser.add_argument(
        "--label-col",
        default="TA_Bin",
        help="Column name for fine label",
    )
    parser.add_argument(
        "--image-path-mode",
        choices=("auto", "directory", "file"),
        default="directory",
        help=(
            "Interpretation of image_path column. "
            "'directory' means PATH=image_path and IMG_1=image_id. "
            "'file' means image_path is a full file path. "
            "'auto' tries file mode only when image_path already ends with image_id."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        metavar=("HEIGHT", "WIDTH"),
        default=(224, 224),
        help="Image size for dataset config",
    )
    parser.add_argument(
        "--require-non-empty-val",
        action="store_true",
        help="Set dataset.require_non_empty_val=true in generated config",
    )
    parser.add_argument(
        "--single-image-source",
        choices=("img1", "img2"),
        default="img1",
        help="Dataset config single_image_source value",
    )
    return parser.parse_args()


def _normalize_required_text(
    dataframe: pd.DataFrame,
    column: str,
    *,
    allow_empty: bool = False,
) -> pd.Series:
    values = dataframe[column].fillna("").astype(str).str.strip()
    if not allow_empty and (values == "").any():
        empty_count = int((values == "").sum())
        raise ValueError(
            f"Column {column!r} contains {empty_count} empty values after normalization."
        )
    return values


def _resolve_manifest_path_and_image(
    image_path_text: str,
    image_id_text: str,
    *,
    mode: str,
) -> tuple[str, str]:
    image_path = Path(image_path_text).expanduser()

    if mode == "directory":
        return str(image_path), image_id_text

    if mode == "file":
        return str(image_path.parent), image_path.name

    candidate = image_path / image_id_text
    if image_path.name == image_id_text:
        return str(image_path.parent), image_path.name
    if candidate.name == image_id_text:
        return str(image_path), image_id_text
    return str(image_path), image_id_text


def _build_manifest_rows(
    dataframe: pd.DataFrame,
    *,
    source_value: str,
    did_col: str,
    image_id_col: str,
    image_path_col: str,
    label_col: str,
    image_path_mode: str,
) -> pd.DataFrame:
    did_values = _normalize_required_text(dataframe, did_col)
    image_id_values = _normalize_required_text(dataframe, image_id_col)
    image_path_values = _normalize_required_text(dataframe, image_path_col)
    label_values = _normalize_required_text(dataframe, label_col)

    manifest_rows: list[dict[str, str]] = []
    for did_text, image_id_text, image_path_text, label_text in zip(
        did_values.tolist(),
        image_id_values.tolist(),
        image_path_values.tolist(),
        label_values.tolist(),
    ):
        path_text, img1_text = _resolve_manifest_path_and_image(
            image_path_text,
            image_id_text,
            mode=image_path_mode,
        )
        manifest_rows.append(
            {
                "DID": did_text,
                "IMG_1": img1_text,
                "Source": source_value,
                "Label": label_text,
                "PATH": path_text,
            }
        )

    manifest_df = pd.DataFrame(manifest_rows)
    if manifest_df["DID"].duplicated().any():
        duplicated = manifest_df.loc[
            manifest_df["DID"].duplicated(), "DID"
        ].tolist()[:5]
        raise ValueError(
            "Generated manifest contains duplicated DID values. "
            f"Examples: {duplicated}"
        )
    return manifest_df


def _validate_manifest_paths(manifest_df: pd.DataFrame) -> None:
    path_values = manifest_df["PATH"].fillna("").astype(str).str.strip()
    if (path_values == "").any():
        raise ValueError("Generated manifest contains empty PATH values.")

    non_absolute = [
        path_text for path_text in path_values.tolist() if not Path(path_text).is_absolute()
    ]
    if non_absolute:
        sample = non_absolute[:5]
        raise ValueError(
            "Generated manifest contains non-absolute PATH values. "
            f"Examples: {sample}"
        )


def _build_label_map(train_df: pd.DataFrame, val_df: pd.DataFrame, label_col: str) -> dict[str, int]:
    labels = pd.concat(
        [
            _normalize_required_text(train_df, label_col),
            _normalize_required_text(val_df, label_col),
        ],
        ignore_index=True,
    )
    unique_labels = sorted(labels.unique().tolist())
    return {label: index for index, label in enumerate(unique_labels)}


def _build_dataset_config(
    manifest_path: Path,
    *,
    image_size: tuple[int, int],
    require_non_empty_val: bool,
    single_image_source: str,
    label_map: dict[str, int],
) -> dict[str, Any]:
    return {
        "name": "domain",
        "manifest_path": str(manifest_path.resolve()),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "merge_images": False,
        "single_image_source": single_image_source,
        "require_non_empty_val": bool(require_non_empty_val),
        "did_field": "DID",
        "img1_field": "IMG_1",
        "img2_field": "IMG_2",
        "source_field": "Source",
        "label_field": "Label",
        "path_field": "PATH",
        "normalize": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "label_map": label_map,
    }


def main() -> None:
    args = parse_args()

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    required_columns = [
        args.did_col,
        args.image_id_col,
        args.image_path_col,
        args.label_col,
    ]
    for column in required_columns:
        if column not in train_df.columns:
            raise KeyError(f"Train CSV is missing required column: {column}")
        if column not in val_df.columns:
            raise KeyError(f"Val CSV is missing required column: {column}")

    manifest_train = _build_manifest_rows(
        train_df,
        source_value="Train",
        did_col=args.did_col,
        image_id_col=args.image_id_col,
        image_path_col=args.image_path_col,
        label_col=args.label_col,
        image_path_mode=args.image_path_mode,
    )
    manifest_val = _build_manifest_rows(
        val_df,
        source_value="Infer",
        did_col=args.did_col,
        image_id_col=args.image_id_col,
        image_path_col=args.image_path_col,
        label_col=args.label_col,
        image_path_mode=args.image_path_mode,
    )
    manifest_df = pd.concat([manifest_train, manifest_val], ignore_index=True)
    _validate_manifest_paths(manifest_df)

    manifest_output_path = Path(args.manifest_output)
    manifest_output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(manifest_output_path, index=False)

    label_map = _build_label_map(train_df, val_df, args.label_col)
    dataset_config = _build_dataset_config(
        manifest_output_path,
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        require_non_empty_val=bool(args.require_non_empty_val),
        single_image_source=args.single_image_source,
        label_map=label_map,
    )

    dataset_config_output_path = Path(args.dataset_config_output)
    dataset_config_output_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(
        config=OmegaConf.create(dataset_config),
        f=dataset_config_output_path,
    )

    print(f"train rows: {len(train_df)}")
    print(f"val rows: {len(val_df)}")
    print(f"manifest rows: {len(manifest_df)}")
    print(f"num labels: {len(label_map)}")
    print(f"manifest output: {manifest_output_path.resolve()}")
    print(f"dataset config output: {dataset_config_output_path.resolve()}")


if __name__ == "__main__":
    main()
