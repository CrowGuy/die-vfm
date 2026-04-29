from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

PAIR_REQUIRED_COLUMNS = [
    "pair_id",
    "did_a",
    "did_b",
    "image_id_a",
    "image_id_b",
    "image_path_a",
    "image_path_b",
]

ANNOTATION_REQUIRED_COLUMNS = [
    "pair_id",
    "review_status",
    "visual_relation",
]

VALID_RELATIONS = ("same", "different", "uncertain")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build pair_benchmark_manifest.csv from pair_candidates.csv and "
            "annotations.csv for inference-only embedding export."
        )
    )
    parser.add_argument(
        "--pair-candidates",
        required=True,
        help="Path to pair_candidates.csv",
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="Path to annotations.csv",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output path for pair_benchmark_manifest.csv",
    )
    parser.add_argument(
        "--relations",
        default="same,different,uncertain",
        help=(
            "Comma-separated relations to include from reviewed annotations. "
            "Default: same,different,uncertain"
        ),
    )
    parser.add_argument(
        "--label-mode",
        choices=("empty", "fine_label"),
        default="empty",
        help=(
            "How to fill manifest Label: empty (default) or fine_label from "
            "pair candidates."
        ),
    )
    parser.add_argument(
        "--image-path-mode",
        choices=("directory", "file", "auto"),
        default="directory",
        help=(
            "Interpretation of image_path_* columns. "
            "'directory' means PATH=image_path_* and IMG_1=image_id_*. "
            "'file' means image_path_* already stores full file path. "
            "'auto' treats image_path_* as file path only when basename equals image_id_*."
        ),
    )
    parser.add_argument(
        "--source-value",
        default="Infer",
        help="Manifest Source column value. Default: Infer",
    )
    return parser.parse_args()


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _resolve_relations(raw_relations: str) -> tuple[str, ...]:
    relations = [item.strip().lower() for item in raw_relations.split(",") if item.strip()]
    invalid = [item for item in relations if item not in VALID_RELATIONS]
    if invalid:
        raise ValueError(
            f"Unsupported relations: {invalid}. Allowed values: {list(VALID_RELATIONS)}"
        )
    if not relations:
        raise ValueError("At least one relation must be provided.")
    return tuple(relations)


def _resolve_path_and_image(
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

    if image_path.name == image_id_text:
        return str(image_path.parent), image_path.name
    return str(image_path), image_id_text


def _load_pairs_and_annotations(
    pair_candidates_path: Path,
    annotations_path: Path,
    *,
    relations: tuple[str, ...],
) -> pd.DataFrame:
    pair_df = pd.read_csv(pair_candidates_path)
    annotation_df = pd.read_csv(annotations_path)

    missing_pair = [column for column in PAIR_REQUIRED_COLUMNS if column not in pair_df.columns]
    if missing_pair:
        raise ValueError(f"pair_candidates.csv missing required columns: {missing_pair}")

    missing_annotation = [
        column for column in ANNOTATION_REQUIRED_COLUMNS if column not in annotation_df.columns
    ]
    if missing_annotation:
        raise ValueError(f"annotations.csv missing required columns: {missing_annotation}")

    normalized_pairs = pair_df.copy()
    for column in normalized_pairs.columns:
        normalized_pairs[column] = _normalize_text(normalized_pairs[column])

    normalized_annotations = annotation_df.copy()
    for column in normalized_annotations.columns:
        normalized_annotations[column] = _normalize_text(normalized_annotations[column])

    normalized_annotations = normalized_annotations.drop_duplicates(
        subset=["pair_id"],
        keep="last",
    )
    normalized_annotations["review_status"] = normalized_annotations["review_status"].str.lower()
    normalized_annotations["visual_relation"] = normalized_annotations["visual_relation"].str.lower()

    filtered_annotations = normalized_annotations[
        (normalized_annotations["review_status"] == "reviewed")
        & (normalized_annotations["visual_relation"].isin(relations))
    ].copy()

    merged = normalized_pairs.merge(
        filtered_annotations[["pair_id", "visual_relation"]],
        on="pair_id",
        how="inner",
    )
    return merged.reset_index(drop=True)


def _build_image_rows(
    reviewed_pairs: pd.DataFrame,
    *,
    label_mode: str,
    image_path_mode: str,
    source_value: str,
) -> pd.DataFrame:
    rows: list[dict[str, str]] = []

    for _, row in reviewed_pairs.iterrows():
        for side in ("a", "b"):
            did = row[f"did_{side}"]
            image_id = row[f"image_id_{side}"]
            image_path = row[f"image_path_{side}"]
            fine_label = row.get(f"fine_label_{side}", "")

            path_text, img1_text = _resolve_path_and_image(
                image_path,
                image_id,
                mode=image_path_mode,
            )

            label_text = fine_label if label_mode == "fine_label" else ""
            rows.append(
                {
                    "DID": did,
                    "IMG_1": img1_text,
                    "Source": source_value,
                    "Label": label_text,
                    "PATH": path_text,
                }
            )

    images_df = pd.DataFrame(rows)
    if images_df.empty:
        raise ValueError(
            "No image rows produced. Check reviewed annotations and relation filter."
        )
    return images_df


def _deduplicate_manifest(images_df: pd.DataFrame) -> pd.DataFrame:
    duplicate_mask = images_df.duplicated(subset=["DID"], keep=False)
    duplicates = images_df[duplicate_mask].copy()
    if not duplicates.empty:
        for did_value, group in duplicates.groupby("DID"):
            if (
                group["IMG_1"].nunique() > 1
                or group["PATH"].nunique() > 1
                or group["Label"].nunique() > 1
            ):
                raise ValueError(
                    "Conflicting rows found for the same DID while building "
                    f"manifest. DID={did_value!r}"
                )

    deduped = images_df.drop_duplicates(subset=["DID"], keep="first").copy()
    return deduped.reset_index(drop=True)


def _validate_manifest_paths(manifest_df: pd.DataFrame) -> None:
    path_values = _normalize_text(manifest_df["PATH"])
    if (path_values == "").any():
        raise ValueError("Manifest contains empty PATH values.")

    non_absolute = [path for path in path_values.tolist() if not Path(path).is_absolute()]
    if non_absolute:
        raise ValueError(
            "Manifest contains non-absolute PATH values. "
            f"Examples: {non_absolute[:5]}"
        )


def main() -> None:
    args = parse_args()
    relations = _resolve_relations(args.relations)

    reviewed_pairs = _load_pairs_and_annotations(
        Path(args.pair_candidates),
        Path(args.annotations),
        relations=relations,
    )
    image_rows = _build_image_rows(
        reviewed_pairs,
        label_mode=args.label_mode,
        image_path_mode=args.image_path_mode,
        source_value=args.source_value,
    )
    manifest_df = _deduplicate_manifest(image_rows)
    _validate_manifest_paths(manifest_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_df.to_csv(output_path, index=False)

    relation_counts = (
        reviewed_pairs["visual_relation"].value_counts().to_dict()
        if "visual_relation" in reviewed_pairs.columns
        else {}
    )

    print(f"reviewed pairs included: {len(reviewed_pairs)}")
    print(f"unique DID rows in manifest: {len(manifest_df)}")
    print(f"relation counts: {relation_counts}")
    print(f"manifest output: {output_path.resolve()}")


if __name__ == "__main__":
    main()
