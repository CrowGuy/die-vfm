from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from die_vfm.artifacts.embedding_loader import load_embedding_split

PAIR_REQUIRED_COLUMNS = [
    "pair_id",
    "did_a",
    "did_b",
    "image_id_a",
    "image_id_b",
]

ANNOTATION_REQUIRED_COLUMNS = [
    "pair_id",
    "review_status",
    "visual_relation",
]

VALID_RELATIONS = ("same", "different", "uncertain")

PAIR_SCORE_COLUMNS = [
    "pair_id",
    "visual_relation",
    "confidence",
    "annotator",
    "reviewed_at",
    "notes",
    "did_a",
    "did_b",
    "image_id_a",
    "image_id_b",
    "artifact_join_key_a",
    "artifact_join_key_b",
    "artifact_image_id_a",
    "artifact_image_id_b",
    "artifact_split_a",
    "artifact_split_b",
    "fine_label_a",
    "fine_label_b",
    "pair_type",
    "candidate_source",
    "cosine_similarity",
    "cosine_distance",
    "l2_distance",
]

UNMATCHED_COLUMNS = [
    "pair_id",
    "visual_relation",
    "join_key_mode",
    "join_key_a",
    "join_key_b",
    "left_found",
    "right_found",
    "did_a",
    "did_b",
    "image_id_a",
    "image_id_b",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate pair benchmark annotations against exported embedding artifacts."
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
        "--embedding-split-dir",
        action="append",
        required=True,
        help=(
            "Embedding split directory containing manifest.yaml and shard payload. "
            "Repeat the argument to load multiple splits."
        ),
    )
    parser.add_argument(
        "--join-key",
        choices=("did", "image_id"),
        default="did",
        help=(
            "Which pair_candidates key to join against embedding artifact image_ids. "
            "Use 'did' for current domain artifact exports."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write pair benchmark outputs.",
    )
    parser.add_argument(
        "--hard-limit",
        type=int,
        default=50,
        help="Number of hardest examples to save per hard-case slice.",
    )
    parser.add_argument(
        "--map-location",
        default="cpu",
        help="torch.load map_location for embedding artifact loading.",
    )
    return parser.parse_args()


def _normalize_required_text(dataframe: pd.DataFrame, column: str) -> pd.Series:
    values = dataframe[column].fillna("").astype(str).str.strip()
    if (values == "").any():
        empty_count = int((values == "").sum())
        raise ValueError(
            f"Column {column!r} contains {empty_count} empty values after normalization."
        )
    return values


def _load_pair_annotations(
    pair_candidates_path: Path,
    annotations_path: Path,
) -> pd.DataFrame:
    pair_df = pd.read_csv(pair_candidates_path)
    annotation_df = pd.read_csv(annotations_path)

    missing_pair = [col for col in PAIR_REQUIRED_COLUMNS if col not in pair_df.columns]
    if missing_pair:
        raise ValueError(
            f"pair_candidates.csv is missing required columns: {missing_pair}"
        )

    missing_annotation = [
        col for col in ANNOTATION_REQUIRED_COLUMNS if col not in annotation_df.columns
    ]
    if missing_annotation:
        raise ValueError(
            f"annotations.csv is missing required columns: {missing_annotation}"
        )

    normalized_pairs = pair_df.copy()
    for column in pair_df.columns:
        normalized_pairs[column] = normalized_pairs[column].fillna("").astype(str)

    normalized_annotations = annotation_df.copy()
    for column in annotation_df.columns:
        normalized_annotations[column] = (
            normalized_annotations[column].fillna("").astype(str)
        )

    normalized_annotations = normalized_annotations.drop_duplicates(
        subset=["pair_id"],
        keep="last",
    )

    merged = normalized_pairs.merge(
        normalized_annotations,
        on="pair_id",
        how="left",
        suffixes=("", "_annotation"),
    )
    merged["review_status"] = merged["review_status"].fillna("").astype(str)
    merged["visual_relation"] = merged["visual_relation"].fillna("").astype(str)

    filtered = merged[
        (merged["review_status"] == "reviewed")
        & (merged["visual_relation"].isin(VALID_RELATIONS))
    ].copy()
    return filtered.reset_index(drop=True)


def _build_embedding_index(
    split_dirs: list[Path],
    *,
    map_location: str,
) -> dict[str, dict[str, Any]]:
    embedding_index: dict[str, dict[str, Any]] = {}

    for split_dir in split_dirs:
        artifact = load_embedding_split(split_dir=split_dir, map_location=map_location)
        for idx, image_id in enumerate(artifact.image_ids):
            join_key = str(image_id)
            if join_key in embedding_index:
                raise ValueError(
                    "Duplicate join key detected across embedding artifacts: "
                    f"{join_key!r}"
                )
            embedding_index[join_key] = {
                "embedding": artifact.embeddings[idx],
                "image_id": image_id,
                "metadata": artifact.metadata[idx],
                "split": artifact.manifest.split,
            }

    return embedding_index


def _resolve_pair_join_keys(row: pd.Series, join_key: str) -> tuple[str, str]:
    if join_key == "did":
        return str(row["did_a"]), str(row["did_b"])
    return str(row["image_id_a"]), str(row["image_id_b"])


def _cosine_similarity(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:
    vector_a = embedding_a.unsqueeze(0).float()
    vector_b = embedding_b.unsqueeze(0).float()
    similarity = F.cosine_similarity(vector_a, vector_b, dim=1)
    return float(similarity.item())


def _l2_distance(
    embedding_a: torch.Tensor,
    embedding_b: torch.Tensor,
) -> float:
    return float(torch.norm(embedding_a.float() - embedding_b.float(), p=2).item())


def _build_pair_scores(
    annotated_pairs: pd.DataFrame,
    *,
    embedding_index: dict[str, dict[str, Any]],
    join_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_rows: list[dict[str, Any]] = []
    unmatched_rows: list[dict[str, Any]] = []

    for _, row in annotated_pairs.iterrows():
        join_key_a, join_key_b = _resolve_pair_join_keys(row, join_key)
        left_record = embedding_index.get(join_key_a)
        right_record = embedding_index.get(join_key_b)

        if left_record is None or right_record is None:
            unmatched_rows.append({
                "pair_id": row["pair_id"],
                "visual_relation": row["visual_relation"],
                "join_key_mode": join_key,
                "join_key_a": join_key_a,
                "join_key_b": join_key_b,
                "left_found": left_record is not None,
                "right_found": right_record is not None,
                "did_a": row.get("did_a", ""),
                "did_b": row.get("did_b", ""),
                "image_id_a": row.get("image_id_a", ""),
                "image_id_b": row.get("image_id_b", ""),
            })
            continue

        cosine_similarity = _cosine_similarity(
            left_record["embedding"],
            right_record["embedding"],
        )
        l2_distance = _l2_distance(
            left_record["embedding"],
            right_record["embedding"],
        )

        matched_rows.append({
            "pair_id": row["pair_id"],
            "visual_relation": row["visual_relation"],
            "confidence": row.get("confidence", ""),
            "annotator": row.get("annotator", ""),
            "reviewed_at": row.get("reviewed_at", ""),
            "notes": row.get("notes", ""),
            "did_a": row.get("did_a", ""),
            "did_b": row.get("did_b", ""),
            "image_id_a": row.get("image_id_a", ""),
            "image_id_b": row.get("image_id_b", ""),
            "artifact_join_key_a": join_key_a,
            "artifact_join_key_b": join_key_b,
            "artifact_image_id_a": left_record["image_id"],
            "artifact_image_id_b": right_record["image_id"],
            "artifact_split_a": left_record["split"],
            "artifact_split_b": right_record["split"],
            "fine_label_a": row.get("fine_label_a", ""),
            "fine_label_b": row.get("fine_label_b", ""),
            "pair_type": row.get("pair_type", ""),
            "candidate_source": row.get("candidate_source", ""),
            "cosine_similarity": cosine_similarity,
            "cosine_distance": 1.0 - cosine_similarity,
            "l2_distance": l2_distance,
        })

    matched_df = pd.DataFrame(matched_rows, columns=PAIR_SCORE_COLUMNS)
    unmatched_df = pd.DataFrame(unmatched_rows, columns=UNMATCHED_COLUMNS)
    return matched_df, unmatched_df


def _describe_metric(values: pd.Series) -> dict[str, float]:
    numeric = values.astype(float)
    return {
        "count": int(numeric.shape[0]),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "p10": float(numeric.quantile(0.10)),
        "p25": float(numeric.quantile(0.25)),
        "p50": float(numeric.quantile(0.50)),
        "p75": float(numeric.quantile(0.75)),
        "p90": float(numeric.quantile(0.90)),
    }


def _compute_auc_like(
    same_scores: pd.Series,
    different_scores: pd.Series,
) -> float | None:
    if same_scores.empty or different_scores.empty:
        return None
    same_tensor = torch.tensor(same_scores.tolist(), dtype=torch.float32)
    different_tensor = torch.tensor(different_scores.tolist(), dtype=torch.float32)
    comparisons = same_tensor[:, None] - different_tensor[None, :]
    wins = (comparisons > 0).float().mean()
    ties = (comparisons == 0).float().mean()
    return float((wins + 0.5 * ties).item())


def _build_summary(
    *,
    annotated_pairs: pd.DataFrame,
    matched_df: pd.DataFrame,
    unmatched_df: pd.DataFrame,
    join_key: str,
    embedding_split_dirs: list[Path],
    pair_candidates_path: Path,
    annotations_path: Path,
) -> dict[str, Any]:
    relation_summary: dict[str, Any] = {}
    for relation in VALID_RELATIONS:
        subset = matched_df[matched_df["visual_relation"] == relation]
        if subset.empty:
            relation_summary[relation] = {
                "count": 0,
                "cosine_similarity": None,
                "cosine_distance": None,
                "l2_distance": None,
            }
            continue
        relation_summary[relation] = {
            "count": int(len(subset)),
            "cosine_similarity": _describe_metric(subset["cosine_similarity"]),
            "cosine_distance": _describe_metric(subset["cosine_distance"]),
            "l2_distance": _describe_metric(subset["l2_distance"]),
        }

    same_scores = matched_df.loc[
        matched_df["visual_relation"] == "same",
        "cosine_similarity",
    ]
    different_scores = matched_df.loc[
        matched_df["visual_relation"] == "different",
        "cosine_similarity",
    ]

    return {
        "benchmark_type": "pair_embedding_similarity",
        "join_key": join_key,
        "input": {
            "pair_candidates_path": str(pair_candidates_path.resolve()),
            "annotations_path": str(annotations_path.resolve()),
            "embedding_split_dirs": [str(path.resolve()) for path in embedding_split_dirs],
        },
        "coverage": {
            "annotated_reviewed_pairs": int(len(annotated_pairs)),
            "matched_pairs": int(len(matched_df)),
            "unmatched_pairs": int(len(unmatched_df)),
            "match_rate": (
                float(len(matched_df) / len(annotated_pairs))
                if len(annotated_pairs) > 0
                else 0.0
            ),
        },
        "relations": relation_summary,
        "separation": {
            "same_vs_different_cosine_auc_like": _compute_auc_like(
                same_scores,
                different_scores,
            )
        },
    }


def _hard_case_slices(pair_scores_df: pd.DataFrame, limit: int) -> dict[str, pd.DataFrame]:
    same_far = (
        pair_scores_df[pair_scores_df["visual_relation"] == "same"]
        .sort_values("cosine_similarity", ascending=True)
        .head(limit)
        .reset_index(drop=True)
    )
    different_close = (
        pair_scores_df[pair_scores_df["visual_relation"] == "different"]
        .sort_values("cosine_similarity", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    uncertain_high = (
        pair_scores_df[pair_scores_df["visual_relation"] == "uncertain"]
        .sort_values("cosine_similarity", ascending=False)
        .head(limit)
        .reset_index(drop=True)
    )
    return {
        "hard_same_far": same_far,
        "hard_different_close": different_close,
        "uncertain_high_similarity": uncertain_high,
    }


def main(args: argparse.Namespace) -> None:
    annotated_pairs = _load_pair_annotations(
        args.pair_candidates_path,
        args.annotations_path,
    )
    if annotated_pairs.empty:
        raise ValueError(
            "No reviewed pair annotations found. "
            "Expected review_status=reviewed and visual_relation in "
            f"{list(VALID_RELATIONS)}."
        )

    embedding_index = _build_embedding_index(
        args.embedding_split_dirs,
        map_location=args.map_location,
    )
    pair_scores_df, unmatched_df = _build_pair_scores(
        annotated_pairs,
        embedding_index=embedding_index,
        join_key=args.join_key,
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pair_scores_path = output_dir / "pair_scores.csv"
    unmatched_path = output_dir / "unmatched_pairs.csv"
    summary_yaml_path = output_dir / "pair_metrics_summary.yaml"
    summary_json_path = output_dir / "pair_metrics_summary.json"

    pair_scores_df.to_csv(pair_scores_path, index=False)
    unmatched_df.to_csv(unmatched_path, index=False)

    hard_slices = _hard_case_slices(pair_scores_df, limit=args.hard_limit)
    for name, dataframe in hard_slices.items():
        dataframe.to_csv(output_dir / f"{name}.csv", index=False)

    summary = _build_summary(
        annotated_pairs=annotated_pairs,
        matched_df=pair_scores_df,
        unmatched_df=unmatched_df,
        join_key=args.join_key,
        embedding_split_dirs=args.embedding_split_dirs,
        pair_candidates_path=args.pair_candidates_path,
        annotations_path=args.annotations_path,
    )
    OmegaConf.save(config=OmegaConf.create(summary), f=summary_yaml_path)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"annotated reviewed pairs: {len(annotated_pairs)}")
    print(f"matched pairs: {len(pair_scores_df)}")
    print(f"unmatched pairs: {len(unmatched_df)}")
    print(f"pair scores: {pair_scores_path}")
    print(f"summary yaml: {summary_yaml_path}")
    print(f"summary json: {summary_json_path}")


if __name__ == "__main__":
    parsed = parse_args()
    parsed.pair_candidates_path = Path(parsed.pair_candidates)
    parsed.annotations_path = Path(parsed.annotations)
    parsed.embedding_split_dirs = [
        Path(path) for path in parsed.embedding_split_dir
    ]
    parsed.output_dir = Path(parsed.output_dir)
    main(parsed)
