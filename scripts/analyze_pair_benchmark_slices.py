from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from omegaconf import OmegaConf

VALID_RELATIONS = ("same", "different", "uncertain")
DEFAULT_SLICE_COLUMNS = (
    ("source_slice", "same_source", "cross_source"),
    ("normality_slice", "both_normal", "both_abnormal", "mixed_normality"),
    ("fine_label_slice", "same_fine_label", "different_fine_label"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run slicing analysis on pair benchmark scores to localize which "
            "pair settings are easiest or hardest for the embedding baseline."
        )
    )
    parser.add_argument(
        "--pair-scores",
        required=True,
        help="Path to pair_scores.csv produced by evaluate_pair_benchmark.py",
    )
    parser.add_argument(
        "--pair-candidates",
        required=True,
        help=(
            "Path to pair_candidates.csv. Used to recover same_lot / "
            "same_machine / same_time_bucket flags for source slicing."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write slicing analysis outputs.",
    )
    parser.add_argument(
        "--confidence",
        default="all",
        choices=("all", "high", "medium", "low"),
        help=(
            "Optional confidence filter. Use 'high' to analyze only the most "
            "trusted reviewed pairs."
        ),
    )
    parser.add_argument(
        "--hard-limit",
        type=int,
        default=20,
        help="Number of hardest examples to keep per slice bucket.",
    )
    return parser.parse_args()


def _read_csv(path: Path, *, required_columns: list[str]) -> pd.DataFrame:
    dataframe = pd.read_csv(path)
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return dataframe


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _normalize_bool(series: pd.Series) -> pd.Series:
    normalized = _normalize_text(series).str.lower()
    return normalized.map(
        {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        }
    ).fillna(False)


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


def _load_enriched_scores(
    pair_scores_path: Path,
    pair_candidates_path: Path,
) -> pd.DataFrame:
    pair_scores = _read_csv(
        pair_scores_path,
        required_columns=[
            "pair_id",
            "visual_relation",
            "confidence",
            "fine_label_a",
            "fine_label_b",
            "cosine_similarity",
            "cosine_distance",
            "l2_distance",
            "pair_type",
            "candidate_source",
        ],
    )
    pair_candidates = _read_csv(
        pair_candidates_path,
        required_columns=[
            "pair_id",
            "same_fine_label",
            "same_lot",
            "same_machine",
            "same_time_bucket",
        ],
    )

    candidate_flags = (
        pair_candidates[
            [
                "pair_id",
                "same_fine_label",
                "same_lot",
                "same_machine",
                "same_time_bucket",
            ]
        ]
        .drop_duplicates(subset=["pair_id"], keep="last")
        .copy()
    )

    for column in (
        "same_fine_label",
        "same_lot",
        "same_machine",
        "same_time_bucket",
    ):
        candidate_flags[column] = _normalize_bool(candidate_flags[column])

    merged = pair_scores.merge(candidate_flags, on="pair_id", how="left")
    for column in (
        "same_fine_label",
        "same_lot",
        "same_machine",
        "same_time_bucket",
    ):
        merged[column] = merged[column].fillna(False).astype(bool)

    merged["visual_relation"] = _normalize_text(merged["visual_relation"])
    merged["confidence"] = _normalize_text(merged["confidence"]).str.lower()
    merged["fine_label_a"] = _normalize_text(merged["fine_label_a"])
    merged["fine_label_b"] = _normalize_text(merged["fine_label_b"])

    merged["same_source"] = (
        merged["same_lot"] & merged["same_machine"] & merged["same_time_bucket"]
    )
    merged["source_slice"] = merged["same_source"].map(
        {True: "same_source", False: "cross_source"}
    )

    merged["is_normal_a"] = merged["fine_label_a"].str.startswith("9_")
    merged["is_normal_b"] = merged["fine_label_b"].str.startswith("9_")
    merged["both_normal"] = merged["is_normal_a"] & merged["is_normal_b"]
    merged["both_abnormal"] = (~merged["is_normal_a"]) & (~merged["is_normal_b"])
    merged["mixed_normality"] = ~(merged["both_normal"] | merged["both_abnormal"])
    merged["normality_slice"] = "mixed_normality"
    merged.loc[merged["both_normal"], "normality_slice"] = "both_normal"
    merged.loc[merged["both_abnormal"], "normality_slice"] = "both_abnormal"

    merged["fine_label_slice"] = merged["same_fine_label"].map(
        {True: "same_fine_label", False: "different_fine_label"}
    )

    return merged


def _apply_confidence_filter(
    dataframe: pd.DataFrame,
    confidence: str,
) -> pd.DataFrame:
    if confidence == "all":
        return dataframe.copy()
    return dataframe[dataframe["confidence"] == confidence].copy()


def _build_slice_summaries(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    summary_payload: dict[str, Any] = {}

    for slice_column, *expected_values in DEFAULT_SLICE_COLUMNS:
        summary_payload[slice_column] = {}

        for slice_value in expected_values:
            subset = dataframe[dataframe[slice_column] == slice_value].copy()
            if subset.empty:
                summary_rows.append(
                    {
                        "slice_name": slice_column,
                        "slice_value": slice_value,
                        "n_pairs": 0,
                        "n_same": 0,
                        "n_different": 0,
                        "n_uncertain": 0,
                        "same_vs_different_cosine_auc_like": None,
                        "same_mean_cosine": None,
                        "different_mean_cosine": None,
                        "same_minus_different_mean_cosine": None,
                    }
                )
                summary_payload[slice_column][slice_value] = {
                    "n_pairs": 0,
                    "relations": {},
                    "same_vs_different_cosine_auc_like": None,
                }
                continue

            relation_counts = (
                subset["visual_relation"].value_counts().reindex(VALID_RELATIONS, fill_value=0)
            )
            same_scores = subset.loc[
                subset["visual_relation"] == "same",
                "cosine_similarity",
            ]
            different_scores = subset.loc[
                subset["visual_relation"] == "different",
                "cosine_similarity",
            ]
            auc_like = _compute_auc_like(same_scores, different_scores)
            same_mean = (
                float(same_scores.mean()) if not same_scores.empty else None
            )
            different_mean = (
                float(different_scores.mean()) if not different_scores.empty else None
            )
            mean_delta = (
                same_mean - different_mean
                if same_mean is not None and different_mean is not None
                else None
            )

            summary_rows.append(
                {
                    "slice_name": slice_column,
                    "slice_value": slice_value,
                    "n_pairs": int(len(subset)),
                    "n_same": int(relation_counts["same"]),
                    "n_different": int(relation_counts["different"]),
                    "n_uncertain": int(relation_counts["uncertain"]),
                    "same_vs_different_cosine_auc_like": auc_like,
                    "same_mean_cosine": same_mean,
                    "different_mean_cosine": different_mean,
                    "same_minus_different_mean_cosine": mean_delta,
                }
            )

            relation_payload: dict[str, Any] = {}
            for relation in VALID_RELATIONS:
                relation_subset = subset[subset["visual_relation"] == relation]
                if relation_subset.empty:
                    relation_payload[relation] = {"count": 0}
                    relation_rows.append(
                        {
                            "slice_name": slice_column,
                            "slice_value": slice_value,
                            "relation": relation,
                            "count": 0,
                            "cosine_mean": None,
                            "cosine_median": None,
                            "cosine_p10": None,
                            "cosine_p90": None,
                            "distance_mean": None,
                            "l2_mean": None,
                        }
                    )
                    continue

                cosine_stats = _describe_metric(relation_subset["cosine_similarity"])
                cosine_distance_stats = _describe_metric(
                    relation_subset["cosine_distance"]
                )
                l2_stats = _describe_metric(relation_subset["l2_distance"])

                relation_payload[relation] = {
                    "count": int(len(relation_subset)),
                    "cosine_similarity": cosine_stats,
                    "cosine_distance": cosine_distance_stats,
                    "l2_distance": l2_stats,
                }
                relation_rows.append(
                    {
                        "slice_name": slice_column,
                        "slice_value": slice_value,
                        "relation": relation,
                        "count": int(len(relation_subset)),
                        "cosine_mean": cosine_stats["mean"],
                        "cosine_median": cosine_stats["median"],
                        "cosine_p10": cosine_stats["p10"],
                        "cosine_p90": cosine_stats["p90"],
                        "distance_mean": cosine_distance_stats["mean"],
                        "l2_mean": l2_stats["mean"],
                    }
                )

            summary_payload[slice_column][slice_value] = {
                "n_pairs": int(len(subset)),
                "relations": relation_payload,
                "same_vs_different_cosine_auc_like": auc_like,
                "same_mean_cosine": same_mean,
                "different_mean_cosine": different_mean,
                "same_minus_different_mean_cosine": mean_delta,
            }

    summary_df = pd.DataFrame(summary_rows)
    relation_df = pd.DataFrame(relation_rows)
    return summary_df, relation_df, summary_payload


def _build_hard_cases(dataframe: pd.DataFrame, limit: int) -> pd.DataFrame:
    hard_rows: list[pd.DataFrame] = []

    for slice_column, *expected_values in DEFAULT_SLICE_COLUMNS:
        for slice_value in expected_values:
            subset = dataframe[dataframe[slice_column] == slice_value].copy()
            if subset.empty:
                continue

            same_far = (
                subset[subset["visual_relation"] == "same"]
                .sort_values("cosine_similarity", ascending=True)
                .head(limit)
                .assign(
                    slice_name=slice_column,
                    slice_value=slice_value,
                    hard_case_type="same_far",
                )
            )
            different_close = (
                subset[subset["visual_relation"] == "different"]
                .sort_values("cosine_similarity", ascending=False)
                .head(limit)
                .assign(
                    slice_name=slice_column,
                    slice_value=slice_value,
                    hard_case_type="different_close",
                )
            )
            hard_rows.extend([same_far, different_close])

    if not hard_rows:
        return pd.DataFrame()

    hard_cases = pd.concat(hard_rows, ignore_index=True)
    leading_columns = [
        "slice_name",
        "slice_value",
        "hard_case_type",
        "pair_id",
        "visual_relation",
        "confidence",
        "cosine_similarity",
        "cosine_distance",
        "l2_distance",
        "fine_label_a",
        "fine_label_b",
        "pair_type",
        "candidate_source",
    ]
    remaining_columns = [
        column for column in hard_cases.columns if column not in leading_columns
    ]
    return hard_cases[leading_columns + remaining_columns]


def main(args: argparse.Namespace) -> None:
    enriched_scores = _load_enriched_scores(
        pair_scores_path=args.pair_scores,
        pair_candidates_path=args.pair_candidates,
    )

    filtered_scores = _apply_confidence_filter(
        enriched_scores,
        confidence=args.confidence,
    )
    if filtered_scores.empty:
        raise ValueError(
            "No rows remain after applying the requested confidence filter."
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    enriched_scores_path = output_dir / "pair_scores_enriched.csv"
    filtered_scores_path = output_dir / "pair_scores_filtered.csv"
    slice_summary_path = output_dir / "slice_summary.csv"
    relation_stats_path = output_dir / "relation_stats.csv"
    hard_cases_path = output_dir / "slice_hard_cases.csv"
    summary_yaml_path = output_dir / "slice_analysis_summary.yaml"
    summary_json_path = output_dir / "slice_analysis_summary.json"

    enriched_scores.to_csv(enriched_scores_path, index=False)
    filtered_scores.to_csv(filtered_scores_path, index=False)

    slice_summary_df, relation_stats_df, summary_payload = _build_slice_summaries(
        filtered_scores
    )
    slice_summary_df.to_csv(slice_summary_path, index=False)
    relation_stats_df.to_csv(relation_stats_path, index=False)

    hard_cases_df = _build_hard_cases(filtered_scores, limit=args.hard_limit)
    hard_cases_df.to_csv(hard_cases_path, index=False)

    summary = {
        "analysis_type": "pair_benchmark_slicing",
        "input": {
            "pair_scores_path": str(args.pair_scores.resolve()),
            "pair_candidates_path": str(args.pair_candidates.resolve()),
        },
        "filters": {
            "confidence": args.confidence,
        },
        "coverage": {
            "input_pair_scores": int(len(enriched_scores)),
            "filtered_pair_scores": int(len(filtered_scores)),
        },
        "slices": summary_payload,
    }
    OmegaConf.save(config=OmegaConf.create(summary), f=summary_yaml_path)
    summary_json_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"input pair scores: {len(enriched_scores)}")
    print(f"filtered pair scores: {len(filtered_scores)}")
    print(f"enriched scores: {enriched_scores_path}")
    print(f"filtered scores: {filtered_scores_path}")
    print(f"slice summary: {slice_summary_path}")
    print(f"relation stats: {relation_stats_path}")
    print(f"hard cases: {hard_cases_path}")
    print(f"summary yaml: {summary_yaml_path}")
    print(f"summary json: {summary_json_path}")


if __name__ == "__main__":
    parsed = parse_args()
    parsed.pair_scores = Path(parsed.pair_scores)
    parsed.pair_candidates = Path(parsed.pair_candidates)
    parsed.output_dir = Path(parsed.output_dir)
    main(parsed)
