from __future__ import annotations

import argparse
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

PAIR_COLUMNS = [
    "pair_id",
    "did_a",
    "did_b",
    "image_id_a",
    "image_id_b",
    "image_path_a",
    "image_path_b",
    "fine_label_a",
    "fine_label_b",
    "freq_bin_a",
    "freq_bin_b",
    "lot_a",
    "lot_b",
    "machine_a",
    "machine_b",
    "time_bucket_a",
    "time_bucket_b",
    "pair_type",
    "candidate_source",
    "same_fine_label",
    "same_lot",
    "same_machine",
    "same_time_bucket",
]

REQUIRED_COLUMNS = [
    "did",
    "image_id",
    "image_path",
    "fine_label",
    "freq_bin",
    "lot",
    "machine",
    "time_bucket",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate pair_candidates.csv from a sampled image pool without "
            "requiring product metadata."
        )
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the sampled image pool CSV.",
    )
    parser.add_argument(
        "--output",
        default="pair_candidates.csv",
        help="Path to write pair candidates CSV.",
    )
    parser.add_argument(
        "--summary-output",
        default="pair_candidates_summary.csv",
        help="Path to write a fine-label usage summary CSV.",
    )
    parser.add_argument(
        "--quota-same-source",
        type=int,
        default=250,
        help="Target count for same_label_same_source pairs.",
    )
    parser.add_argument(
        "--quota-cross-source",
        type=int,
        default=250,
        help="Target count for same_label_cross_source pairs.",
    )
    parser.add_argument(
        "--quota-different-label",
        type=int,
        default=500,
        help="Target count for different_label_candidate pairs.",
    )
    parser.add_argument(
        "--max-pairs-per-image",
        type=int,
        default=4,
        help="Maximum number of pairs a single image may appear in.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    return parser.parse_args()


def load_sample_pool(csv_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"sample pool is missing required columns: {missing}")

    normalized = df.copy()
    for column in REQUIRED_COLUMNS:
        normalized[column] = normalized[column].fillna("NA").astype(str).str.strip()
        normalized.loc[normalized[column] == "", column] = "NA"

    normalized = normalized.drop_duplicates(subset=["did"]).copy()
    normalized["bucket_key"] = (
        normalized["lot"]
        + "||"
        + normalized["machine"]
        + "||"
        + normalized["time_bucket"]
    )
    normalized["lot_machine_key"] = (
        normalized["lot"] + "||" + normalized["machine"]
    )
    return normalized.reset_index(drop=True)


def canonical_rows(
    row_a: pd.Series,
    row_b: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    if str(row_a["did"]) <= str(row_b["did"]):
        return row_a, row_b
    return row_b, row_a


def pair_key(row_a: pd.Series, row_b: pd.Series) -> tuple[str, str]:
    left, right = canonical_rows(row_a, row_b)
    return str(left["did"]), str(right["did"])


def build_pair_record(
    row_a: pd.Series,
    row_b: pd.Series,
    *,
    pair_type: str,
    candidate_source: str,
    pair_index: int,
) -> dict[str, object]:
    left, right = canonical_rows(row_a, row_b)
    return {
        "pair_id": f"pair_{pair_index:06d}",
        "did_a": str(left["did"]),
        "did_b": str(right["did"]),
        "image_id_a": str(left["image_id"]),
        "image_id_b": str(right["image_id"]),
        "image_path_a": str(left["image_path"]),
        "image_path_b": str(right["image_path"]),
        "fine_label_a": str(left["fine_label"]),
        "fine_label_b": str(right["fine_label"]),
        "freq_bin_a": str(left["freq_bin"]),
        "freq_bin_b": str(right["freq_bin"]),
        "lot_a": str(left["lot"]),
        "lot_b": str(right["lot"]),
        "machine_a": str(left["machine"]),
        "machine_b": str(right["machine"]),
        "time_bucket_a": str(left["time_bucket"]),
        "time_bucket_b": str(right["time_bucket"]),
        "pair_type": pair_type,
        "candidate_source": candidate_source,
        "same_fine_label": left["fine_label"] == right["fine_label"],
        "same_lot": left["lot"] == right["lot"],
        "same_machine": left["machine"] == right["machine"],
        "same_time_bucket": left["time_bucket"] == right["time_bucket"],
    }


def build_sampling_summary(
    pair_df: pd.DataFrame,
    sample_df: pd.DataFrame,
) -> pd.DataFrame:
    original_counts = (
        sample_df.groupby("fine_label").size().rename("original_image_count")
    )

    pair_counts = pd.concat(
        [
            pair_df[["fine_label_a"]].rename(columns={"fine_label_a": "fine_label"}),
            pair_df[["fine_label_b"]].rename(columns={"fine_label_b": "fine_label"}),
        ],
        ignore_index=True,
    )

    pair_summary = pair_counts.groupby("fine_label").size().rename("pair_appear_count")
    summary = (
        pd.concat([original_counts, pair_summary], axis=1)
        .fillna({"pair_appear_count": 0})
        .reset_index()
    )
    summary["pair_appear_count"] = summary["pair_appear_count"].astype(int)

    return summary.sort_values(
        ["pair_appear_count", "original_image_count"],
        ascending=[False, False],
    ).reset_index(drop=True)


def generate_pair_candidates(
    df: pd.DataFrame,
    *,
    quota_same_source: int,
    quota_cross_source: int,
    quota_different_label: int,
    max_pairs_per_image: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seen_pairs: set[tuple[str, str]] = set()
    image_use_count: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    pair_index = 1

    def try_add_pair(
        row_a: pd.Series,
        row_b: pd.Series,
        *,
        pair_type: str,
        candidate_source: str,
    ) -> bool:
        nonlocal pair_index

        did_left, did_right = pair_key(row_a, row_b)
        if did_left == did_right:
            return False
        if (did_left, did_right) in seen_pairs:
            return False
        if image_use_count[did_left] >= max_pairs_per_image:
            return False
        if image_use_count[did_right] >= max_pairs_per_image:
            return False

        records.append(
            build_pair_record(
                row_a,
                row_b,
                pair_type=pair_type,
                candidate_source=candidate_source,
                pair_index=pair_index,
            )
        )
        pair_index += 1
        seen_pairs.add((did_left, did_right))
        image_use_count[did_left] += 1
        image_use_count[did_right] += 1
        return True

    same_source_added = 0
    same_source_groups = list(df.groupby(["fine_label", "bucket_key"], sort=False))
    rng.shuffle(same_source_groups)

    for (_, _), group in same_source_groups:
        if same_source_added >= quota_same_source:
            break
        if len(group) < 2:
            continue

        shuffled = group.sample(
            frac=1.0,
            random_state=int(rng.integers(0, 1_000_000)),
        ).reset_index(drop=True)

        local_added = 0
        local_cap = min(2, len(shuffled) // 2)
        for index_a, index_b in combinations(range(len(shuffled)), 2):
            if same_source_added >= quota_same_source or local_added >= local_cap:
                break
            if try_add_pair(
                shuffled.iloc[index_a],
                shuffled.iloc[index_b],
                pair_type="same_label_same_source",
                candidate_source="same_label_same_source_rule",
            ):
                same_source_added += 1
                local_added += 1

    cross_source_added = 0
    fine_label_groups = list(df.groupby("fine_label", sort=False))
    rng.shuffle(fine_label_groups)

    for _, group in fine_label_groups:
        if cross_source_added >= quota_cross_source:
            break

        bucket_groups = list(group.groupby("bucket_key", sort=False))
        if len(bucket_groups) < 2:
            continue

        rng.shuffle(bucket_groups)
        per_label_added = 0
        per_label_cap = 3
        sampled_buckets: list[pd.DataFrame] = []
        for _, bucket_df in bucket_groups:
            take = min(2, len(bucket_df))
            sampled = bucket_df.sample(
                n=take,
                random_state=int(rng.integers(0, 1_000_000)),
            )
            sampled_buckets.append(sampled.reset_index(drop=True))

        for bucket_i in range(len(sampled_buckets)):
            if cross_source_added >= quota_cross_source or per_label_added >= per_label_cap:
                break
            for bucket_j in range(bucket_i + 1, len(sampled_buckets)):
                if cross_source_added >= quota_cross_source or per_label_added >= per_label_cap:
                    break

                left_df = sampled_buckets[bucket_i]
                right_df = sampled_buckets[bucket_j]
                left_indices = list(left_df.index)
                right_indices = list(right_df.index)
                rng.shuffle(left_indices)
                rng.shuffle(right_indices)

                added = False
                for left_index in left_indices:
                    for right_index in right_indices:
                        if try_add_pair(
                            left_df.loc[left_index],
                            right_df.loc[right_index],
                            pair_type="same_label_cross_source",
                            candidate_source="same_label_cross_source_rule",
                        ):
                            cross_source_added += 1
                            per_label_added += 1
                            added = True
                            break
                    if added:
                        break

    different_label_added = 0
    candidate_group_specs = [
        ("bucket_key", "different_label_same_bucket_rule"),
        ("lot_machine_key", "different_label_same_lot_machine_rule"),
        ("lot", "different_label_same_lot_rule"),
        ("machine", "different_label_same_machine_rule"),
    ]

    for group_key, source_name in candidate_group_specs:
        if different_label_added >= quota_different_label:
            break

        candidate_groups = list(df.groupby(group_key, sort=False))
        rng.shuffle(candidate_groups)

        for _, group in candidate_groups:
            if different_label_added >= quota_different_label:
                break
            if len(group["fine_label"].unique()) < 2:
                continue

            label_groups = list(group.groupby("fine_label", sort=False))
            rng.shuffle(label_groups)
            local_added = 0
            local_cap = 3

            sampled_by_label: list[tuple[str, pd.DataFrame]] = []
            for label_name, label_df in label_groups:
                take = min(2, len(label_df))
                sampled = label_df.sample(
                    n=take,
                    random_state=int(rng.integers(0, 1_000_000)),
                )
                sampled_by_label.append((label_name, sampled.reset_index(drop=True)))

            for label_i in range(len(sampled_by_label)):
                if different_label_added >= quota_different_label or local_added >= local_cap:
                    break
                label_name_i, left_df = sampled_by_label[label_i]
                for label_j in range(label_i + 1, len(sampled_by_label)):
                    if different_label_added >= quota_different_label or local_added >= local_cap:
                        break
                    label_name_j, right_df = sampled_by_label[label_j]
                    if label_name_i == label_name_j:
                        continue

                    left_indices = list(left_df.index)
                    right_indices = list(right_df.index)
                    rng.shuffle(left_indices)
                    rng.shuffle(right_indices)

                    added = False
                    for left_index in left_indices:
                        for right_index in right_indices:
                            if try_add_pair(
                                left_df.loc[left_index],
                                right_df.loc[right_index],
                                pair_type="different_label_candidate",
                                candidate_source=source_name,
                            ):
                                different_label_added += 1
                                local_added += 1
                                added = True
                                break
                        if added:
                            break

    pair_df = pd.DataFrame(records, columns=PAIR_COLUMNS)
    return pair_df.sort_values(["pair_type", "pair_id"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    sample_df = load_sample_pool(args.input)

    pair_df = generate_pair_candidates(
        sample_df,
        quota_same_source=args.quota_same_source,
        quota_cross_source=args.quota_cross_source,
        quota_different_label=args.quota_different_label,
        max_pairs_per_image=args.max_pairs_per_image,
        seed=args.seed,
    )
    summary_df = build_sampling_summary(pair_df, sample_df)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pair_df.to_csv(output_path, index=False)

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    print(f"sample images: {len(sample_df)}")
    print(f"generated pairs: {len(pair_df)}")
    print(pair_df["pair_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
