from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a Round2 SSL pilot subset by excluding benchmark DIDs, "
            "keeping train rows only, computing source buckets, and enforcing "
            "a same-source cap."
        )
    )
    parser.add_argument("--input", required=True, help="Input CSV path.")
    parser.add_argument("--output", required=True, help="Output CSV path.")
    parser.add_argument("--did-column", default="DID")
    parser.add_argument("--source-column", default="Source")
    parser.add_argument("--train-source-value", default="Train")
    parser.add_argument("--fine-label-column", default="TA_Bin")
    parser.add_argument("--lot-column", default="LOT_ID")
    parser.add_argument("--machine-column", default="SCAN_TOOL")
    parser.add_argument("--time-column", default="SCAN_DATE_TIME")
    parser.add_argument(
        "--exclude-dids",
        action="append",
        default=[],
        help=(
            "Path to a CSV/TXT file containing one DID per row or a DID column. "
            "Repeat this flag to apply multiple exclusion lists."
        ),
    )
    parser.add_argument("--target-size", type=int, default=500000)
    parser.add_argument("--same-source-cap", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _normalize_text(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.strip()


def _read_excluded_dids(paths: list[Path], *, did_column: str) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        dataframe = pd.read_csv(path)
        if did_column in dataframe.columns:
            values = _normalize_text(dataframe[did_column])
        else:
            values = _normalize_text(dataframe.iloc[:, 0])
        excluded.update(value for value in values.tolist() if value)
    return excluded


def _fill_bucket_component(series: pd.Series, *, sentinel: str) -> pd.Series:
    normalized = _normalize_text(series)
    return normalized.mask(normalized == "", sentinel)


def _enforce_same_source_cap(
    dataframe: pd.DataFrame,
    *,
    same_source_cap: int,
    seed: int,
) -> pd.DataFrame:
    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    capped = (
        shuffled.groupby("round2_source_bucket", group_keys=False)
        .head(same_source_cap)
        .reset_index(drop=True)
    )
    return capped


def _downsample_with_label_coverage(
    dataframe: pd.DataFrame,
    *,
    target_size: int,
    fine_label_column: str,
    seed: int,
) -> pd.DataFrame:
    if len(dataframe) <= target_size:
        return dataframe.reset_index(drop=True)

    shuffled = dataframe.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    guaranteed = (
        shuffled.groupby(fine_label_column, group_keys=False)
        .head(1)
        .drop_duplicates(subset=["round2_did"], keep="first")
    )
    if len(guaranteed) >= target_size:
        return guaranteed.head(target_size).reset_index(drop=True)

    guaranteed_dids = set(guaranteed["round2_did"].tolist())
    remaining = shuffled[~shuffled["round2_did"].isin(guaranteed_dids)].copy()
    remaining_needed = target_size - len(guaranteed)
    sampled_remaining = remaining.head(remaining_needed)
    combined = pd.concat([guaranteed, sampled_remaining], ignore_index=True)
    return combined.reset_index(drop=True)


def build_round2_pilot_subset(
    *,
    input_path: Path,
    output_path: Path,
    did_column: str,
    source_column: str,
    train_source_value: str,
    fine_label_column: str,
    lot_column: str,
    machine_column: str,
    time_column: str,
    exclude_did_paths: list[Path],
    target_size: int,
    same_source_cap: int,
    seed: int,
) -> dict[str, Any]:
    dataframe = pd.read_csv(input_path)
    required_columns = [
        did_column,
        source_column,
        fine_label_column,
        lot_column,
        machine_column,
        time_column,
    ]
    missing = [column for column in required_columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")

    normalized = dataframe.copy()
    normalized["round2_did"] = _normalize_text(normalized[did_column])
    normalized[source_column] = _normalize_text(normalized[source_column])
    normalized[fine_label_column] = _fill_bucket_component(
        normalized[fine_label_column],
        sentinel="UNK_FINE_LABEL",
    )
    normalized[lot_column] = _fill_bucket_component(
        normalized[lot_column],
        sentinel="UNK_LOT",
    )
    normalized[machine_column] = _fill_bucket_component(
        normalized[machine_column],
        sentinel="UNK_MACHINE",
    )
    normalized[time_column] = _fill_bucket_component(
        normalized[time_column],
        sentinel="UNK_TIME",
    )

    train_rows = normalized[
        normalized[source_column].str.lower() == str(train_source_value).strip().lower()
    ].copy()

    excluded_dids = _read_excluded_dids(exclude_did_paths, did_column=did_column)
    if excluded_dids:
        train_rows = train_rows[~train_rows["round2_did"].isin(excluded_dids)].copy()

    train_rows["round2_source_bucket"] = (
        train_rows[fine_label_column]
        + "||"
        + train_rows[lot_column]
        + "||"
        + train_rows[machine_column]
        + "||"
        + train_rows[time_column]
    )

    capped = _enforce_same_source_cap(
        train_rows,
        same_source_cap=same_source_cap,
        seed=seed,
    )
    sampled = _downsample_with_label_coverage(
        capped,
        target_size=target_size,
        fine_label_column=fine_label_column,
        seed=seed,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sampled.to_csv(output_path, index=False)

    return {
        "input_rows": int(len(dataframe)),
        "train_rows_before_exclusion": int(
            (normalized[source_column].str.lower() == str(train_source_value).strip().lower()).sum()
        ),
        "excluded_dids": int(len(excluded_dids)),
        "rows_after_exclusion": int(len(train_rows)),
        "rows_after_same_source_cap": int(len(capped)),
        "rows_written": int(len(sampled)),
        "same_source_cap": int(same_source_cap),
        "target_size": int(target_size),
        "output_path": str(output_path.resolve()),
    }


def main() -> None:
    args = parse_args()
    result = build_round2_pilot_subset(
        input_path=Path(args.input),
        output_path=Path(args.output),
        did_column=args.did_column,
        source_column=args.source_column,
        train_source_value=args.train_source_value,
        fine_label_column=args.fine_label_column,
        lot_column=args.lot_column,
        machine_column=args.machine_column,
        time_column=args.time_column,
        exclude_did_paths=[Path(path) for path in args.exclude_dids],
        target_size=int(args.target_size),
        same_source_cap=int(args.same_source_cap),
        seed=int(args.seed),
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
