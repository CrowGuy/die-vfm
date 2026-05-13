from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
from PIL import Image

from scripts.build_round2_pilot_subset import build_round2_pilot_subset
from scripts.scan_domain_manifest_images import scan_domain_manifest_images


def test_build_round2_pilot_subset_applies_exclusions_and_same_source_cap(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "input.csv"
    exclude_path = tmp_path / "exclude.csv"
    output_path = tmp_path / "round2_pilot.csv"

    rows = []
    for index in range(8):
        rows.append(
            {
                "DID": f"did_{index}",
                "Source": "Train",
                "TA_Bin": "label_a",
                "LOT_ID": "lot_1",
                "SCAN_TOOL": "tool_1",
                "SCAN_DATE_TIME": "time_1",
            }
        )
    rows.append(
        {
            "DID": "did_val",
            "Source": "Infer",
            "TA_Bin": "label_b",
            "LOT_ID": "lot_2",
            "SCAN_TOOL": "tool_2",
            "SCAN_DATE_TIME": "time_2",
        }
    )

    pd.DataFrame(rows).to_csv(input_path, index=False)
    pd.DataFrame({"DID": ["did_0"]}).to_csv(exclude_path, index=False)

    result = build_round2_pilot_subset(
        input_path=input_path,
        output_path=output_path,
        did_column="DID",
        source_column="Source",
        train_source_value="Train",
        fine_label_column="TA_Bin",
        lot_column="LOT_ID",
        machine_column="SCAN_TOOL",
        time_column="SCAN_DATE_TIME",
        exclude_did_paths=[exclude_path],
        target_size=10,
        same_source_cap=6,
        seed=123,
    )

    output_df = pd.read_csv(output_path)
    assert result["rows_written"] == 6
    assert "did_0" not in output_df["round2_did"].tolist()
    assert set(output_df["Source"].tolist()) == {"Train"}
    assert output_df["round2_source_bucket"].nunique() == 1
    assert len(output_df) == 6


def test_scan_domain_manifest_images_filters_truncated_rows(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    good_image_path = image_dir / "good.png"
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(good_image_path)

    valid_bytes = good_image_path.read_bytes()
    bad_image_path = image_dir / "bad.png"
    bad_image_path.write_bytes(valid_bytes[: len(valid_bytes) // 2])

    input_path = tmp_path / "manifest.csv"
    clean_output_path = tmp_path / "clean_manifest.csv"
    bad_output_path = tmp_path / "bad_images.csv"

    pd.DataFrame(
        [
            {
                "DID": "did_good",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "good.png",
                "IMG_2": "",
            },
            {
                "DID": "did_bad",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "bad.png",
                "IMG_2": "",
            },
        ]
    ).to_csv(input_path, index=False)

    stats = scan_domain_manifest_images(
        input_path=input_path,
        output_clean_path=clean_output_path,
        output_bad_path=bad_output_path,
        did_column="DID",
        path_column="PATH",
        img1_column="IMG_1",
        img2_column="IMG_2",
        merge_images=False,
        single_image_source="img1",
        allow_truncated_images=False,
        workers=1,
        max_pending_tasks=0,
        progress_every=0,
    )

    clean_df = pd.read_csv(clean_output_path)
    bad_df = pd.read_csv(bad_output_path)

    assert stats.input_rows == 2
    assert stats.clean_rows == 1
    assert stats.bad_rows == 1
    assert clean_df["DID"].tolist() == ["did_good"]
    assert bad_df["DID"].tolist() == ["did_bad"]
    assert bad_df["error_type"].tolist()[0] in {
        "OSError",
        "UnidentifiedImageError",
    }
    assert "bad.png" in bad_df["selected_image_paths"].iloc[0]


def test_scan_domain_manifest_images_supports_multi_worker_scanning(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    for index in range(3):
        Image.new("RGB", (8, 8), color=(index, index, index)).save(
            image_dir / f"good_{index}.png"
        )

    valid_bytes = (image_dir / "good_0.png").read_bytes()
    (image_dir / "bad.png").write_bytes(valid_bytes[: len(valid_bytes) // 2])

    input_path = tmp_path / "manifest.csv"
    clean_output_path = tmp_path / "clean_manifest.csv"
    bad_output_path = tmp_path / "bad_images.csv"

    pd.DataFrame(
        [
            {
                "DID": "did_good_0",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "good_0.png",
                "IMG_2": "",
            },
            {
                "DID": "did_bad",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "bad.png",
                "IMG_2": "",
            },
            {
                "DID": "did_good_1",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "good_1.png",
                "IMG_2": "",
            },
            {
                "DID": "did_good_2",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "good_2.png",
                "IMG_2": "",
            },
        ]
    ).to_csv(input_path, index=False)

    stats = scan_domain_manifest_images(
        input_path=input_path,
        output_clean_path=clean_output_path,
        output_bad_path=bad_output_path,
        did_column="DID",
        path_column="PATH",
        img1_column="IMG_1",
        img2_column="IMG_2",
        merge_images=False,
        single_image_source="img1",
        allow_truncated_images=False,
        workers=2,
        max_pending_tasks=2,
        progress_every=0,
    )

    clean_df = pd.read_csv(clean_output_path)
    bad_df = pd.read_csv(bad_output_path)

    assert stats.input_rows == 4
    assert stats.clean_rows == 3
    assert stats.bad_rows == 1
    assert stats.workers == 2
    assert stats.max_pending_tasks == 2
    assert clean_df["DID"].tolist() == [
        "did_good_0",
        "did_good_1",
        "did_good_2",
    ]
    assert bad_df["DID"].tolist() == ["did_bad"]


def test_scan_domain_manifest_images_emits_periodic_progress(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()

    for index in range(2):
        Image.new("RGB", (8, 8), color=(index, 0, 0)).save(
            image_dir / f"good_{index}.png"
        )

    input_path = tmp_path / "manifest.csv"
    clean_output_path = tmp_path / "clean_manifest.csv"
    bad_output_path = tmp_path / "bad_images.csv"
    progress_buffer = io.StringIO()

    pd.DataFrame(
        [
            {
                "DID": "did_good_0",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "good_0.png",
                "IMG_2": "",
            },
            {
                "DID": "did_good_1",
                "PATH": str(image_dir.resolve()),
                "IMG_1": "good_1.png",
                "IMG_2": "",
            },
        ]
    ).to_csv(input_path, index=False)

    stats = scan_domain_manifest_images(
        input_path=input_path,
        output_clean_path=clean_output_path,
        output_bad_path=bad_output_path,
        did_column="DID",
        path_column="PATH",
        img1_column="IMG_1",
        img2_column="IMG_2",
        merge_images=False,
        single_image_source="img1",
        allow_truncated_images=False,
        workers=1,
        max_pending_tasks=0,
        progress_every=1,
        progress_stream=progress_buffer,
    )

    progress_text = progress_buffer.getvalue()

    assert stats.input_rows == 2
    assert "[scan_domain_manifest_images]" in progress_text
    assert "processed=1" in progress_text
    assert "processed=2" in progress_text
