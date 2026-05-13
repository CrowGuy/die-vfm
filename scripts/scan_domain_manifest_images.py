from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, TextIO

from PIL import Image
from PIL import ImageFile
from PIL import UnidentifiedImageError


_SUPPORTED_SINGLE_IMAGE_SOURCE = frozenset({"img1", "img2"})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a domain-style manifest for decode failures and write both a "
            "clean manifest CSV and a bad-image report CSV."
        )
    )
    parser.add_argument("--input", required=True, help="Input manifest CSV path.")
    parser.add_argument(
        "--output-clean",
        required=True,
        help="Output path for the filtered clean manifest CSV.",
    )
    parser.add_argument(
        "--output-bad",
        required=True,
        help="Output path for the bad-image report CSV.",
    )
    parser.add_argument("--did-column", default="DID")
    parser.add_argument("--path-column", default="PATH")
    parser.add_argument("--img1-column", default="IMG_1")
    parser.add_argument("--img2-column", default="IMG_2")
    parser.add_argument(
        "--merge-images",
        action="store_true",
        help="Validate merged IMG_1 + IMG_2 loading behavior.",
    )
    parser.add_argument(
        "--single-image-source",
        choices=("img1", "img2"),
        default="img1",
        help="Image selection policy when merge_images is disabled.",
    )
    parser.add_argument(
        "--allow-truncated-images",
        action="store_true",
        help=(
            "Allow PIL to load truncated images. This is intended only for "
            "debugging, not for formal data cleanup."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help=(
            "Number of concurrent image decode workers. Use 1 to disable "
            "threaded scanning."
        ),
    )
    parser.add_argument(
        "--max-pending-tasks",
        type=int,
        default=0,
        help=(
            "Maximum number of in-flight scan tasks. Set to 0 to use an "
            "automatic bound based on --workers."
        ),
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=100000,
        help=(
            "Emit periodic progress logs every N processed rows. Use 0 to "
            "disable progress logging."
        ),
    )
    return parser.parse_args()


@dataclass(frozen=True)
class ScanStats:
    input_rows: int
    clean_rows: int
    bad_rows: int
    output_clean_path: str
    output_bad_path: str
    merge_images: bool
    single_image_source: str
    allow_truncated_images: bool
    workers: int
    max_pending_tasks: int
    progress_every: int
    elapsed_seconds: float


@dataclass(frozen=True)
class _ScanRowResult:
    row: dict[str, str]
    selected_image_source: str
    selected_image_paths: str
    error_type: str | None
    error_message: str | None

    @property
    def is_bad(self) -> bool:
        return self.error_type is not None


def _normalize_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return text


def _require_columns(fieldnames: list[str] | None, required_columns: list[str]) -> None:
    if fieldnames is None:
        raise ValueError("Input CSV is missing a header row.")
    missing = [column for column in required_columns if column not in fieldnames]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def _resolve_selected_image_paths(
    row: dict[str, str],
    *,
    path_column: str,
    img1_column: str,
    img2_column: str,
    merge_images: bool,
    single_image_source: str,
) -> tuple[list[Path], str]:
    path_text = _normalize_text(row.get(path_column))
    if path_text is None:
        raise ValueError(f"Empty {path_column} field.")

    root_path = Path(path_text)
    img1_text = _normalize_text(row.get(img1_column))
    if img1_text is None:
        raise ValueError(f"Empty {img1_column} field.")
    img1_path = root_path / img1_text

    img2_text = _normalize_text(row.get(img2_column))
    img2_path = root_path / img2_text if img2_text is not None else None

    if merge_images:
        if img2_path is None:
            raise ValueError(
                f"merge_images=true requires non-empty {img2_column}."
            )
        return ([img1_path, img2_path], "merged")

    if single_image_source == "img2" and img2_path is not None:
        return ([img2_path], "img2")

    return ([img1_path], "img1")


def _decode_image(path: Path) -> None:
    with Image.open(path) as image:
        image.load()


def _validate_row_images(
    row: dict[str, str],
    *,
    did_column: str,
    path_column: str,
    img1_column: str,
    img2_column: str,
    merge_images: bool,
    single_image_source: str,
) -> dict[str, str]:
    did = _normalize_text(row.get(did_column)) or ""
    selected_paths, selected_source = _resolve_selected_image_paths(
        row,
        path_column=path_column,
        img1_column=img1_column,
        img2_column=img2_column,
        merge_images=merge_images,
        single_image_source=single_image_source,
    )

    for selected_path in selected_paths:
        if not selected_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {selected_path}")
        _decode_image(selected_path)

    return {
        "did": did,
        "selected_image_source": selected_source,
        "selected_image_paths": "||".join(str(path) for path in selected_paths),
    }


def _scan_manifest_row(
    row: dict[str, str],
    *,
    did_column: str,
    path_column: str,
    img1_column: str,
    img2_column: str,
    merge_images: bool,
    single_image_source: str,
) -> _ScanRowResult:
    selected_source_text = "merged" if merge_images else single_image_source
    selected_paths_text = ""
    try:
        selected_paths, selected_source = _resolve_selected_image_paths(
            row,
            path_column=path_column,
            img1_column=img1_column,
            img2_column=img2_column,
            merge_images=merge_images,
            single_image_source=single_image_source,
        )
        selected_paths_text = "||".join(str(path) for path in selected_paths)
        selected_source_text = selected_source
        _validate_row_images(
            row,
            did_column=did_column,
            path_column=path_column,
            img1_column=img1_column,
            img2_column=img2_column,
            merge_images=merge_images,
            single_image_source=single_image_source,
        )
    except (
        OSError,
        FileNotFoundError,
        UnidentifiedImageError,
        ValueError,
    ) as exc:
        return _ScanRowResult(
            row=row,
            selected_image_source=selected_source_text,
            selected_image_paths=selected_paths_text,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    return _ScanRowResult(
        row=row,
        selected_image_source=selected_source_text,
        selected_image_paths=selected_paths_text,
        error_type=None,
        error_message=None,
    )


def _iter_scan_results(
    rows: Iterator[dict[str, str]],
    *,
    did_column: str,
    path_column: str,
    img1_column: str,
    img2_column: str,
    merge_images: bool,
    single_image_source: str,
    workers: int,
    max_pending_tasks: int,
) -> Iterator[_ScanRowResult]:
    if workers <= 1:
        for row in rows:
            yield _scan_manifest_row(
                dict(row),
                did_column=did_column,
                path_column=path_column,
                img1_column=img1_column,
                img2_column=img2_column,
                merge_images=merge_images,
                single_image_source=single_image_source,
            )
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending_futures: deque[Future[_ScanRowResult]] = deque()
        for row in rows:
            future = executor.submit(
                _scan_manifest_row,
                dict(row),
                did_column=did_column,
                path_column=path_column,
                img1_column=img1_column,
                img2_column=img2_column,
                merge_images=merge_images,
                single_image_source=single_image_source,
            )
            pending_futures.append(future)
            if len(pending_futures) >= max_pending_tasks:
                yield pending_futures.popleft().result()

        while pending_futures:
            yield pending_futures.popleft().result()


def _log_progress(
    *,
    stream: TextIO,
    input_rows: int,
    clean_rows: int,
    bad_rows: int,
    started_at: float,
) -> None:
    elapsed_seconds = max(time.monotonic() - started_at, 1e-9)
    rows_per_second = input_rows / elapsed_seconds
    stream.write(
        "[scan_domain_manifest_images] "
        f"processed={input_rows} "
        f"clean={clean_rows} "
        f"bad={bad_rows} "
        f"elapsed_sec={elapsed_seconds:.1f} "
        f"rows_per_sec={rows_per_second:.1f}\n"
    )
    stream.flush()


def scan_domain_manifest_images(
    *,
    input_path: Path,
    output_clean_path: Path,
    output_bad_path: Path,
    did_column: str,
    path_column: str,
    img1_column: str,
    img2_column: str,
    merge_images: bool,
    single_image_source: str,
    allow_truncated_images: bool,
    workers: int,
    max_pending_tasks: int,
    progress_every: int,
    progress_stream: TextIO | None = None,
) -> ScanStats:
    if single_image_source not in _SUPPORTED_SINGLE_IMAGE_SOURCE:
        raise ValueError(
            "single_image_source must be one of {'img1', 'img2'}, got "
            f"{single_image_source!r}"
        )
    if workers < 1:
        raise ValueError(f"workers must be >= 1, got {workers}")
    if progress_every < 0:
        raise ValueError(f"progress_every must be >= 0, got {progress_every}")

    previous_load_truncated_images = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = bool(allow_truncated_images)

    input_path = input_path.expanduser().resolve()
    output_clean_path = output_clean_path.expanduser().resolve()
    output_bad_path = output_bad_path.expanduser().resolve()
    output_clean_path.parent.mkdir(parents=True, exist_ok=True)
    output_bad_path.parent.mkdir(parents=True, exist_ok=True)

    required_columns = [did_column, path_column, img1_column]
    if merge_images or single_image_source == "img2":
        required_columns.append(img2_column)

    resolved_max_pending_tasks = int(max_pending_tasks)
    if resolved_max_pending_tasks <= 0:
        resolved_max_pending_tasks = max(int(workers) * 8, int(workers))

    input_rows = 0
    clean_rows = 0
    bad_rows = 0
    started_at = time.monotonic()
    resolved_progress_stream = progress_stream or sys.stderr

    try:
        with input_path.open("r", newline="", encoding="utf-8") as input_file:
            reader = csv.DictReader(input_file)
            _require_columns(reader.fieldnames, required_columns)
            manifest_fieldnames = list(reader.fieldnames or [])

            bad_fieldnames = [
                did_column,
                path_column,
                img1_column,
                img2_column,
                "selected_image_source",
                "selected_image_paths",
                "error_type",
                "error_message",
            ]

            with output_clean_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as clean_file, output_bad_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as bad_file:
                clean_writer = csv.DictWriter(clean_file, fieldnames=manifest_fieldnames)
                bad_writer = csv.DictWriter(bad_file, fieldnames=bad_fieldnames)
                clean_writer.writeheader()
                bad_writer.writeheader()

                for result in _iter_scan_results(
                    reader,
                    did_column=did_column,
                    path_column=path_column,
                    img1_column=img1_column,
                    img2_column=img2_column,
                    merge_images=merge_images,
                    single_image_source=single_image_source,
                    workers=int(workers),
                    max_pending_tasks=resolved_max_pending_tasks,
                ):
                    input_rows += 1
                    if result.is_bad:
                        bad_rows += 1
                        bad_writer.writerow(
                            {
                                did_column: result.row.get(did_column, ""),
                                path_column: result.row.get(path_column, ""),
                                img1_column: result.row.get(img1_column, ""),
                                img2_column: result.row.get(img2_column, ""),
                                "selected_image_source": result.selected_image_source,
                                "selected_image_paths": result.selected_image_paths,
                                "error_type": result.error_type,
                                "error_message": result.error_message,
                            }
                        )
                    else:
                        clean_rows += 1
                        clean_writer.writerow(result.row)

                    if progress_every > 0 and input_rows % progress_every == 0:
                        _log_progress(
                            stream=resolved_progress_stream,
                            input_rows=input_rows,
                            clean_rows=clean_rows,
                            bad_rows=bad_rows,
                            started_at=started_at,
                        )
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = previous_load_truncated_images

    elapsed_seconds = time.monotonic() - started_at
    return ScanStats(
        input_rows=input_rows,
        clean_rows=clean_rows,
        bad_rows=bad_rows,
        output_clean_path=str(output_clean_path),
        output_bad_path=str(output_bad_path),
        merge_images=bool(merge_images),
        single_image_source=single_image_source,
        allow_truncated_images=bool(allow_truncated_images),
        workers=int(workers),
        max_pending_tasks=resolved_max_pending_tasks,
        progress_every=int(progress_every),
        elapsed_seconds=float(elapsed_seconds),
    )


def _stats_to_lines(stats: ScanStats) -> Iterator[str]:
    yield f"input_rows: {stats.input_rows}"
    yield f"clean_rows: {stats.clean_rows}"
    yield f"bad_rows: {stats.bad_rows}"
    yield f"output_clean_path: {stats.output_clean_path}"
    yield f"output_bad_path: {stats.output_bad_path}"
    yield f"merge_images: {stats.merge_images}"
    yield f"single_image_source: {stats.single_image_source}"
    yield f"allow_truncated_images: {stats.allow_truncated_images}"
    yield f"workers: {stats.workers}"
    yield f"max_pending_tasks: {stats.max_pending_tasks}"
    yield f"progress_every: {stats.progress_every}"
    yield f"elapsed_seconds: {stats.elapsed_seconds:.3f}"


def main() -> None:
    args = parse_args()
    stats = scan_domain_manifest_images(
        input_path=Path(args.input),
        output_clean_path=Path(args.output_clean),
        output_bad_path=Path(args.output_bad),
        did_column=str(args.did_column),
        path_column=str(args.path_column),
        img1_column=str(args.img1_column),
        img2_column=str(args.img2_column),
        merge_images=bool(args.merge_images),
        single_image_source=str(args.single_image_source),
        allow_truncated_images=bool(args.allow_truncated_images),
        workers=int(args.workers),
        max_pending_tasks=int(args.max_pending_tasks),
        progress_every=int(args.progress_every),
    )
    for line in _stats_to_lines(stats):
        print(line)


if __name__ == "__main__":
    main()
