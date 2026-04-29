from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

PAIR_REQUIRED_COLUMNS = [
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

ANNOTATION_COLUMNS = [
    "pair_id",
    "review_status",
    "visual_relation",
    "confidence",
    "annotator",
    "notes",
    "reviewed_at",
]


class AnnotationUpdate(BaseModel):
    review_status: Literal["pending", "reviewed", "skipped"] = "reviewed"
    visual_relation: Literal["same", "different", "uncertain"] | None = None
    confidence: Literal["high", "medium", "low"] | None = None
    annotator: str = Field(default="", max_length=200)
    notes: str = Field(default="", max_length=5000)


class ReviewStore:
    def __init__(self, pair_candidates_path: Path, annotations_path: Path) -> None:
        self._lock = Lock()
        self.pair_candidates_path = pair_candidates_path.resolve()
        self.annotations_path = annotations_path.resolve()
        self._pairs = self._load_pairs()
        self._annotations = self._load_annotations()

    def _load_pairs(self) -> pd.DataFrame:
        pair_df = pd.read_csv(self.pair_candidates_path)
        missing = [
            column
            for column in PAIR_REQUIRED_COLUMNS
            if column not in pair_df.columns
        ]
        if missing:
            raise ValueError(
                f"pair_candidates.csv is missing required columns: {missing}"
            )

        normalized = pair_df.copy()
        for column in PAIR_REQUIRED_COLUMNS:
            normalized[column] = normalized[column].fillna("").astype(str)

        if normalized["pair_id"].duplicated().any():
            raise ValueError("pair_candidates.csv contains duplicated pair_id values")

        return normalized.set_index("pair_id", drop=False).sort_index()

    def _load_annotations(self) -> pd.DataFrame:
        if not self.annotations_path.exists():
            empty = pd.DataFrame(columns=ANNOTATION_COLUMNS)
            return empty.set_index("pair_id", drop=False)

        annotation_df = pd.read_csv(self.annotations_path)
        for column in ANNOTATION_COLUMNS:
            if column not in annotation_df.columns:
                annotation_df[column] = ""

        normalized = annotation_df[ANNOTATION_COLUMNS].copy()
        normalized = normalized.fillna("").astype(str)
        if normalized.empty:
            return normalized.set_index("pair_id", drop=False)

        normalized = normalized.drop_duplicates(subset=["pair_id"], keep="last")
        return normalized.set_index("pair_id", drop=False).sort_index()

    def _default_annotation(self, pair_id: str) -> dict[str, str]:
        return {
            "pair_id": pair_id,
            "review_status": "pending",
            "visual_relation": "",
            "confidence": "",
            "annotator": "",
            "notes": "",
            "reviewed_at": "",
        }

    def _merged_pair(self, pair_id: str) -> dict[str, str]:
        if pair_id not in self._pairs.index:
            raise KeyError(pair_id)

        pair_record = self._pairs.loc[pair_id].to_dict()
        annotation = (
            self._annotations.loc[pair_id].to_dict()
            if pair_id in self._annotations.index
            else self._default_annotation(pair_id)
        )
        merged = {**pair_record, **annotation}
        merged["image_url_a"] = f"/api/pairs/{pair_id}/image/a"
        merged["image_url_b"] = f"/api/pairs/{pair_id}/image/b"
        return merged

    def list_pairs(self, review_status: str) -> list[dict[str, str]]:
        items: list[dict[str, str]] = []
        for pair_id in self._pairs.index.tolist():
            merged = self._merged_pair(pair_id)
            if review_status != "all" and merged["review_status"] != review_status:
                continue
            items.append(
                {
                    "pair_id": pair_id,
                    "pair_type": merged["pair_type"],
                    "candidate_source": merged["candidate_source"],
                    "did_a": merged["did_a"],
                    "did_b": merged["did_b"],
                    "fine_label_a": merged["fine_label_a"],
                    "fine_label_b": merged["fine_label_b"],
                    "review_status": merged["review_status"],
                    "visual_relation": merged["visual_relation"],
                    "confidence": merged["confidence"],
                }
            )
        return items

    def get_pair(self, pair_id: str) -> dict[str, str]:
        return self._merged_pair(pair_id)

    def get_image_path(self, pair_id: str, side: str) -> Path:
        pair = self._merged_pair(pair_id)
        image_path_key = "image_path_a" if side == "a" else "image_path_b"
        image_id_key = "image_id_a" if side == "a" else "image_id_b"
        base_path = Path(pair[image_path_key])
        image_id = pair[image_id_key]

        # Current pair-review contract expects the concrete image file to live at
        # <image_path>/<image_id>. Keep a file-path fallback so older CSVs that
        # already store a full file path in image_path_* still work.
        candidate_paths = [base_path / image_id, base_path]
        for candidate in candidate_paths:
            if candidate.exists():
                return candidate

        attempted = ", ".join(str(path) for path in candidate_paths)
        raise FileNotFoundError(attempted)

    def summary(self) -> dict[str, object]:
        counts = {"pending": 0, "reviewed": 0, "skipped": 0}
        pair_types = self._pairs["pair_type"].value_counts().to_dict()
        for pair_id in self._pairs.index.tolist():
            status = self._merged_pair(pair_id)["review_status"]
            counts[status] = counts.get(status, 0) + 1
        return {
            "pair_candidates_path": str(self.pair_candidates_path),
            "annotations_path": str(self.annotations_path),
            "total_pairs": int(len(self._pairs)),
            "review_status_counts": counts,
            "pair_type_counts": pair_types,
        }

    def save_annotation(
        self,
        pair_id: str,
        update: AnnotationUpdate,
        *,
        reviewed_at: str,
    ) -> dict[str, str]:
        if pair_id not in self._pairs.index:
            raise KeyError(pair_id)
        if update.review_status == "reviewed" and update.visual_relation is None:
            raise ValueError(
                "visual_relation is required when review_status=reviewed"
            )

        record = {
            "pair_id": pair_id,
            "review_status": update.review_status,
            "visual_relation": update.visual_relation or "",
            "confidence": update.confidence or "",
            "annotator": update.annotator.strip(),
            "notes": update.notes,
            "reviewed_at": reviewed_at,
        }

        with self._lock:
            self._annotations.loc[pair_id] = record
            self._persist_annotations()

        return self._merged_pair(pair_id)

    def _persist_annotations(self) -> None:
        output_df = self._annotations.reset_index(drop=True)[ANNOTATION_COLUMNS]
        self.annotations_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.annotations_path.with_suffix(".tmp")
        output_df.to_csv(tmp_path, index=False)
        tmp_path.replace(self.annotations_path)


def create_app(pair_candidates_path: Path, annotations_path: Path) -> FastAPI:
    static_dir = Path(__file__).resolve().parent / "pair_review_static"
    store = ReviewStore(pair_candidates_path, annotations_path)

    app = FastAPI(title="Pair Review Tool")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (static_dir / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/summary")
    def summary() -> dict[str, object]:
        return store.summary()

    @app.get("/api/pairs")
    def list_pairs(review_status: str = "all") -> dict[str, object]:
        allowed_status = {"all", "pending", "reviewed", "skipped"}
        if review_status not in allowed_status:
            raise HTTPException(
                status_code=400,
                detail=f"review_status must be one of {sorted(allowed_status)}",
            )
        return {"items": store.list_pairs(review_status)}

    @app.get("/api/pairs/{pair_id}")
    def get_pair(pair_id: str) -> dict[str, str]:
        try:
            return store.get_pair(pair_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="pair not found") from exc

    @app.get("/api/pairs/{pair_id}/image/{side}")
    def get_image(pair_id: str, side: str) -> FileResponse:
        if side not in {"a", "b"}:
            raise HTTPException(status_code=400, detail="side must be 'a' or 'b'")
        try:
            image_path = store.get_image_path(pair_id, side)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="pair not found") from exc
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=404,
                detail=f"image not found: {exc}",
            ) from exc
        return FileResponse(image_path)

    @app.post("/api/annotations/{pair_id}")
    def save_annotation(
        pair_id: str,
        payload: AnnotationUpdate,
    ) -> dict[str, str]:
        try:
            return store.save_annotation(
                pair_id,
                payload,
                reviewed_at=pd.Timestamp.utcnow().isoformat(),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="pair not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    return app
