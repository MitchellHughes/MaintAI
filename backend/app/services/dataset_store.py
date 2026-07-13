"""Server-side storage for uploaded datasets (avoids re-posting large row payloads)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd

PREVIEW_ROW_LIMIT = 200
CLIENT_ROW_LIMIT = 5000


def _upload_root() -> Path:
    root = Path(os.getenv("MAINTAINER_UPLOAD_DIR", os.getenv("UNCLASSIFIED_DIR", "/app/unclassified")))
    path = root / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def store_dataframe(df: pd.DataFrame, *, filename: str = "") -> str:
    upload_id = uuid.uuid4().hex
    folder = _upload_root() / upload_id
    folder.mkdir(parents=True, exist_ok=True)
    df.fillna("").astype(str).to_csv(folder / "data.csv", index=False)
    meta = {
        "filename": filename,
        "row_count": int(len(df)),
        "columns": [str(c) for c in df.columns],
    }
    (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    return upload_id


def load_meta(upload_id: str) -> dict | None:
    meta_path = _upload_root() / upload_id / "meta.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def load_dataframe(upload_id: str) -> pd.DataFrame:
    csv_path = _upload_root() / upload_id / "data.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Upload '{upload_id}' was not found or has expired.")
    return pd.read_csv(csv_path, dtype=str, keep_default_na=False).fillna("")


def load_rows(upload_id: str) -> list[dict]:
    df = load_dataframe(upload_id)
    rows = df.to_dict(orient="records")
    for index, row in enumerate(rows):
        row["id"] = index
    return rows


def dataframe_for_predict(
    upload_id: str,
    *,
    text_column: str,
    part_column: str | None = None,
) -> pd.DataFrame:
    """Load upload as DataFrame only (no list[dict]) — much lower RAM on large files."""
    df = load_dataframe(upload_id)
    df = df.copy()
    df["id"] = range(len(df))
    if part_column and part_column in df.columns:
        merged_texts = []
        for _, row in df.iterrows():
            text = str(row.get(text_column) or "").strip()
            part = str(row.get(part_column) or "").strip()
            if part:
                part_lower = part.lower()
                text_lower = text.lower()
                if not text_lower or part_lower not in text_lower:
                    text = f"{part} — {text}".strip() if text else part
            merged_texts.append(text)
        df[text_column] = merged_texts
    return df


def rows_for_client(df: pd.DataFrame) -> tuple[list[dict], bool, int]:
    """Return rows for the browser, preview-only flag, and total count."""
    total = len(df)
    preview_only = total > CLIENT_ROW_LIMIT
    limit = PREVIEW_ROW_LIMIT if preview_only else total
    subset = df.head(limit).fillna("").astype(str)
    rows = subset.to_dict(orient="records")
    for index, row in enumerate(rows):
        row["id"] = index
    return rows, preview_only, total
