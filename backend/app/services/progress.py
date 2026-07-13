"""Structured progress updates for long-running predict jobs."""

from __future__ import annotations

from typing import Any, Callable

ProgressCallback = Callable[[str | dict[str, Any]], None]


def format_progress_message(update: dict[str, Any]) -> str:
    phase = update.get("phase", "")
    if phase == "chunk":
        model = update.get("model_label") or update.get("model") or "Model"
        chunk = update.get("chunk", 0)
        chunks = update.get("chunks", 0)
        row_start = update.get("row_start", 0)
        row_end = update.get("row_end", 0)
        elapsed = update.get("elapsed_seconds")
        extra = f" · {elapsed:.1f}s" if isinstance(elapsed, (int, float)) else ""
        return (
            f"{model}: scoring chunk {chunk}/{chunks} "
            f"(rows {int(row_start):,}–{int(row_end):,}){extra}"
        )
    if phase == "fit":
        return str(update.get("message") or "Training models…")
    return str(update.get("message") or "Working…")


def emit_progress(callback: ProgressCallback | None, update: str | dict[str, Any]) -> None:
    if callback is None:
        return
    if isinstance(update, dict) and "message" not in update:
        update = {**update, "message": format_progress_message(update)}
    callback(update)
