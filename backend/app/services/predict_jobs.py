"""Background predict jobs for large uploads (avoids proxy timeout + OOM retries)."""

from __future__ import annotations

import gzip
import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.services.predict_service import execute_predict
from app.services.progress import format_progress_message

logger = logging.getLogger(__name__)

ASYNC_ROW_THRESHOLD = 5_000

_lock = threading.Lock()
_jobs: dict[str, "PredictJob"] = {}


def _jobs_root() -> Path:
    import os

    root = Path(os.getenv("MAINTAINER_UPLOAD_DIR", os.getenv("UNCLASSIFIED_DIR", "/app/unclassified")))
    path = root / "predict_jobs"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class PredictJob:
    id: str
    status: str = "queued"
    progress: str = "Queued"
    chunk_progress: dict | None = None
    error: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    finished_at: str | None = None
    timing: dict | None = None
    result_path: str | None = None


def should_run_async(row_count: int) -> bool:
    return row_count > ASYNC_ROW_THRESHOLD


def get_job(job_id: str) -> PredictJob | None:
    with _lock:
        return _jobs.get(job_id)


def _save_result(job_id: str, payload: dict[str, Any]) -> Path:
    path = _jobs_root() / f"{job_id}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path


def load_job_result(job_id: str) -> dict[str, Any] | None:
    path = _jobs_root() / f"{job_id}.json.gz"
    if not path.exists():
        return None
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def start_predict_job(**kwargs) -> str:
    job_id = uuid.uuid4().hex
    job = PredictJob(id=job_id)
    with _lock:
        _jobs[job_id] = job

    def run() -> None:
        with _lock:
            job.status = "running"
            job.progress = "Starting…"

        def on_progress(update: str | dict) -> None:
            with _lock:
                if isinstance(update, dict):
                    job.chunk_progress = update
                    job.progress = update.get("message") or format_progress_message(update)
                else:
                    job.progress = update

        try:
            result = execute_predict(on_progress=on_progress, spool_to_disk=True, **kwargs)
            path = _save_result(job_id, result)
            with _lock:
                job.status = "done"
                job.progress = "Complete"
                job.result_path = str(path)
                job.timing = result.get("timing")
                job.finished_at = datetime.now(timezone.utc).isoformat()
        except Exception as exc:
            logger.exception("Background predict job %s failed", job_id)
            with _lock:
                job.status = "failed"
                job.error = str(exc)
                job.progress = "Failed"
                job.finished_at = datetime.now(timezone.utc).isoformat()

    threading.Thread(target=run, name=f"predict-{job_id[:8]}", daemon=True).start()
    return job_id


def job_status_payload(job: PredictJob, *, include_result: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "job_id": job.id,
        "status": job.status,
        "progress": job.progress,
        "chunk_progress": job.chunk_progress,
        "error": job.error,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "timing": job.timing,
    }
    if include_result and job.status == "done":
        result = load_job_result(job.id)
        if result:
            payload["result"] = result
    return payload
