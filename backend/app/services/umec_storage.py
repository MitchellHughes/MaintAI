"""Storage + audit trail for UMEC predictions."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from umec.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parents[3]
_HISTORY: list[dict] = []


def _history_dir() -> Path:
    """Persist under UNCLASSIFIED_DIR so Docker volume survives container restarts."""
    explicit = os.getenv("MAINTAINER_HISTORY_DIR", "").strip()
    if explicit:
        path = Path(explicit)
    else:
        root = Path(os.getenv("UNCLASSIFIED_DIR", "/app/unclassified"))
        path = root / "history"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _audit_log_path() -> Path:
    return _history_dir() / "maintenance_log.jsonl"


def _ensure_dir() -> None:
    _history_dir()


def warm_history_cache() -> int:
    """Load snapshot index from disk on process start."""
    if _HISTORY:
        return len(_HISTORY)
    directory = _history_dir()
    for path in sorted(directory.glob("*.json")):
        try:
            _HISTORY.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError):
            continue
    return len(_HISTORY)


def _get_config_dir() -> Path:
    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _model_version() -> str:
    try:
        cfg = load_config(_get_config_dir())
        return f"umec-{Path(cfg.project.model_dir).name}"
    except Exception:
        return "umec-unknown"


def _audit(entry: dict) -> None:
    _ensure_dir()
    with _audit_log_path().open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def save_predictions(
    records: list[dict],
    user: str = "anonymous",
    before: list[dict] | None = None,
) -> dict:
    _ensure_dir()
    rec_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    model_version = _model_version()

    record = {
        "id": rec_id,
        "timestamp": ts,
        "model_version": model_version,
        "user": user,
        "num_records": len(records or []),
        "before": before or [],
        "after": records or [],
    }

    safe_ts = ts.replace(":", "-")
    out_file = _history_dir() / f"{safe_ts}_{rec_id}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    _HISTORY.append(record)
    _audit(
        {
            "timestamp": ts,
            "id": rec_id,
            "model_version": model_version,
            "user": user,
            "action": "save_predictions",
            "num_records": record["num_records"],
            "file": out_file.name,
        }
    )

    return {
        "id": rec_id,
        "timestamp": ts,
        "model_version": model_version,
        "num_records": record["num_records"],
        "file": out_file.name,
    }


def _summary(record: dict) -> dict:
    summary = {
        "id": record["id"],
        "timestamp": record["timestamp"],
        "model_version": record["model_version"],
        "user": record["user"],
        "num_records": record["num_records"],
    }
    snap = record.get("snapshot") or {}
    name = (
        snap.get("sourceFilename")
        or (snap.get("prediction") or {}).get("source_filename")
        or ""
    )
    if name:
        summary["source_filename"] = str(name).strip()
    return summary


def list_history() -> list[dict]:
    if _HISTORY:
        return [_summary(r) for r in reversed(_HISTORY)]

    directory = _history_dir()
    if not directory.exists():
        return []
    summaries = []
    for f in sorted(directory.glob("*.json"), reverse=True):
        try:
            summaries.append(_summary(json.loads(f.read_text(encoding="utf-8"))))
        except (OSError, ValueError, KeyError):
            continue
    return summaries


def get_history_item(record_id: str) -> dict | None:
    for r in _HISTORY:
        if r["id"] == record_id:
            return r
    directory = _history_dir()
    if not directory.exists():
        return None
    for f in directory.glob(f"*_{record_id}.json"):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


def save_run_snapshot(snapshot: dict, user: str = "anonymous") -> dict:
    """Persist full workspace + prediction payload for reopen in the UI."""
    _ensure_dir()
    rec_id = str(uuid.uuid4())
    ts = datetime.now(timezone.utc).isoformat()
    model_version = snapshot.get("prediction", {}).get("model_version") or _model_version()
    row_count = (
        snapshot.get("prediction", {}).get("row_count")
        or snapshot.get("row_count")
        or 0
    )
    if not row_count:
        results = snapshot.get("prediction", {}).get("results_by_model") or {}
        first = next(iter(results.values()), [])
        row_count = len(first) if isinstance(first, list) else 0

    record = {
        "id": rec_id,
        "timestamp": ts,
        "type": "run_snapshot",
        "model_version": model_version,
        "user": user,
        "num_records": row_count,
        "snapshot": snapshot,
    }

    safe_ts = ts.replace(":", "-")
    out_file = _history_dir() / f"{safe_ts}_{rec_id}.json"
    out_file.write_text(json.dumps(record, indent=2), encoding="utf-8")

    _HISTORY.append(record)
    _audit(
        {
            "timestamp": ts,
            "id": rec_id,
            "model_version": model_version,
            "user": user,
            "action": "save_run_snapshot",
            "num_records": row_count,
            "file": out_file.name,
        }
    )

    return {
        "id": rec_id,
        "timestamp": ts,
        "model_version": model_version,
        "num_records": row_count,
        "type": "run_snapshot",
    }
