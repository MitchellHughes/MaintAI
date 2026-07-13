# predict.py - unsupervised on-the-fly fit + inference on uploaded rows.

from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from app.services.dataset_store import load_meta, load_rows
from app.services.inference import (
    MODEL_EQUIPMENT,
    MODEL_SEMANTIC,
    MODEL_TOKEN,
    MODEL_UMEC,
)
from app.services.predict_jobs import (
    get_job,
    job_status_payload,
    load_job_result,
    should_run_async,
    start_predict_job,
)
from app.services.predict_service import execute_predict

predict_bp = Blueprint("predict", __name__)
logger = logging.getLogger(__name__)

VALID_MODELS = {MODEL_TOKEN, MODEL_EQUIPMENT, MODEL_SEMANTIC, MODEL_UMEC}


def _merge_part_column(rows: list[dict], text_column: str, part_column: str | None) -> list[dict]:
    if not part_column:
        return rows
    merged: list[dict] = []
    for row in rows:
        if not isinstance(row, dict):
            merged.append(row)
            continue
        copy = dict(row)
        text = str(copy.get(text_column) or "").strip()
        part = str(copy.get(part_column) or "").strip()
        if part:
            part_lower = part.lower()
            text_lower = text.lower()
            if not text_lower or part_lower not in text_lower:
                copy[text_column] = f"{part} — {text}".strip() if text else part
        merged.append(copy)
    return merged


def _parse_predict_request(payload: dict) -> tuple[dict, str | None]:
    upload_id = payload.get("upload_id")
    rows = payload.get("rows")
    text_column = payload.get("text_column")
    raw_label_col = payload.get("label_column")
    label_column = (
        str(raw_label_col).strip()
        if raw_label_col is not None and str(raw_label_col).strip()
        else None
    )
    models = payload.get("models") or [MODEL_UMEC]
    analysis_config = payload.get("analysis_config") or {}
    use_saved_models = bool(payload.get("use_saved_models", False))
    part_column = payload.get("part_column")
    part_column = str(part_column).strip() if part_column is not None and str(part_column).strip() else None
    force_async = bool(payload.get("async", False))
    raw_filename = payload.get("source_filename") or payload.get("filename")
    source_filename = (
        str(raw_filename).strip()
        if raw_filename is not None and str(raw_filename).strip()
        else None
    )

    if upload_id:
        try:
            meta = load_meta(str(upload_id))
            if meta is None:
                raise FileNotFoundError(f"Upload '{upload_id}' was not found or has expired.")
            row_count = int(meta.get("row_count") or 0)
        except FileNotFoundError as exc:
            return {"error": str(exc), "_status": 404}, None
        rows = None
    elif not isinstance(rows, list) or not rows:
        return {"error": "Body must include 'upload_id' or a non-empty 'rows' list."}, None
    else:
        row_count = len(rows)
        rows = _merge_part_column(rows, text_column, part_column)

    if not text_column:
        return {"error": "Body must include 'text_column'."}, None

    if isinstance(models, str):
        models = [models]
    models = [m for m in models if m in VALID_MODELS]
    if not models:
        return {"error": f"'models' must include one of: {sorted(VALID_MODELS)}"}, None

    kwargs = {
        "rows": rows,
        "text_column": text_column,
        "models": models,
        "analysis_config": analysis_config,
        "label_column": label_column,
        "upload_id": str(upload_id) if upload_id else None,
        "source_filename": source_filename,
        "part_column": part_column,
        "use_saved_models": use_saved_models,
    }
    meta = {
        "row_count": row_count if upload_id else len(rows or []),
        "force_async": force_async or should_run_async(row_count if upload_id else len(rows or [])),
    }
    return kwargs, meta


@predict_bp.route("/api/predict", methods=["POST"])
def predict():
    payload = request.get_json(silent=True) or {}
    kwargs, meta = _parse_predict_request(payload)
    if meta is None:
        status = int(kwargs.pop("_status", 400)) if isinstance(kwargs, dict) else 400
        return jsonify(kwargs), status

    if meta["force_async"]:
        job_id = start_predict_job(**kwargs)
        logger.info("Queued async predict job %s for %s rows", job_id, meta["row_count"])
        return jsonify(
            {
                "async": True,
                "job_id": job_id,
                "row_count": meta["row_count"],
                "message": "Large dataset — processing in background. Poll /api/predict/jobs/<job_id>.",
            }
        ), 202

    try:
        result = execute_predict(**kwargs)
        return jsonify(result)
    except Exception as exc:
        logger.exception("Predict failed")
        return jsonify({"error": str(exc)}), 400


@predict_bp.route("/api/predict/jobs/<job_id>", methods=["GET"])
def predict_job_status(job_id: str):
    job = get_job(job_id)
    if job is None:
        stored = load_job_result(job_id)
        if stored:
            return jsonify(
                {
                    "job_id": job_id,
                    "status": "done",
                    "progress": "Complete",
                    "result": stored,
                }
            )
        return jsonify({"error": "Job not found."}), 404

    include_result = request.args.get("result") == "1" or job.status == "done"
    return jsonify(job_status_payload(job, include_result=include_result))
