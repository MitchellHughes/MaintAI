"""Export labeled datasets as CSV or Excel."""

from __future__ import annotations

from io import BytesIO

import pandas as pd
from flask import Blueprint, Response, jsonify, request

from app.services.dataset_store import load_dataframe

export_bp = Blueprint("export", __name__)


def _label_for_row(pred_map: dict, index: int) -> str:
    pred = pred_map.get(index) or pred_map.get(str(index))
    if not pred:
        return ""
    raw = pred.get("final_condition") or pred.get("predicted_condition") or ""
    return str(raw).strip().upper()


def _findings_rows(predictions: list[dict]) -> list[dict]:
    rows = []
    for pred in predictions:
        simple = (pred.get("xai") or {}).get("simple") or {}
        confidence = pred.get("confidence")
        rows.append(
            {
                "row_id": pred.get("row_id"),
                "predicted_condition": pred.get("predicted_condition", ""),
                "final_condition": pred.get("final_condition") or pred.get("predicted_condition", ""),
                "confidence_percent": round(float(confidence) * 1000) / 10 if confidence is not None else "",
                "review_tier": pred.get("confidence_tier") or simple.get("tier") or "",
                "models_agree": pred.get("models_agree") or simple.get("models_agree") or "",
                "models_total": pred.get("models_total") or simple.get("models_total") or "",
                "runner_up": pred.get("runner_up") or simple.get("runner_up") or "",
                "explanation": simple.get("one_liner") or (pred.get("xai") or {}).get("explanation") or "",
                "keywords": ", ".join(simple.get("keywords") or []),
                "actual_label": pred.get("actual_label") or "",
                "model": pred.get("model") or "",
            }
        )
    return rows


def _workbook_bytes(main_df: pd.DataFrame, findings_df: pd.DataFrame) -> bytes:
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        main_df.to_excel(writer, index=False, sheet_name="Dataset")
        findings_df.to_excel(writer, index=False, sheet_name="Findings")
    buffer.seek(0)
    return buffer.getvalue()


@export_bp.route("/api/export", methods=["POST"])
def export_dataset():
    payload = request.get_json(silent=True) or {}
    rows = payload.get("rows")
    fmt = str(payload.get("format", "csv")).lower()

    if not isinstance(rows, list) or not rows:
        return jsonify({"error": "Body must include a non-empty 'rows' list."}), 400

    df = pd.DataFrame(rows)

    if fmt == "xlsx":
        buffer = BytesIO()
        df.to_excel(buffer, index=False, engine="openpyxl")
        buffer.seek(0)
        return Response(
            buffer.getvalue(),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=export.xlsx"},
        )

    if fmt != "csv":
        return jsonify({"error": "format must be 'csv' or 'xlsx'."}), 400

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=export.csv"},
    )


@export_bp.route("/api/export/labeled", methods=["POST"])
def export_labeled_dataset():
    payload = request.get_json(silent=True) or {}
    upload_id = payload.get("upload_id")
    rows = payload.get("rows")
    predictions = payload.get("predictions") or []
    column_name = str(payload.get("column_name") or "FAILURE_MECHANISM").strip() or "FAILURE_MECHANISM"
    fmt = str(payload.get("format", "xlsx")).lower()

    if upload_id:
        try:
            df = load_dataframe(str(upload_id))
        except FileNotFoundError as exc:
            return jsonify({"error": str(exc)}), 404
    elif isinstance(rows, list) and rows:
        df = pd.DataFrame(rows)
    else:
        return jsonify({"error": "Provide upload_id or rows."}), 400

    pred_map = {p.get("row_id"): p for p in predictions if p.get("row_id") is not None}
    df = df.copy()
    df[column_name] = [_label_for_row(pred_map, i) for i in range(len(df))]
    findings_df = pd.DataFrame(_findings_rows(predictions))

    if fmt == "xlsx":
        return Response(
            _workbook_bytes(df, findings_df),
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": "attachment; filename=labeled_export.xlsx"},
        )

    if fmt != "csv":
        return jsonify({"error": "format must be 'csv' or 'xlsx'."}), 400

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return Response(
        csv_bytes,
        mimetype="text/csv; charset=utf-8",
        headers={"Content-Disposition": "attachment; filename=labeled_export.csv"},
    )
