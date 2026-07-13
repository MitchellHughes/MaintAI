# upload.py - accepts a CSV or Excel file, validates it, returns parsed rows.
# Thin route: parsing/validation only, no ML logic. Parsing is delegated to
# umec.data.io.read_data_buffer so CSV and Excel share one code path.

from pathlib import Path

from flask import Blueprint, jsonify, request

from app.services.dataset_store import rows_for_client, store_dataframe
from umec.data.io import read_data_buffer

upload_bp = Blueprint("upload", __name__)

REQUIRED_COLUMNS = []
ALLOWED_EXTENSIONS = {".csv", ".xlsx", ".xlsm", ".xls"}


@upload_bp.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"valid": False, "errors": ["No file part in request."]}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"valid": False, "errors": ["No file selected."]}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return jsonify(
            {"valid": False, "errors": ["File must be a .csv, .xlsx, or .xls"]}
        ), 400

    try:
        df = read_data_buffer(file.read(), file.filename)
    except Exception as exc:
        return jsonify({"valid": False, "errors": [f"Could not read file: {exc}"]}), 400

    columns = [str(c) for c in df.columns]
    errors = []

    missing = [c for c in REQUIRED_COLUMNS if c not in columns]
    if missing:
        errors.append(f"Missing required column(s): {', '.join(missing)}")

    rows = []
    preview_only = False
    row_count = len(df)
    upload_id = None

    if not missing:
        df = df.fillna("").astype(str)
        if not len(df):
            errors.append("File has headers but no data rows.")
        else:
            upload_id = store_dataframe(df, filename=file.filename)
            rows, preview_only, row_count = rows_for_client(df)

    valid = not errors
    return jsonify(
        {
            "valid": valid,
            "errors": errors,
            "columns": columns,
            "rows": rows,
            "upload_id": upload_id,
            "row_count": row_count,
            "preview_only": preview_only,
            "filename": file.filename if upload_id else None,
        }
    ), (200 if valid else 400)
