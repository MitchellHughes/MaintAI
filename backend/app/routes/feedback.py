# feedback.py - persists user-edited predictions + writes the audit trail.

from flask import Blueprint, jsonify, request

from app.services.active_learning import merge_keyword_suggestions, suggest_keywords_from_edits
from app.services.umec_storage import save_predictions

feedback_bp = Blueprint("feedback", __name__)


@feedback_bp.route("/api/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return jsonify({"error": "Body must include a non-empty 'records' list."}), 400

    saved = save_predictions(
        records=records,
        user=payload.get("user", "anonymous"),
        before=payload.get("before"),
    )
    return jsonify(saved), 201


@feedback_bp.route("/api/feedback/active-learning", methods=["POST"])
def active_learning():
    payload = request.get_json(silent=True) or {}
    edits = payload.get("edits") or payload.get("predictions")
    categories = payload.get("custom_categories") or payload.get("categories")
    if not isinstance(edits, list) or not edits:
        return jsonify({"error": "Body must include a non-empty 'edits' list."}), 400
    if not isinstance(categories, list) or not categories:
        return jsonify({"error": "Body must include 'custom_categories'."}), 400

    analysis = suggest_keywords_from_edits(edits, categories)
    apply = bool(payload.get("apply", False))
    updated = categories
    if apply and analysis.get("suggestions"):
        updated = merge_keyword_suggestions(categories, analysis["suggestions"])

    return jsonify({**analysis, "custom_categories": updated})
