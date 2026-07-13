# evaluation.py — classification report for predictions with reference labels.

from __future__ import annotations

from typing import Any

from flask import Blueprint, jsonify, request

from app.services.evaluation_report import evaluation_report_from_predictions
from app.services.label_group_suggestions import label_group_preview
from umec.evaluation.label_groups import apply_label_groups_config

evaluation_bp = Blueprint("evaluation", __name__)


def _parse_report_payload(payload: dict) -> tuple[list, list, str, str, list | None, Any]:
    predictions = payload.get("predictions")
    categories = payload.get("custom_categories") or payload.get("categories") or []
    pred_key = str(payload.get("pred_key") or "final_condition").strip()
    actual_key = str(payload.get("actual_key") or "actual_label").strip()
    label_groups = payload.get("label_groups")
    top_support_n = payload.get("top_support_n")
    top_k = payload.get("top_k")
    return predictions, categories, pred_key, actual_key, label_groups, top_support_n, top_k


@evaluation_bp.route("/api/evaluation/report", methods=["POST"])
def classification_report():
    payload = request.get_json(silent=True) or {}
    predictions, categories, pred_key, actual_key, label_groups, top_support_n, top_k = _parse_report_payload(
        payload
    )

    if not isinstance(predictions, list) or not predictions:
        return jsonify({"error": "Body must include a non-empty 'predictions' list."}), 400
    if not categories:
        return jsonify({"error": "Body must include 'custom_categories'."}), 400

    try:
        top_n = int(top_support_n) if top_support_n is not None else 10
        top_n = max(1, min(top_n, 50))
    except (TypeError, ValueError):
        top_n = 10

    try:
        eval_top_k = int(top_k) if top_k is not None else 3
        eval_top_k = max(1, min(eval_top_k, 10))
    except (TypeError, ValueError):
        eval_top_k = 3

    try:
        report = evaluation_report_from_predictions(
            predictions,
            categories,
            pred_key=pred_key,
            actual_key=actual_key,
            label_groups=label_groups,
            top_support_n=top_n,
            top_k=eval_top_k,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if report.get("error"):
        return jsonify(report), 422

    return jsonify(report)


@evaluation_bp.route("/api/evaluation/suggest-label-groups", methods=["POST"])
def suggest_label_groups_route():
    payload = request.get_json(silent=True) or {}
    predictions, categories, pred_key, actual_key, label_groups, _top_n, _top_k = _parse_report_payload(
        payload
    )

    if not isinstance(predictions, list) or not predictions:
        return jsonify({"error": "Body must include a non-empty 'predictions' list."}), 400
    if not categories:
        return jsonify({"error": "Body must include 'custom_categories'."}), 400

    try:
        preview = label_group_preview(
            predictions,
            categories,
            label_groups,
            pred_key=pred_key,
            actual_key=actual_key,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if preview.get("baseline_report", {}).get("error"):
        return jsonify(preview["baseline_report"]), 422

    return jsonify(
        {
            "baseline_macro_f1": preview.get("baseline_macro_f1"),
            "merged_macro_f1": preview.get("merged_macro_f1"),
            "suggested_macro_f1": preview.get("suggested_macro_f1"),
            "baseline_classes": preview.get("baseline_classes"),
            "merged_classes": preview.get("merged_classes"),
            "suggested_classes": preview.get("suggested_classes"),
            "suggested_groups": preview.get("suggested_groups") or [],
            "available_labels": preview.get("available_labels") or [],
        }
    )


@evaluation_bp.route("/api/evaluation/apply-label-groups", methods=["POST"])
def apply_label_groups_route():
    payload = request.get_json(silent=True) or {}
    categories = payload.get("custom_categories") or payload.get("categories") or []
    label_groups = payload.get("label_groups") or []
    merge_categories = bool(payload.get("merge_categories"))

    if not categories:
        return jsonify({"error": "Body must include 'custom_categories'."}), 400
    if not isinstance(label_groups, list):
        return jsonify({"error": "'label_groups' must be a list."}), 400

    try:
        updated = apply_label_groups_config(
            categories,
            label_groups,
            merge_categories=merge_categories,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify(updated)
