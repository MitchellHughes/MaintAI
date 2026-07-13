# history.py - lists / fetches archived prediction records.

from flask import Blueprint, jsonify, request

from app.services.umec_storage import get_history_item, list_history, save_run_snapshot

history_bp = Blueprint("history", __name__)


@history_bp.route("/api/history", methods=["GET"])
def history():
    return jsonify({"items": list_history()})


@history_bp.route("/api/history/<record_id>", methods=["GET"])
def history_item(record_id):
    item = get_history_item(record_id)
    if item is None:
        return jsonify({"error": "Record not found."}), 404
    return jsonify(item)


@history_bp.route("/api/history/snapshot", methods=["POST"])
def save_snapshot():
    payload = request.get_json(silent=True) or {}
    snapshot = payload.get("snapshot")
    if not isinstance(snapshot, dict) or not snapshot.get("prediction"):
        return jsonify({"error": "Body must include snapshot.prediction."}), 400
    saved = save_run_snapshot(snapshot, user=payload.get("user", "anonymous"))
    return jsonify(saved), 201
