"""Expose user-tunable model settings (schema + defaults from config)."""

from flask import Blueprint, jsonify

from app.services.user_settings import get_settings_catalog

settings_bp = Blueprint("settings", __name__, url_prefix="/api")


@settings_bp.get("/settings")
def get_analysis_settings():
    return jsonify(get_settings_catalog())
