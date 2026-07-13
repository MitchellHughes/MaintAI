# health.py monitors the health/status of the backend server. 
# It provides a simple endpoint that can be used to check if the server is running and responsive.

from flask import Blueprint, jsonify

health_bp = Blueprint("health", __name__)

@health_bp.route("/api/health")
def health():
    return jsonify({
        "status": "Backend running"
    })