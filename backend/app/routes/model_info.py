# Expose available models and their descriptions for the UI
from flask import Blueprint, jsonify
from umec.models import (
    EquipmentBasedClassifier,
    SemanticSimilarityClassifier,
    TokenMatchingClassifier,
    UMECClassifier,
)

model_info_bp = Blueprint("model_info", __name__)

@model_info_bp.route("/api/models", methods=["GET"])
def list_models():
    models = [
        {
            "name": "TokenMatchingClassifier",
            "display_name": "Token Matching",
            "description": TokenMatchingClassifier.__doc__,
        },
        {
            "name": "EquipmentBasedClassifier",
            "display_name": "Equipment Based",
            "description": EquipmentBasedClassifier.__doc__,
        },
        {
            "name": "SemanticSimilarityClassifier",
            "display_name": "Semantic Similarity",
            "description": SemanticSimilarityClassifier.__doc__,
        },
        {
            "name": "UMECClassifier",
            "display_name": "UMEC Ensemble",
            "description": UMECClassifier.__doc__,
        },
    ]
    return jsonify({"models": models})
