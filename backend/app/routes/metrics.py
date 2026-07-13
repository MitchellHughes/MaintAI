from flask import Blueprint, jsonify
import pandas as pd
from pathlib import Path

metrics_bp = Blueprint("metrics", __name__)


@metrics_bp.route("/api/metrics")
def get_metrics():

    try:

        root = Path("/app")

        csv_file = (
            root /
            "reports" /
            "metrics" /
            "umec_classification_report.csv"
        )

        print(csv_file)

        df = pd.read_csv(csv_file)

        return jsonify(
            df.to_dict(
                orient="records"
            )
        )

    except Exception as e:

        print(e)

        return jsonify(
            {
                "error": str(e)
            }
        ), 500
