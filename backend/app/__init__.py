import sys
from pathlib import Path

from flask import Flask
from flask_cors import CORS

# Make the ML package importable from src/ without requiring an editable
# install (keeps `python backend/run.py` working out of the box).
_SRC = Path(__file__).resolve().parents[2] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def create_app():
    app = Flask(__name__)

    # No cap on upload / request body size. CSV and Excel uploads (and the
    # /predict JSON payload) can be large; None disables the limits. The
    # MAX_FORM_* keys exist on Flask >= 3.1 and are ignored on older versions.
    app.config["MAX_CONTENT_LENGTH"] = None
    app.config["MAX_FORM_MEMORY_SIZE"] = None
    app.config["MAX_FORM_PARTS"] = None

    CORS(
        app,
        supports_credentials=True,
        resources={
            r"/api/*": {
                "origins": [
                    "http://localhost:5173",
                    "http://127.0.0.1:5173",
                ]
            }
        },
    )

    from app.routes.root import root_bp
    from app.routes.health import health_bp
    from app.routes.upload import upload_bp
    from app.routes.predict import predict_bp
    from app.routes.train import train_bp
    from app.routes.feedback import feedback_bp
    from app.routes.history import history_bp
    from app.routes.metrics import metrics_bp
    from app.routes.model_info import model_info_bp
    from app.routes.tokens import tokens_bp
    from app.routes.settings import settings_bp
    from app.routes.export_data import export_bp
    from app.routes.evaluation import evaluation_bp

    from app.services.umec_storage import warm_history_cache

    warm_history_cache()

    for bp in (
        root_bp,
        health_bp,
        upload_bp,
        predict_bp,
        train_bp,
        feedback_bp,
        history_bp,
        metrics_bp,
        model_info_bp,
        tokens_bp,
        settings_bp,
        export_bp,
        evaluation_bp,
    ):
        app.register_blueprint(bp)

    return app

