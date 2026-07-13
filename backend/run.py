# Entry point for the Flask app.

import os

from app import create_app

app = create_app()

if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "1") == "1"
    use_reloader = os.getenv("FLASK_USE_RELOADER", "0") == "1"
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=debug,
        use_reloader=use_reloader,
        threaded=True,
    )
