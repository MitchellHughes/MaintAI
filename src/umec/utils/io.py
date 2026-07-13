import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_json_or_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Resource not found: {path}")

    if path.suffix.lower() in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
