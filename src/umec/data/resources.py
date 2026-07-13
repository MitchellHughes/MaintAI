from __future__ import annotations

from typing import Dict

from umec.utils.io import load_json_or_yaml


def load_failure_keywords(path: str) -> Dict[str, list[str]]:
    data = load_json_or_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Failure keywords must be a dict of class -> keywords.")

    return {str(k): list(v) for k, v in data.items()}


def load_token_mappings(path: str) -> Dict[str, str]:
    data = load_json_or_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Token mappings must be a dict of token -> normalized token.")
    return {str(k): str(v) for k, v in data.items()}


def load_label_mappings(path: str) -> Dict[str, str]:
    data = load_json_or_yaml(path)
    if not isinstance(data, dict):
        raise ValueError("Label mappings must be a dict of raw label -> normalized label.")
    return {str(k): str(v) for k, v in data.items()}
