"""Schema-driven user settings merged into pipeline config overrides."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

from umec.utils.config import load_config

PREDICTION_MODE_ID = "umec.prediction_mode"
REQUIRE_KEYWORD_EVIDENCE_ID = "analysis.require_keyword_evidence"
VIRTUAL_SETTING_IDS = {PREDICTION_MODE_ID, REQUIRE_KEYWORD_EVIDENCE_ID}


def _config_dir() -> Path:
    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _schema_path() -> Path:
    return _config_dir() / "ui_settings.yaml"


def _load_schema_file() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _schema_path()
    if not path.exists():
        return [], {}
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    settings = data.get("settings") or []
    groups = data.get("groups") or {}
    return (
        [s for s in settings if isinstance(s, dict) and s.get("id")],
        groups if isinstance(groups, dict) else {},
    )


def _model_section(cfg, model_key: str) -> dict[str, Any]:
    return getattr(cfg.models, model_key, None) or {}


def _get_by_path(obj: dict[str, Any], path: str) -> Any:
    cur: Any = obj
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(path)
        cur = cur[part]
    return cur


def _set_by_path(target: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    cur = target
    for part in parts[:-1]:
        nxt = cur.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[part] = nxt
        cur = nxt
    cur[parts[-1]] = value


def default_for_setting(cfg, setting_id: str) -> Any:
    if setting_id == REQUIRE_KEYWORD_EVIDENCE_ID:
        return True
    if setting_id == PREDICTION_MODE_ID:
        allow = _get_by_path(_model_section(cfg, "umec"), "decode.allow_unclassified")
        return "allow_unclassified" if allow else "predict_all"
    model_key, subpath = setting_id.split(".", 1)
    section = _model_section(cfg, model_key)
    return copy.deepcopy(_get_by_path(section, subpath))


def get_settings_catalog() -> dict[str, Any]:
    cfg = load_config(_config_dir())
    schema, groups = _load_schema_file()
    defaults: dict[str, Any] = {}
    for entry in schema:
        sid = entry["id"]
        try:
            defaults[sid] = default_for_setting(cfg, sid)
        except KeyError:
            defaults[sid] = None
    return {"settings": schema, "defaults": defaults, "groups": groups}


def _normalize_user_settings(user_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Map virtual UI settings onto model.yaml override paths."""
    if not user_settings:
        return {}
    normalized = dict(user_settings)
    mode = normalized.get(PREDICTION_MODE_ID)
    if mode is not None:
        normalized["umec.decode.allow_unclassified"] = mode == "allow_unclassified"
    return normalized


def _deep_update(base: dict, updates: dict | None) -> dict:
    if not updates:
        return base
    out = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def user_settings_to_models(user_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Convert flat setting ids to the nested models.* override structure."""
    if not user_settings:
        return {}
    models: dict[str, Any] = {}
    for setting_id, value in user_settings.items():
        if value is None or setting_id in VIRTUAL_SETTING_IDS:
            continue
        model_key, subpath = setting_id.split(".", 1)
        patch: dict[str, Any] = {}
        _set_by_path(patch, subpath, value)
        models[model_key] = _deep_update(models.get(model_key, {}), patch)
    return models


def resolve_model_overrides(
    analysis_config: dict | None,
) -> dict | None:
    """
    Merge legacy model_overrides with schema-based user_settings.

    Returns the normalized shape expected by the pipeline: {"models": {...}, "data": {...}}.
    """
    analysis_config = analysis_config or {}
    raw = analysis_config.get("model_overrides")
    if raw and ("models" in raw or "data" in raw):
        base = copy.deepcopy(raw)
    elif raw:
        base = {"models": copy.deepcopy(raw)}
    else:
        base = {}

    from_models = user_settings_to_models(
        _normalize_user_settings(analysis_config.get("user_settings"))
    )
    if from_models:
        base.setdefault("models", {})
        for model_key, patch in from_models.items():
            base["models"][model_key] = _deep_update(
                base["models"].get(model_key, {}),
                patch,
            )

    if not base:
        return None
    return base if ("models" in base or "data" in base) else {"models": base}


def require_keyword_evidence(analysis_config: dict | None) -> bool:
    if not analysis_config:
        return False
    if "require_keyword_evidence" in analysis_config:
        return bool(analysis_config["require_keyword_evidence"])
    user_settings = analysis_config.get("user_settings") or {}
    value = user_settings.get(REQUIRE_KEYWORD_EVIDENCE_ID)
    return True if value is None else bool(value)


def settings_fingerprint(analysis_config: dict | None) -> str:
    import json

    resolved = resolve_model_overrides(analysis_config) or {}
    return json.dumps(resolved, sort_keys=True, default=str)
