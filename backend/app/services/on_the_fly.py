"""Fit unsupervised models on the current upload at predict time (no disk required)."""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import pandas as pd

from app.services.inference import (
    MODEL_EQUIPMENT,
    MODEL_SEMANTIC,
    MODEL_TOKEN,
    MODEL_UMEC,
    _build_keyword_index,
)
from app.services.umec_training import _parse_custom_categories
from app.services.user_settings import (
    require_keyword_evidence,
    resolve_model_overrides,
    settings_fingerprint,
)
from umec.pipeline.runner import fit_on_dataframe
from umec.utils.config import load_config
from umec.utils.serialization import load_model

from app.services.model_cache import get_cache

logger = logging.getLogger(__name__)

# Only one heavy on-the-fly fit at a time (avoids two tabs each training FastText).
_fit_lock = threading.Lock()


def _get_config_dir() -> Path:
    import os

    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _dataset_fingerprint(
    df: pd.DataFrame,
    text_column: str,
    custom_categories: dict | None,
    models: list[str],
    analysis_config: dict | None = None,
) -> str:
    n = len(df)
    parts = [
        text_column,
        str(n),
        json.dumps(custom_categories or {}, sort_keys=True),
        ",".join(sorted(models)),
        settings_fingerprint(analysis_config),
    ]
    if n and text_column in df.columns:
        for idx in (0, n // 2, n - 1):
            parts.append(str(df.iloc[idx][text_column])[:256])
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:20]


def _artifacts_to_assets(artifacts: dict) -> dict[str, object]:
    token_clf = artifacts.get("token_clf")
    semantic_clf = artifacts.get("semantic_clf")
    if semantic_clf is not None:
        semantic_clf.config.show_progress = False

    keyword_index = None
    if token_clf is not None and token_clf.vectorizer is not None:
        keyword_index = _build_keyword_index(token_clf, token_clf.failure_keywords)

    return {
        "cfg": artifacts["config"],
        "data": artifacts["data"],
        "source_text_column": artifacts["source_text_column"],
        "umec": artifacts.get("umec"),
        "token_clf": token_clf,
        "equipment_clf": artifacts.get("equipment_clf"),
        "semantic_clf": semantic_clf,
        "token_map": artifacts["token_map"],
        "keyword_index": keyword_index,
        "failure_keywords": artifacts.get("failure_keywords") or {},
        "fitted_on_the_fly": True,
        "models_fitted": artifacts.get("models_fitted", []),
        "memory_tier": artifacts.get("memory_tier", "standard"),
    }


def _load_saved_assets(config_dir: Path) -> dict[str, object] | None:
    cfg = load_config(config_dir)
    model_dir = Path(cfg.project.model_dir)
    required = (
        "token_matching.joblib",
        "equipment_based.joblib",
        "semantic_similarity.joblib",
        "umec.joblib",
    )
    if not all((model_dir / name).exists() for name in required):
        return None

    token_map = None
    if cfg.data.preprocess.get("enabled", True):
        from umec.data.resources import load_token_mappings

        token_map = load_token_mappings(cfg.data.resources["token_mappings"])

    token_clf = load_model(model_dir / "token_matching.joblib")
    equipment_clf = load_model(model_dir / "equipment_based.joblib")
    semantic_clf = load_model(model_dir / "semantic_similarity.joblib")
    semantic_clf.config.show_progress = False
    umec = load_model(model_dir / "umec.joblib")

    return {
        "cfg": cfg,
        "data": None,
        "source_text_column": None,
        "umec": umec,
        "token_clf": token_clf,
        "equipment_clf": equipment_clf,
        "semantic_clf": semantic_clf,
        "token_map": token_map,
        "keyword_index": _build_keyword_index(token_clf, token_clf.failure_keywords),
        "fitted_on_the_fly": False,
        "models_fitted": [MODEL_TOKEN, MODEL_EQUIPMENT, MODEL_SEMANTIC, MODEL_UMEC],
    }


def _apply_xai_top_k(assets: dict[str, object], analysis_config: dict | None) -> None:
    raw_top_k = (analysis_config or {}).get("xai_top_k")
    if raw_top_k is None:
        user_settings = (analysis_config or {}).get("user_settings") or {}
        raw_top_k = user_settings.get("analysis.xai_top_k")
    try:
        assets["xai_top_k"] = max(1, min(10, int(raw_top_k or 3)))
    except (TypeError, ValueError):
        assets["xai_top_k"] = 3


def get_prediction_assets(
    text_column: str,
    analysis_config: dict | None = None,
    *,
    rows: list[dict] | None = None,
    df: pd.DataFrame | None = None,
    models: list[str] | None = None,
    use_saved_models: bool = False,
    fast_fit: bool = True,
) -> dict[str, object]:
    """
    Return inference assets, fitting only the classifiers needed for ``models``.

    Token-only skips FastText entirely (seconds vs many minutes).
    """
    config_dir = _get_config_dir()
    analysis_config = analysis_config or {}
    models = models or [MODEL_UMEC]

    if use_saved_models:
        saved = _load_saved_assets(config_dir)
        if saved is not None:
            logger.info("Using saved models from disk")
            return saved

    if df is None:
        if not rows:
            raise ValueError("rows or df is required for on-the-fly fit.")
        df = pd.DataFrame(rows)
    if "id" not in df.columns:
        df = df.copy()
        df["id"] = range(len(df))

    corpus = df[text_column].fillna("").astype(str).tolist() if text_column in df.columns else []
    custom_categories = _parse_custom_categories(
        {"custom_categories": analysis_config.get("custom_categories") or []},
        corpus=corpus,
    )
    if not custom_categories:
        raise ValueError(
            "Define at least one dataset category (with label) before running prediction."
        )
    overrides = resolve_model_overrides(analysis_config)

    fingerprint = _dataset_fingerprint(
        df, text_column, custom_categories, models, analysis_config
    )
    cache_key = f"otf:{fingerprint}"
    cached = get_cache().get(cache_key)
    if cached is not None:
        logger.info("Using in-memory on-the-fly models (cache hit) for %s", models)
        _apply_xai_top_k(cached, analysis_config)
        return cached

    with _fit_lock:
        cached = get_cache().get(cache_key)
        if cached is not None:
            _apply_xai_top_k(cached, analysis_config)
            return cached

        logger.info(
            "On-the-fly fit: %s rows, models=%s, fast_fit=%s",
            len(df),
            models,
            fast_fit,
        )
        artifacts = fit_on_dataframe(
            str(config_dir),
            df,
            source_text_column=text_column,
            overrides=overrides,
            custom_categories=custom_categories,
            models_needed=models,
            fast_fit=fast_fit,
            reference_label_column=analysis_config.get("label_column"),
        )
        if artifacts.get("memory_tier") == "extreme":
            requested = {str(m) for m in (models or [])}
            if MODEL_SEMANTIC in requested and MODEL_UMEC not in requested:
                logger.warning(
                    "Extreme memory tier with semantic-only run can be unstable; "
                    "consider token/equipment models or UMEC for large uploads."
                )
        assets = _artifacts_to_assets(artifacts)
        assets["require_keyword_evidence"] = require_keyword_evidence(analysis_config)
        assets["memory_tier"] = artifacts.get("memory_tier", "standard")
        _apply_xai_top_k(assets, analysis_config)
        get_cache()[cache_key] = assets
        return assets
