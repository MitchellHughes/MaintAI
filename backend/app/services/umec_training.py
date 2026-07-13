"""Training wrapper that calls the UMEC pipeline."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from umec.data.keyword_generation import resolve_all_category_keywords
from umec.data.resources import load_label_mappings, load_token_mappings
from umec.pipeline.runner import run_evaluate, run_train, run_train_from_dataframe
from umec.utils.config import load_config

from app.services.user_settings import resolve_model_overrides


def _get_config_dir() -> Path:
    return Path(os.getenv("UMEC_CONFIG_DIR", "configs/core"))


def _model_version() -> str:
    try:
        cfg = load_config(_get_config_dir())
        return f"umec-{Path(cfg.project.model_dir).name}"
    except Exception:
        return "umec-unknown"


def _normalize_overrides(raw: dict | None) -> dict | None:
    if not raw:
        return None
    if "models" in raw or "data" in raw:
        return raw
    return {"models": raw}


def _parse_custom_categories(
    dataset_meta: dict | None,
    corpus: list[str] | None = None,
    *,
    df: pd.DataFrame | None = None,
) -> dict[str, list[str]] | None:
    if not isinstance(dataset_meta, dict):
        return None
    raw_categories = dataset_meta.get("custom_categories") or []
    if not isinstance(raw_categories, list) or not raw_categories:
        return None

    category_rows = []
    for item in raw_categories:
        label = str(item.get("label", "")).strip()
        if not label:
            continue
        category_rows.append({"label": label, "keywords": item.get("keywords") or ""})

    if not category_rows:
        return None

    label_mappings: dict[str, str] = {}
    token_map: dict[str, str] = {}
    try:
        label_mappings = load_label_mappings("configs/mappings/label_mappings.json")
    except Exception:
        pass
    try:
        token_map = load_token_mappings("configs/mappings/token_mappings.json")
    except Exception:
        pass

    text_column = dataset_meta.get("text_column")
    label_column = dataset_meta.get("label_column")

    resolved = resolve_all_category_keywords(
        category_rows,
        corpus=corpus,
        df=df,
        text_column=text_column if df is not None else None,
        label_column=label_column,
        label_mappings=label_mappings,
        token_map=token_map,
    )
    return resolved or None


def run_training(dataset_meta: dict | None = None, feedback: list | None = None) -> dict:
    config_dir = _get_config_dir()
    meta = dataset_meta if isinstance(dataset_meta, dict) else {}
    overrides = resolve_model_overrides(
        {
            "model_overrides": meta.get("model_overrides"),
            "user_settings": meta.get("user_settings"),
        }
    )

    rows = (dataset_meta or {}).get("rows") if isinstance(dataset_meta, dict) else None
    text_column = (dataset_meta or {}).get("text_column") if isinstance(dataset_meta, dict) else None

    trained_on_upload = isinstance(rows, list) and bool(rows) and bool(text_column)
    if trained_on_upload:
        df = pd.DataFrame(rows)
        corpus = df[text_column].fillna("").astype(str).tolist() if text_column in df.columns else []
        custom_categories = _parse_custom_categories(dataset_meta, corpus=corpus, df=df)
        run_train_from_dataframe(
            str(config_dir),
            df=df,
            source_text_column=str(text_column),
            overrides=overrides,
            custom_categories=custom_categories,
        )
    else:
        custom_categories = _parse_custom_categories(dataset_meta)
        run_train(
            str(config_dir),
            overrides=overrides,
            custom_categories=custom_categories,
        )

    metrics = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
    }
    report = {}

    if not trained_on_upload:
        try:
            custom_categories = _parse_custom_categories(dataset_meta)
            eval_out = run_evaluate(
                str(config_dir),
                overrides=overrides,
                custom_categories=custom_categories,
            )
            macro = eval_out.get("macro_f1")
            if macro is not None:
                metrics = {
                    "accuracy": macro,
                    "precision": macro,
                    "recall": macro,
                    "f1": macro,
                }
            report = {"report_path": eval_out.get("report_path")}
            report["top2_accuracy"] = eval_out.get("top2_accuracy")
        except Exception:
            pass

    return {
        "model_version": _model_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "num_feedback_records": len(feedback or []),
        "dataset_meta": dataset_meta or {},
        "metrics": metrics,
        "report": report,
    }
