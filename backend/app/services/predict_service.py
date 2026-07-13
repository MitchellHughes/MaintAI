"""Core predict pipeline (sync + background jobs)."""

from __future__ import annotations

import gc
import gzip
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from app.services.dataset_store import dataframe_for_predict, load_meta
from app.services.inference import (
    MODEL_EQUIPMENT,
    MODEL_SEMANTIC,
    MODEL_TOKEN,
    MODEL_UMEC,
    predict_with_model,
)
from app.services.on_the_fly import get_prediction_assets
from app.services.progress import emit_progress, format_progress_message

logger = logging.getLogger(__name__)

MODEL_DISPLAY = {
    MODEL_TOKEN: "Token matching",
    MODEL_EQUIPMENT: "Equipment based",
    MODEL_SEMANTIC: "Semantic similarity",
    MODEL_UMEC: "UMEC ensemble",
}

SLIM_ROW_THRESHOLD = 10_000
ULTRA_SLIM_ROW_THRESHOLD = 50_000


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    rem = seconds - minutes * 60
    return f"{minutes}m {rem:.0f}s"


def slim_prediction_row(row: dict) -> dict:
    xai = row.get("xai") or {}
    simple = xai.get("simple") or {}
    slim_simple = {
        "tier": simple.get("tier"),
        "tier_label": simple.get("tier_label"),
        "confidence": simple.get("confidence"),
        "one_liner": simple.get("one_liner"),
        "keywords": simple.get("keywords"),
        "runner_up": simple.get("runner_up"),
        "top_ranked": simple.get("top_ranked"),
        "top_k_details": simple.get("top_k_details"),
        "text_spans": simple.get("text_spans"),
        "models_agree": simple.get("models_agree"),
        "models_total": simple.get("models_total"),
    }
    return {
        "row_id": row.get("row_id"),
        "discrepancy": row.get("discrepancy"),
        "component": row.get("component"),
        "predicted_condition": row.get("predicted_condition"),
        "top_predictions": row.get("top_predictions"),
        "confidence": row.get("confidence"),
        "confidence_tier": row.get("confidence_tier"),
        "runner_up": row.get("runner_up"),
        "models_agree": row.get("models_agree"),
        "models_total": row.get("models_total"),
        "actual_label": row.get("actual_label"),
        "model": row.get("model"),
        "xai": {
            "simple": slim_simple,
            "explanation": simple.get("one_liner") or xai.get("explanation"),
        },
    }


def _attach_actual_labels(
    results_by_model: dict[str, list],
    *,
    label_column: str | None,
    label_source: pd.DataFrame | None,
    custom_categories: list | None = None,
    label_groups: list | None = None,
    xai_top_k: int = 3,
) -> None:
    if not label_column or label_source is None or label_column not in label_source.columns:
        return
    from umec.evaluation.category_matching import build_category_matcher
    from umec.explainability.text_highlights import (
        build_text_highlights,
        find_non_overlapping_spans,
        surface_terms_in_text,
    )

    match = build_category_matcher(custom_categories or [], label_groups=label_groups)
    labels = label_source[label_column].astype(str).tolist()
    cap = max(1, min(int(xai_top_k or 3), 10))

    for model_rows in results_by_model.values():
        for pred in model_rows:
            row_id = pred.get("row_id")
            try:
                idx = int(row_id)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(labels):
                actual = labels[idx].strip()
                pred["actual_label"] = actual
                text = str(pred.get("discrepancy") or "")
                xai = pred.get("xai") or {}
                simple = xai.setdefault("simple", {})
                predicted = str(pred.get("predicted_condition") or "").strip()

                ranked = list(pred.get("top_predictions") or [])
                ranked_keys = {
                    str(item.get("label", "")).strip().lower()
                    for item in ranked
                    if str(item.get("label", "")).strip()
                }
                mapped_actual = match(actual) if actual else None
                predicted_key = predicted.strip().lower()

                def _sync_top_k(ranked_list: list[dict]) -> None:
                    trimmed = ranked_list[:cap]
                    pred["top_predictions"] = trimmed
                    simple["top_ranked"] = [
                        {
                            "label": str(item["label"]),
                            "confidence": float(item.get("confidence") or 0.0),
                        }
                        for item in trimmed
                    ]
                    if simple.get("top_k_details"):
                        simple["top_k_details"] = [
                            {
                                **detail,
                                "rank": index + 1,
                            }
                            for index, detail in enumerate(simple["top_k_details"][:cap])
                        ]

                if mapped_actual and mapped_actual not in ranked_keys:
                    in_text = bool(surface_terms_in_text(text, [mapped_actual, actual]))
                    if in_text or predicted_key != mapped_actual:
                        ranked.append(
                            {
                                "label": mapped_actual,
                                "confidence": 0.0,
                                "evidence_backed": in_text,
                                "reference_only": not in_text,
                            }
                        )
                        _sync_top_k(ranked)
                elif actual and not mapped_actual and surface_terms_in_text(text, [actual]):
                    actual_key = actual.strip().lower()
                    if actual_key not in ranked_keys:
                        ranked.append(
                            {
                                "label": actual,
                                "confidence": 0.0,
                                "evidence_backed": False,
                                "reference_only": True,
                            }
                        )
                        _sync_top_k(ranked)

                highlights = build_text_highlights(
                    text,
                    predicted_label=predicted,
                    top_predictions=pred.get("top_predictions") or [],
                    failure_keywords=None,
                    actual_label=actual,
                )
                simple["other_terms"] = highlights.get("other_terms") or []
                existing = list(simple.get("text_spans") or [])
                for span in highlights.get("spans") or []:
                    if span.get("role") != "other":
                        continue
                    key = (span["start"], span["end"])
                    if key not in {(s["start"], s["end"]) for s in existing}:
                        existing.append(span)
                if existing:
                    simple["text_spans"] = sorted(existing, key=lambda s: s["start"])


def execute_predict(
    *,
    rows: list[dict] | None = None,
    text_column: str,
    models: list[str],
    analysis_config: dict,
    label_column: str | None = None,
    upload_id: str | None = None,
    source_filename: str | None = None,
    part_column: str | None = None,
    use_saved_models: bool = False,
    on_progress: Callable[[str | dict], None] | None = None,
    spool_to_disk: bool = False,
) -> dict[str, Any]:
    phase_seconds: dict[str, float] = {}
    t_load_start = time.perf_counter()
    predict_df: pd.DataFrame | None = None
    if upload_id:
        predict_df = dataframe_for_predict(
            str(upload_id),
            text_column=text_column,
            part_column=part_column,
        )
        row_count = len(predict_df)
    else:
        if not rows:
            raise ValueError("Either upload_id or rows is required.")
        row_count = len(rows)
        predict_df = pd.DataFrame(rows)
        if "id" not in predict_df.columns:
            predict_df["id"] = range(len(predict_df))
        if part_column and part_column in predict_df.columns:
            merged = []
            for _, row in predict_df.iterrows():
                text = str(row.get(text_column) or "").strip()
                part = str(row.get(part_column) or "").strip()
                if part:
                    part_lower = part.lower()
                    text_lower = text.lower()
                    if not text_lower or part_lower not in text_lower:
                        text = f"{part} — {text}".strip() if text else part
                merged.append(text)
            predict_df[text_column] = merged

    phase_seconds["data_prepare_seconds"] = round(time.perf_counter() - t_load_start, 2)
    resolved_filename = str(source_filename or "").strip()
    if upload_id and not resolved_filename:
        meta = load_meta(str(upload_id))
        if meta and meta.get("filename"):
            resolved_filename = str(meta["filename"]).strip()
    slim_response = row_count > SLIM_ROW_THRESHOLD

    def progress(update: str | dict) -> None:
        if isinstance(update, dict):
            msg = update.get("message") or format_progress_message(update)
            logger.info(msg)
            emit_progress(on_progress, {**update, "message": msg})
        else:
            logger.info(update)
            if on_progress:
                on_progress(update)

    t0 = time.perf_counter()
    progress(f"Training models on {row_count:,} rows…")

    t_fit = time.perf_counter()
    assets = get_prediction_assets(
        text_column,
        analysis_config=analysis_config,
        df=predict_df,
        models=models,
        use_saved_models=use_saved_models,
        fast_fit=True,
    )
    if row_count > ULTRA_SLIM_ROW_THRESHOLD:
        assets["slim_rows"] = True
    assets["part_column"] = (
        part_column
        if part_column and predict_df is not None and part_column in predict_df.columns
        else None
    )
    assets["boost_equipment"] = bool(assets.get("part_column"))
    fit_seconds = time.perf_counter() - t_fit
    phase_seconds["fit_seconds"] = round(fit_seconds, 2)
    gc.collect()
    progress(f"Training finished in {format_seconds(fit_seconds)}")

    cfg = assets["cfg"]
    df = assets["data"]
    if df is None:
        raise ValueError(
            "Saved models cannot be used without on-the-fly fit for this upload. "
            "Omit use_saved_models or run predict with your uploaded rows."
        )

    processed_column = cfg.data.text_column
    source_text_column = str(assets.get("source_text_column") or text_column)
    memory_tier = str(assets.get("memory_tier") or "standard")

    results_by_model: dict[str, list] = {}
    spool_dir = Path("/app/unclassified/predict_jobs_tmp")
    if spool_to_disk:
        spool_dir.mkdir(parents=True, exist_ok=True)
    spool_paths: dict[str, str] = {}
    model_seconds: dict[str, float] = {}
    t_predict = time.perf_counter()

    for model_name in models:
        label = MODEL_DISPLAY.get(model_name, model_name)
        progress(f"Scoring with {label} ({row_count:,} rows)…")
        t_model = time.perf_counter()
        if spool_to_disk:
            out_path = spool_dir / f"{int(time.time() * 1000)}_{model_name}.json.gz"
            with gzip.open(out_path, "wt", encoding="utf-8") as handle:
                handle.write("[")
                first = True

                def _write_chunk(chunk_rows: list[dict[str, Any]]) -> None:
                    nonlocal first
                    rows_to_write = (
                        [slim_prediction_row(row) for row in chunk_rows]
                        if slim_response and not bool(assets.get("slim_rows", False))
                        else chunk_rows
                    )
                    for row in rows_to_write:
                        if not first:
                            handle.write(",")
                        json.dump(row, handle)
                        first = False

                predict_with_model(
                    model_name,
                    df,
                    processed_column,
                    source_text_column,
                    assets,
                    on_progress=on_progress,
                    on_chunk_rows=_write_chunk,
                )
                handle.write("]")

            spool_paths[model_name] = str(out_path)
            results_by_model[model_name] = []
            gc.collect()
        else:
            model_rows = predict_with_model(
                model_name,
                df,
                processed_column,
                source_text_column,
                assets,
                on_progress=on_progress,
            )
            if slim_response and not bool(assets.get("slim_rows", False)):
                model_rows = [slim_prediction_row(row) for row in model_rows]
            results_by_model[model_name] = model_rows
        elapsed = time.perf_counter() - t_model
        model_seconds[model_name] = round(elapsed, 2)
        progress(f"{label} done in {format_seconds(elapsed)}")
        gc.collect()

    predict_seconds = time.perf_counter() - t_predict
    phase_seconds["score_total_seconds"] = round(predict_seconds, 2)
    total_seconds = time.perf_counter() - t0

    t_reload = time.perf_counter()
    if spool_to_disk:
        loaded: dict[str, list] = {}
        for name, path in spool_paths.items():
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                loaded[name] = json.load(handle)
        results_by_model = loaded
    phase_seconds["spool_reload_seconds"] = round(time.perf_counter() - t_reload, 2)

    categories = (analysis_config or {}).get("custom_categories") or []
    eval_top_k = int((analysis_config or {}).get("xai_top_k") or 3)
    groups = (analysis_config or {}).get("label_groups") or []
    t_attach = time.perf_counter()
    _attach_actual_labels(
        results_by_model,
        label_column=label_column,
        label_source=predict_df if label_column else None,
        custom_categories=categories,
        label_groups=groups,
        xai_top_k=eval_top_k,
    )
    phase_seconds["attach_labels_seconds"] = round(time.perf_counter() - t_attach, 2)

    t_timing = time.perf_counter()
    timing = {
        "row_count": row_count,
        "fit_seconds": round(fit_seconds, 2),
        "predict_seconds": round(predict_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        "fit_display": format_seconds(fit_seconds),
        "predict_display": format_seconds(predict_seconds),
        "total_display": format_seconds(total_seconds),
        "memory_tier": memory_tier,
        "large_dataset_mode": memory_tier != "standard",
        "models": {
            name: {
                "seconds": model_seconds.get(name, 0),
                "display": format_seconds(model_seconds.get(name, 0)),
                "label": MODEL_DISPLAY.get(name, name),
            }
            for name in models
        },
        "phase_seconds": phase_seconds,
    }
    timing["phase_seconds"]["timing_build_seconds"] = round(time.perf_counter() - t_timing, 2)
    timing["phase_seconds"]["total_seconds"] = round(total_seconds, 2)

    return {
        "model_version": f"umec-{Path(cfg.project.model_dir).name}",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "text_column": text_column,
        **({"label_column": label_column} if label_column else {}),
        **({"upload_id": upload_id} if upload_id else {}),
        **({"source_filename": resolved_filename} if resolved_filename else {}),
        "models": models,
        "fitted_on_the_fly": bool(assets.get("fitted_on_the_fly")),
        "slim_response": slim_response,
        "timing": timing,
        "results_by_model": results_by_model,
    }
