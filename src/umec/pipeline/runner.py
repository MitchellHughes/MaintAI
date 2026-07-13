from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Tuple

from umec.models.equipment_based import EquipmentBasedClassifier, EquipmentBasedConfig

import pandas as pd

from umec.data.io import read_data, save_data
from umec.data.keyword_generation import generate_keywords_for_label, resolve_all_category_keywords
from umec.data.preprocessing import preprocess_dataframe, apply_label_map
from umec.data.resources import load_failure_keywords, load_label_mappings, load_token_mappings
from umec.prediction.keywords import find_overlapping_keywords
from umec.data.validation import validate_columns
from umec.evaluation.metrics import classification_report_df, macro_f1, top_k_accuracy
from umec.evaluation.plots import plot_confusion_matrix
from umec.models.semantic_similarity import SemanticSimilarityClassifier, SemanticSimilarityConfig
from umec.models.token_matching import TokenMatchingClassifier, TokenMatchingConfig
from umec.models.umec import UMECClassifier, UMECConfig
from umec.utils.config import load_config
from umec.utils.logging import get_logger
from umec.utils.paths import ensure_dir
from umec.utils.serialization import save_model
from umec.utils.seed import set_seed

TOKEN_MODEL = "TokenMatchingClassifier"
EQUIPMENT_MODEL = "EquipmentBasedClassifier"
SEMANTIC_MODEL = "SemanticSimilarityClassifier"
UMEC_MODEL = "UMECClassifier"
ALL_MODELS = (TOKEN_MODEL, EQUIPMENT_MODEL, SEMANTIC_MODEL, UMEC_MODEL)


def _deep_update(base: dict, updates: dict | None) -> dict:
    if not updates:
        return base
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _merge_keywords(default_keywords: dict[str, list[str]], custom_categories: dict | None) -> dict[str, list[str]]:
    merged = {str(label): list(words) for label, words in default_keywords.items()}
    if not custom_categories:
        return merged

    for label, words in custom_categories.items():
        if label not in merged:
            merged[str(label)] = []
        existing = set(merged[str(label)])
        for word in words or []:
            word = str(word).strip()
            if word and word not in existing:
                merged[str(label)].append(word)
                existing.add(word)
    return merged


def _categories_from_label_column(
    df: pd.DataFrame,
    cfg,
    corpus: list[str] | None,
) -> dict[str, list[str]] | None:
    col = cfg.data.label_column
    if not col or col not in df.columns:
        return None
    out: dict[str, list[str]] = {}
    for raw in df[col].dropna().unique():
        clean = str(raw).strip().lower()
        if not clean:
            continue
        out[clean] = generate_keywords_for_label(clean, corpus=corpus, dataset_only=True)
    return out or None


def _reference_label_column(
    cfg,
    df: pd.DataFrame | None,
    reference_label_column: str | None = None,
) -> str | None:
    if df is None:
        return None
    candidates = [
        reference_label_column,
        getattr(cfg.data, "source_label_column", None),
        getattr(cfg.data, "label_column", None),
    ]
    for col in candidates:
        if col and col in df.columns:
            return col
    return None


def _resolve_failure_keywords(
    cfg,
    custom_categories: dict | None,
    corpus: list[str] | None = None,
    df: pd.DataFrame | None = None,
    *,
    reference_label_column: str | None = None,
) -> dict[str, list[str]]:
    """Keywords from explicit UI entry or mined from this upload only (no global lists)."""
    if not custom_categories and df is not None:
        custom_categories = _categories_from_label_column(df, cfg, corpus)

    if not custom_categories:
        raise ValueError(
            "Define at least one category for this dataset before running prediction or training."
        )

    text_col = cfg.data.source_text_column or cfg.data.text_column
    label_col = _reference_label_column(cfg, df, reference_label_column)
    category_rows = [{"label": k, "keywords": v} for k, v in custom_categories.items()]

    label_mappings: dict[str, str] = {}
    try:
        label_mappings = load_label_mappings(cfg.data.resources["label_mappings"])
    except Exception:
        pass

    token_map: dict[str, str] = {}
    try:
        token_map = load_token_mappings(cfg.data.resources["token_mappings"])
    except Exception:
        pass

    resolved = resolve_all_category_keywords(
        category_rows,
        corpus=corpus,
        df=df,
        text_column=text_col if df is not None else None,
        label_column=label_col,
        label_mappings=label_mappings,
        token_map=token_map,
    )
    if not resolved:
        raise ValueError("At least one category label is required.")

    overlaps = find_overlapping_keywords(resolved)
    if overlaps:
        sample = ", ".join(f"'{t}'→{labels}" for t, labels in list(overlaps.items())[:5])
        get_logger("pipeline").warning(
            "Keywords shared across categories (%d tokens); evidence may be ambiguous. Examples: %s",
            len(overlaps),
            sample,
        )
    return resolved


def _umec_config_from_cfg(cfg) -> UMECConfig:
    umec_params = dict(cfg.models.umec)
    spectral = umec_params.get("spectral", {})
    return UMECConfig(
        ecoc_scheme=umec_params.get("ecoc", {}).get("scheme", "pairwise"),
        aggregation=umec_params.get("aggregation", "spectral"),
        prior_weight=umec_params.get("decode", {}).get("prior_weight", 0.35),
        allow_unclassified=umec_params.get("decode", {}).get("allow_unclassified", True),
        unclassified_threshold=umec_params.get("decode", {}).get("unclassified_threshold", -4.5),
        use_spectral=spectral.get("enabled", True),
        spectral_components=spectral.get("components", 3),
        use_third_moment=spectral.get("use_third_moment", True),
        fit_sample_rows=int(umec_params.get("fit_sample_rows") or 0),
        predict_chunk_size=int(umec_params.get("predict_chunk_size") or 5000),
    )


def _init_classifiers(
    cfg,
    failure_keywords: dict[str, list[str]] | None = None,
    token_map: dict[str, str] | None = None,
) -> Tuple[TokenMatchingClassifier, EquipmentBasedClassifier, SemanticSimilarityClassifier]:
    token_params = dict(cfg.models.token_matching)
    if "ngram_range" in token_params and isinstance(token_params["ngram_range"], list):
        token_params["ngram_range"] = tuple(token_params["ngram_range"])
    token_cfg = TokenMatchingConfig(**token_params)

    equip_params = dict(getattr(cfg.models, "equipment_based", {}) or {})
    if "ngram_range" in equip_params and isinstance(equip_params["ngram_range"], list):
        equip_params["ngram_range"] = tuple(equip_params["ngram_range"])
    equip_cfg = EquipmentBasedConfig(**equip_params) if equip_params else EquipmentBasedConfig()

    sem_cfg = SemanticSimilarityConfig(**cfg.models.semantic_similarity)

    failure_keywords = failure_keywords or load_failure_keywords(cfg.data.resources["failure_keywords"])
    token_map = token_map or load_token_mappings(cfg.data.resources["token_mappings"])

    token_clf = TokenMatchingClassifier(
        failure_keywords=failure_keywords,
        token_map=token_map,
        config=token_cfg,
    )

    equipment_clf = EquipmentBasedClassifier.from_resource_files(
        failure_keywords=failure_keywords,
        part_keywords_path=cfg.data.resources["part_keywords"],
        prominence_path=cfg.data.resources["part_failure_prominence"],
        token_map=token_map,
        config=equip_cfg,
    )

    semantic_clf = SemanticSimilarityClassifier(
        failure_keywords=failure_keywords,
        config=sem_cfg,
    )

    return token_clf, equipment_clf, semantic_clf


def _fit_base_classifiers(df, cfg, token_clf, equipment_clf, semantic_clf) -> None:
    corpus = df[cfg.data.text_column].fillna("").astype(str)
    logger = get_logger("pipeline", cfg.project.log_level)
    logger.info("Fitting token matching classifier")
    token_clf.fit(corpus)
    logger.info("Fitting equipment-based classifier")
    equipment_clf.fit(corpus)
    logger.info("Fitting semantic similarity classifier")
    semantic_clf.fit(corpus)


def _build_umec(cfg, token_clf, equipment_clf, semantic_clf) -> UMECClassifier:
    return UMECClassifier(
        classifiers=[token_clf, equipment_clf, semantic_clf],
        config=_umec_config_from_cfg(cfg),
    )


def _save_models(cfg, token_clf, equipment_clf, semantic_clf, umec) -> Path:
    model_dir = ensure_dir(cfg.project.model_dir)
    save_model(token_clf, Path(model_dir) / "token_matching.joblib")
    save_model(equipment_clf, Path(model_dir) / "equipment_based.joblib")
    save_model(semantic_clf, Path(model_dir) / "semantic_similarity.joblib")
    save_model(umec, Path(model_dir) / "umec.joblib")
    return model_dir


def _load_and_preprocess(cfg, token_map=None) -> pd.DataFrame:
    read_kwargs = cfg.data.read_kwargs or {}
    df = read_data(cfg.data.path, file_format=cfg.data.format, **read_kwargs)

    source_text = cfg.data.source_text_column
    if source_text is None and cfg.data.text_column.startswith("processed_"):
        source_text = cfg.data.text_column.replace("processed_", "", 1)
    source_text = source_text or cfg.data.text_column

    source_label = cfg.data.source_label_column
    if source_label is None and cfg.data.label_column.startswith("processed_"):
        source_label = cfg.data.label_column.replace("processed_", "", 1)

    source_required = [source_text]
    if source_label:
        source_required.append(source_label)
    validate_columns(df, source_required)

    token_map = token_map or load_token_mappings(cfg.data.resources["token_mappings"])
    label_map = load_label_mappings(cfg.data.resources["label_mappings"])

    if cfg.data.preprocess.get("enabled", True):
        df = preprocess_dataframe(
            df,
            text_column=source_text,
            output_column=cfg.data.text_column,
            preprocess_cfg=cfg.data.preprocess,
            token_map=token_map,
        )

    if cfg.data.label_column in df.columns:
        df[cfg.data.label_column] = apply_label_map(df[cfg.data.label_column].astype(str), label_map)
    elif source_label and source_label in df.columns:
        df[cfg.data.label_column] = apply_label_map(df[source_label].astype(str), label_map)

    output_path = cfg.data.output.get("processed_path")
    if output_path:
        save_data(df, output_path, file_format=Path(output_path).suffix.lstrip("."))

    if cfg.data.required_columns:
        validate_columns(df, cfg.data.required_columns)

    return df


def _preprocess_uploaded_frame(
    df: pd.DataFrame,
    cfg,
    source_text_column: str,
    token_map=None,
) -> pd.DataFrame:
    validate_columns(df, [source_text_column])
    token_map = token_map or load_token_mappings(cfg.data.resources["token_mappings"])
    if cfg.data.preprocess.get("enabled", True):
        df = preprocess_dataframe(
            df,
            text_column=source_text_column,
            output_column=cfg.data.text_column,
            preprocess_cfg=cfg.data.preprocess,
            token_map=token_map,
        )
    else:
        df = df.copy()
        df[cfg.data.text_column] = df[source_text_column].fillna("").astype(str)
    return df


def _fit_plan(models_needed: list[str] | None) -> dict[str, bool]:
    needed = set(models_needed or ALL_MODELS)
    return {
        "token": TOKEN_MODEL in needed or UMEC_MODEL in needed,
        "equipment": EQUIPMENT_MODEL in needed or UMEC_MODEL in needed,
        "semantic": SEMANTIC_MODEL in needed or UMEC_MODEL in needed,
        "umec": UMEC_MODEL in needed,
    }


def _apply_fast_fit_config(cfg):
    """Lighter settings for interactive / on-the-fly API fitting."""
    sem = dict(cfg.models.semantic_similarity)
    sem.update(
        {
            "epochs": 1,
            "embedding_dim": min(int(sem.get("embedding_dim", 100)), 80),
            "cwem_augment": False,
            "cwem_repeats": 0,
            "show_progress": False,
            "remove_pc": False,
            "max_fit_rows": int(sem.get("max_fit_rows") or 5000),
            "workers": min(int(sem.get("workers", 4)), 2),
            "n_jobs": min(int(sem.get("n_jobs", 4)), 4),
        }
    )
    cfg.models.semantic_similarity = sem
    return cfg


def _apply_memory_tier_config(cfg, n_rows: int) -> tuple[object, str]:
    """
    Scale FastText / UMEC fit and predict chunk sizes to row count (avoids OOM exit 137).
    """
    cfg = _apply_fast_fit_config(cfg)
    tier = "standard"
    safety = dict(getattr(cfg.models, "large_dataset_safety", {}) or {})
    safety_enabled = bool(safety.get("enabled", True))
    medium_threshold = int(safety.get("medium_threshold", 4_000) or 4_000)
    large_threshold = int(safety.get("large_threshold", 20_000) or 20_000)
    huge_threshold = int(safety.get("huge_threshold", 50_000) or 50_000)
    extreme_threshold = int(safety.get("extreme_threshold", 100_000) or 100_000)

    sem = dict(cfg.models.semantic_similarity)
    umec = dict(cfg.models.umec)
    spectral = dict(umec.get("spectral", {}))

    if n_rows > medium_threshold:
        tier = "medium"
        sem.update(
            {
                "max_fit_rows": 2_000,
                "cwem_augment": False,
                "cwem_repeats": 0,
                "remove_pc": False,
                "epochs": 1,
            }
        )
        umec["fit_sample_rows"] = min(5_000, max(2_000, n_rows // 20))
        umec["predict_chunk_size"] = 3_000

    if n_rows > large_threshold:
        tier = "large"
        sem.update(
            {
                "max_fit_rows": 1_500,
                "embedding_dim": min(60, int(sem.get("embedding_dim", 80))),
                "workers": 1,
                "n_jobs": 2,
            }
        )
        umec["fit_sample_rows"] = min(4_000, max(1_500, n_rows // 30))
        umec["predict_chunk_size"] = 2_500
        spectral.update({"components": 2, "use_third_moment": False})

    if n_rows > huge_threshold:
        tier = "huge"
        sem.update(
            {
                "max_fit_rows": int(safety.get("huge_semantic_max_fit_rows", 1_000) or 1_000),
                "embedding_dim": int(safety.get("huge_semantic_embedding_dim", 50) or 50),
                "workers": int(safety.get("huge_semantic_workers", 1) or 1),
                "n_jobs": int(safety.get("huge_semantic_n_jobs", 1) or 1),
            }
        )
        umec["fit_sample_rows"] = int(
            safety.get("huge_umec_fit_sample_rows", 2_000) or 2_000
        )
        umec["predict_chunk_size"] = int(
            safety.get("huge_predict_chunk_size", 2_000) or 2_000
        )
        spectral.update({"enabled": True, "components": 2, "use_third_moment": False})

    if safety_enabled and n_rows > extreme_threshold:
        tier = "extreme"
        sem.update(
            {
                "max_fit_rows": int(
                    safety.get("extreme_semantic_max_fit_rows", 600) or 600
                ),
                "embedding_dim": int(
                    safety.get("extreme_semantic_embedding_dim", 40) or 40
                ),
                "workers": int(safety.get("extreme_semantic_workers", 1) or 1),
                "n_jobs": int(safety.get("extreme_semantic_n_jobs", 1) or 1),
            }
        )
        umec["fit_sample_rows"] = int(
            safety.get("extreme_umec_fit_sample_rows", 1_200) or 1_200
        )
        umec["predict_chunk_size"] = int(
            safety.get("extreme_predict_chunk_size", 1_200) or 1_200
        )
        spectral.update({"enabled": True, "components": 2, "use_third_moment": False})

    cfg.models.semantic_similarity = sem
    umec["spectral"] = spectral
    cfg.models.umec = umec
    return cfg, tier


def _slim_inference_dataframe(
    df: pd.DataFrame,
    source_text_column: str,
    text_column: str,
    label_column: str | None = None,
) -> pd.DataFrame:
    """Keep only columns needed for scoring (reduces RAM on 50k–170k uploads)."""
    keep: list[str] = []
    if "id" in df.columns:
        keep.append("id")
    for col in (source_text_column, text_column):
        if col and col in df.columns and col not in keep:
            keep.append(col)
    if label_column and label_column in df.columns and label_column not in keep:
        keep.append(label_column)
    if not keep:
        return df
    return df[keep].copy()


def _apply_config_overrides(cfg, overrides: dict | None):
    if not overrides:
        return cfg
    if "models" in overrides:
        cfg.models.token_matching = _deep_update(cfg.models.token_matching, overrides["models"].get("token_matching"))
        if hasattr(cfg.models, "equipment_based"):
            cfg.models.equipment_based = _deep_update(
                cfg.models.equipment_based, overrides["models"].get("equipment_based")
            )
        cfg.models.semantic_similarity = _deep_update(
            cfg.models.semantic_similarity, overrides["models"].get("semantic_similarity")
        )
        cfg.models.umec = _deep_update(cfg.models.umec, overrides["models"].get("umec"))
    if "data" in overrides:
        cfg.data.preprocess = _deep_update(cfg.data.preprocess, overrides["data"].get("preprocess"))
        cfg.data.resources = _deep_update(cfg.data.resources, overrides["data"].get("resources"))
    return cfg


def fit_on_dataframe(
    config_dir: str,
    df: pd.DataFrame,
    source_text_column: str,
    overrides: dict | None = None,
    custom_categories: dict | None = None,
    models_needed: list[str] | None = None,
    fast_fit: bool = False,
    reference_label_column: str | None = None,
) -> dict:
    """
    Unsupervised fit on the provided frame (in memory).

    Only fits classifiers required by ``models_needed`` (e.g. token-only skips FastText).
  """
    cfg = load_config(config_dir)
    cfg = _apply_config_overrides(cfg, overrides)

    token_map = load_token_mappings(cfg.data.resources["token_mappings"])
    df = _preprocess_uploaded_frame(df, cfg, source_text_column, token_map=token_map)

    memory_tier = "standard"
    if fast_fit:
        cfg, memory_tier = _apply_memory_tier_config(cfg, len(df))

    plan = _fit_plan(models_needed)
    logger = get_logger("pipeline", cfg.project.log_level)
    set_seed(cfg.project.random_state)
    if memory_tier != "standard":
        logger.info(
            "Memory tier %s: %s rows — sampled FastText/UMEC fit, chunked predict",
            memory_tier,
            len(df),
        )
    logger.info("Fitting on %s rows — plan: %s", len(df), plan)

    corpus = df[cfg.data.text_column].fillna("").astype(str)
    raw_corpus = df[source_text_column].fillna("").astype(str).tolist()
    failure_keywords = _resolve_failure_keywords(
        cfg,
        custom_categories,
        corpus=raw_corpus,
        df=df,
        reference_label_column=reference_label_column,
    )

    token_clf = equipment_clf = semantic_clf = None
    if plan["token"] or plan["equipment"] or plan["semantic"]:
        token_clf, equipment_clf, semantic_clf = _init_classifiers(
            cfg, failure_keywords=failure_keywords, token_map=token_map
        )

    if plan["token"]:
        logger.info("Fitting token matching")
        token_clf.fit(corpus)
    else:
        token_clf = None

    if plan["equipment"]:
        logger.info("Fitting equipment-based")
        equipment_clf.fit(corpus)
    else:
        equipment_clf = None

    if plan["semantic"]:
        logger.info("Fitting semantic similarity (FastText — slowest step)")
        semantic_clf.fit(corpus)
    else:
        semantic_clf = None

    umec = None
    if plan["umec"]:
        if not (plan["token"] and plan["equipment"] and plan["semantic"]):
            raise ValueError("UMEC requires token, equipment, and semantic bases to be fitted.")
        umec = _build_umec(cfg, token_clf, equipment_clf, semantic_clf)
        y = df[cfg.data.label_column] if cfg.data.label_column in df.columns else None
        logger.info("Fitting UMEC ensemble (ECOC + spectral moments)")
        umec.fit(df, y=y, column_name=cfg.data.text_column)

    slim_df = _slim_inference_dataframe(
        df,
        source_text_column,
        cfg.data.text_column,
        reference_label_column,
    )

    return {
        "config": cfg,
        "data": slim_df,
        "memory_tier": memory_tier,
        "source_text_column": source_text_column,
        "token_clf": token_clf,
        "equipment_clf": equipment_clf,
        "semantic_clf": semantic_clf,
        "umec": umec,
        "failure_keywords": failure_keywords,
        "token_map": token_map,
        "models_fitted": list(models_needed or ALL_MODELS),
    }


def run_train_from_dataframe(
    config_dir: str,
    df: pd.DataFrame,
    source_text_column: str,
    overrides: dict | None = None,
    custom_categories: dict | None = None,
) -> dict:
    """Train on upload and persist joblibs to disk."""
    artifacts = fit_on_dataframe(
        config_dir,
        df,
        source_text_column,
        overrides=overrides,
        custom_categories=custom_categories,
        models_needed=list(ALL_MODELS),
        fast_fit=False,
    )
    cfg = artifacts["config"]
    model_dir = _save_models(
        cfg,
        artifacts["token_clf"],
        artifacts["equipment_clf"],
        artifacts["semantic_clf"],
        artifacts["umec"],
    )
    get_logger("pipeline", cfg.project.log_level).info("Saved models to %s", model_dir)
    return artifacts


def run_train(
    config_dir: str,
    overrides: dict | None = None,
    custom_categories: dict | None = None,
) -> dict:
    cfg = load_config(config_dir)
    if overrides:
        if "models" in overrides:
            cfg.models.token_matching = _deep_update(cfg.models.token_matching, overrides["models"].get("token_matching"))
            if hasattr(cfg.models, "equipment_based"):
                cfg.models.equipment_based = _deep_update(
                    cfg.models.equipment_based, overrides["models"].get("equipment_based")
                )
            cfg.models.semantic_similarity = _deep_update(cfg.models.semantic_similarity, overrides["models"].get("semantic_similarity"))
            cfg.models.umec = _deep_update(cfg.models.umec, overrides["models"].get("umec"))
        if "data" in overrides:
            cfg.data.preprocess = _deep_update(cfg.data.preprocess, overrides["data"].get("preprocess"))
            cfg.data.resources = _deep_update(cfg.data.resources, overrides["data"].get("resources"))

    logger = get_logger("pipeline", cfg.project.log_level)
    set_seed(cfg.project.random_state)

    logger.info("Loading and preprocessing data")
    token_map = load_token_mappings(cfg.data.resources["token_mappings"])
    df = _load_and_preprocess(cfg, token_map=token_map)
    logger.info("Data ready: %s rows", len(df))
    corpus = df[cfg.data.source_text_column or cfg.data.text_column].fillna("").astype(str).tolist()
    failure_keywords = _resolve_failure_keywords(cfg, custom_categories, corpus=corpus, df=df)
    token_clf, equipment_clf, semantic_clf = _init_classifiers(
        cfg, failure_keywords=failure_keywords, token_map=token_map
    )
    _fit_base_classifiers(df, cfg, token_clf, equipment_clf, semantic_clf)
    umec = _build_umec(cfg, token_clf, equipment_clf, semantic_clf)

    y = df[cfg.data.label_column] if cfg.data.label_column in df.columns else None
    logger.info("Fitting UMEC ensemble (ECOC + spectral moments)")
    umec.fit(df, y=y, column_name=cfg.data.text_column)
    logger.info("UMEC training complete")

    sample_n = min(5, len(df))
    if sample_n > 0:
        sample_df = df.head(sample_n)
        tm_preds, tm_scores = token_clf.predict(sample_df, column_name=cfg.data.text_column)
        eq_preds, eq_scores = equipment_clf.predict(sample_df, column_name=cfg.data.text_column)
        ss_preds, ss_scores = semantic_clf.predict(sample_df, column_name=cfg.data.text_column)
        umec_preds, _ = umec.predict(sample_df, column_name=cfg.data.text_column)

        logger.info("Sample predictions (first %s rows):", sample_n)
        for idx in sample_df.index:
            tm_top = tm_scores.loc[idx].sort_values(ascending=False).head(3)
            eq_top = eq_scores.loc[idx].sort_values(ascending=False).head(3)
            ss_top = ss_scores.loc[idx].sort_values(ascending=False).head(3)
            logger.info(
                "Row %s | token=%s | equipment=%s | semantic=%s | umec=%s",
                idx,
                tm_top.to_dict(),
                eq_top.to_dict(),
                ss_top.to_dict(),
                umec_preds.loc[idx],
            )

    model_dir = _save_models(cfg, token_clf, equipment_clf, semantic_clf, umec)
    logger.info("Saved models to %s", model_dir)

    return {
        "config": cfg,
        "data": df,
        "token_clf": token_clf,
        "equipment_clf": equipment_clf,
        "semantic_clf": semantic_clf,
        "umec": umec,
    }


def run_evaluate(
    config_dir: str,
    overrides: dict | None = None,
    custom_categories: dict | None = None,
) -> dict:
    artifacts = run_train(config_dir, overrides=overrides, custom_categories=custom_categories)
    cfg = artifacts["config"]
    df = artifacts["data"]
    umec = artifacts["umec"]

    y_true = df[cfg.data.label_column] if cfg.data.label_column in df.columns else None
    if y_true is None:
        raise ValueError("Label column not found; evaluation requires labels.")

    y_pred, reduction = umec.predict(df, column_name=cfg.data.text_column)
    class_scores = umec.class_score_df(reduction)

    labels = list(umec.classes) + ["unclassified"]
    report_df = classification_report_df(y_true, y_pred, labels=labels)
    macro = macro_f1(y_true, y_pred, labels=labels)

    metrics_dir = ensure_dir(Path(cfg.project.output_dir) / "metrics")
    report_path = metrics_dir / "umec_classification_report.csv"
    report_df.to_csv(report_path)

    fig_dir = ensure_dir(Path(cfg.project.output_dir) / "figures")
    try:
        if y_true is not None and len(y_true.dropna()) > 0:
            plot_confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
                title="UMEC Confusion Matrix",
                save_path=str(fig_dir / "umec_confusion_matrix.png"),
            ),
            
        else:
            print("skipping confusion matrix plot due to missing or empty true labels")
    except Exception as e:
        print(f"Skipping confusion matrix: {e}")

    top2 = top_k_accuracy(class_scores, y_true, k=2)
    print("Evaluation completed")
    print("Metrics saved to:", report_path,)

    return {
        "report_path": str(report_path),
        "macro_f1": macro,
        "top2_accuracy": top2,
    }
