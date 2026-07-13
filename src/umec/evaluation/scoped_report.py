"""Classification report scoped to user-defined categories (not all CMMS codes)."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from umec.evaluation.category_matching import build_category_matcher, category_labels, normalize_label
from umec.evaluation.label_groups import apply_groups_to_pairs, evaluation_labels, normalize_label_groups


def _rows_to_pairs(
    rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    *,
    pred_key: str = "final_condition",
    actual_key: str = "actual_label",
    label_groups: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str], dict[str, int]]:
    match = build_category_matcher(categories, label_groups=label_groups)
    y_true: list[str] = []
    y_pred: list[str] = []
    skipped = 0

    for row in rows:
        actual = match(row.get(actual_key))
        if not actual:
            skipped += 1
            continue
        raw_pred = row.get(pred_key) or row.get("predicted_condition") or ""
        pred = match(raw_pred) or normalize_label(raw_pred)
        y_true.append(actual)
        y_pred.append(pred)

    return y_true, y_pred, {"skipped_rows": skipped, "evaluated_rows": len(y_true)}


def _top_k_labels_for_row(
    row: dict[str, Any],
    match,
    *,
    k: int,
) -> list[str]:
    """
    Merge ranked labels for top-K evaluation.

    Always includes the model's assigned ``predicted_condition`` first so the
    ranked list stays aligned with what was actually predicted (stored ranks can
    be shorter or filtered differently after keyword evidence).

    Strictly model-only for evaluation: no reference-label injection.
    """
    labels: list[str] = []
    seen: set[str] = set()

    def _add(raw: Any) -> None:
        if len(labels) >= max(1, k):
            return
        mapped = match(raw) or normalize_label(raw)
        if not mapped or mapped in seen:
            return
        labels.append(mapped)
        seen.add(mapped)

    _add(row.get("predicted_condition"))

    simple = (row.get("xai") or {}).get("simple") or {}
    sources: list[list[Any]] = []
    if row.get("top_predictions"):
        sources.append(row["top_predictions"])
    if simple.get("top_k_details"):
        sources.append(simple["top_k_details"])
    if simple.get("top_ranked"):
        sources.append(simple["top_ranked"])

    for ranked in sources:
        for item in ranked:
            if len(labels) >= max(1, k):
                break
            raw = item.get("label") if isinstance(item, dict) else item
            _add(raw)

    for raw in (row.get("runner_up"), simple.get("runner_up")):
        _add(raw)

    return labels


def _top_k_relaxed_pairs(
    rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    *,
    k: int = 3,
    actual_key: str = "actual_label",
    label_groups: list[dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """
    Relaxed top-K labels for metrics: if reference is in top-K, treat as that label;
    otherwise use the top-1 ranked prediction.
    """
    match = build_category_matcher(categories, label_groups=label_groups)
    y_true: list[str] = []
    y_pred: list[str] = []
    for row in rows:
        actual = match(row.get(actual_key))
        if not actual:
            continue
        top_labels = _top_k_labels_for_row(row, match, k=k)
        fallback = (
            top_labels[0]
            if top_labels
            else match(row.get("predicted_condition"))
            or normalize_label(row.get("predicted_condition") or "")
        )
        pred = actual if actual in top_labels else (fallback or actual)
        y_true.append(actual)
        y_pred.append(pred)
    return y_true, y_pred


def top_k_hit_accuracy(
    rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    *,
    k: int = 3,
    actual_key: str = "actual_label",
    label_groups: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Fraction of rows where the mapped reference label appears in top-K predictions.
    """
    match = build_category_matcher(categories, label_groups=label_groups)
    hits = 0
    evaluated = 0
    for row in rows:
        actual = match(row.get(actual_key))
        if not actual:
            continue
        evaluated += 1
        if actual in _top_k_labels_for_row(row, match, k=k):
            hits += 1
    accuracy = (hits / evaluated) if evaluated else 0.0
    return {
        "top_k": max(1, k),
        "top_k_accuracy": round(accuracy, 4),
        "top_k_hits": hits,
        "top_k_evaluated_rows": evaluated,
    }


def top_k_relaxed_f1(
    rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    *,
    k: int = 3,
    actual_key: str = "actual_label",
    labels: list[str] | None = None,
) -> dict[str, Any]:
    """Macro/weighted F1 when a correct pick from top-K counts as that class."""
    y_true, y_pred = _top_k_relaxed_pairs(
        rows, categories, k=k, actual_key=actual_key
    )
    if not y_true:
        return {
            "top_k_macro_f1": 0.0,
            "top_k_macro_precision": 0.0,
            "top_k_macro_recall": 0.0,
            "top_k_weighted_f1": 0.0,
            "top_k_per_class": {},
        }

    eval_labels = labels or evaluation_labels(categories, [])
    report = classification_report(
        y_true,
        y_pred,
        labels=eval_labels,
        zero_division=0,
        output_dict=True,
    )
    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    per_class: dict[str, dict[str, float]] = {}
    for label in eval_labels:
        if label not in report:
            continue
        row = report[label]
        per_class[label] = {
            "precision": round(float(row["precision"]), 4),
            "recall": round(float(row["recall"]), 4),
            "f1_score": round(float(row["f1-score"]), 4),
        }

    return {
        "top_k_macro_f1": round(float(macro.get("f1-score", 0)), 4),
        "top_k_macro_precision": round(float(macro.get("precision", 0)), 4),
        "top_k_macro_recall": round(float(macro.get("recall", 0)), 4),
        "top_k_weighted_f1": round(float(weighted.get("f1-score", 0)), 4),
        "top_k_per_class": per_class,
    }


def build_scoped_classification_report(
    rows: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    *,
    pred_key: str = "final_condition",
    actual_key: str = "actual_label",
    label_groups: list[dict[str, Any]] | None = None,
    top_support_n: int = 10,
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Build sklearn classification report over mapped labels only.

    Rows whose reference label cannot be mapped to a user category are excluded.
    Optional label_groups collapse member categories before metrics are computed.
    """
    normalized_groups = normalize_label_groups(label_groups)
    labels = evaluation_labels(categories, normalized_groups)
    if not labels:
        raise ValueError("At least one category with a non-empty label is required.")

    y_true, y_pred, counts = _rows_to_pairs(
        rows,
        categories,
        pred_key=pred_key,
        actual_key=actual_key,
        label_groups=normalized_groups,
    )
    if not y_true:
        return {
            "error": "No rows could be mapped to your categories for evaluation.",
            "target_classes": len(labels),
            **counts,
        }

    if normalized_groups:
        y_true, y_pred = apply_groups_to_pairs(y_true, y_pred, normalized_groups)

    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        zero_division=0,
        output_dict=True,
    )
    report_df = pd.DataFrame(report).transpose()

    per_class = []
    for label in labels:
        if label not in report:
            continue
        row = report[label]
        per_class.append(
            {
                "label": label,
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1_score": round(float(row["f1-score"]), 4),
                "support": int(row["support"]),
            }
        )

    macro = report.get("macro avg", {})
    weighted = report.get("weighted avg", {})
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    strong_classes = macro_f1_for_top_support(per_class, top_n=top_support_n)

    y_true_k, y_pred_k = _top_k_relaxed_pairs(
        rows,
        categories,
        k=top_k,
        actual_key=actual_key,
        label_groups=normalized_groups,
    )
    if normalized_groups:
        y_true_k, y_pred_k = apply_groups_to_pairs(y_true_k, y_pred_k, normalized_groups)

    hit_stats = top_k_hit_accuracy(
        rows,
        categories,
        k=top_k,
        actual_key=actual_key,
        label_groups=normalized_groups,
    )
    top_k_stats = {
        "top_k": max(1, top_k),
        "top_k_evaluated_rows": hit_stats["top_k_evaluated_rows"],
        "top_k_hits": hit_stats["top_k_hits"],
        "top_k_accuracy": hit_stats["top_k_accuracy"],
        "top_k_relaxed_hits": sum(1 for t, p in zip(y_true_k, y_pred_k) if t == p),
    }
    if top_k_stats["top_k_evaluated_rows"]:
        top_k_stats["top_k_relaxed_accuracy"] = round(
            top_k_stats["top_k_relaxed_hits"] / top_k_stats["top_k_evaluated_rows"],
            4,
        )
    else:
        top_k_stats["top_k_relaxed_accuracy"] = 0.0

    top_k_f1_stats = {"top_k_per_class": {}}
    if y_true_k:
        tk_report = classification_report(
            y_true_k,
            y_pred_k,
            labels=labels,
            zero_division=0,
            output_dict=True,
        )
        tk_macro = tk_report.get("macro avg", {})
        tk_weighted = tk_report.get("weighted avg", {})
        top_k_f1_stats.update(
            {
                "top_k_macro_f1": round(float(tk_macro.get("f1-score", 0)), 4),
                "top_k_macro_precision": round(float(tk_macro.get("precision", 0)), 4),
                "top_k_macro_recall": round(float(tk_macro.get("recall", 0)), 4),
                "top_k_weighted_f1": round(float(tk_weighted.get("f1-score", 0)), 4),
            }
        )
        for label in labels:
            if label not in tk_report:
                continue
            row = tk_report[label]
            top_k_f1_stats["top_k_per_class"][label] = {
                "precision": round(float(row["precision"]), 4),
                "recall": round(float(row["recall"]), 4),
                "f1_score": round(float(row["f1-score"]), 4),
            }
    else:
        top_k_f1_stats.update(
            {
                "top_k_macro_f1": 0.0,
                "top_k_macro_precision": 0.0,
                "top_k_macro_recall": 0.0,
                "top_k_weighted_f1": 0.0,
            }
        )

    top_k_per_class = top_k_f1_stats.pop("top_k_per_class", {})
    tk_lookup = {
        normalize_label(key): value for key, value in top_k_per_class.items()
    }
    for row in per_class:
        extra = tk_lookup.get(normalize_label(row["label"]), {})
        row["top_k_f1"] = extra.get("f1_score", 0.0)
        row["top_k_recall"] = extra.get("recall", 0.0)
        row["top_k_precision"] = extra.get("precision", 0.0)

    top_k_f1_stats.setdefault("top_k_macro_f1", 0.0)
    top_k_f1_stats.setdefault("top_k_macro_precision", 0.0)
    top_k_f1_stats.setdefault("top_k_macro_recall", 0.0)
    top_k_f1_stats.setdefault("top_k_weighted_f1", 0.0)

    return {
        "target_classes": len(labels),
        "evaluated_rows": counts["evaluated_rows"],
        "skipped_rows": counts["skipped_rows"],
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_f1": round(float(macro.get("f1-score", 0)), 4),
        "macro_precision": round(float(macro.get("precision", 0)), 4),
        "macro_recall": round(float(macro.get("recall", 0)), 4),
        "weighted_f1": round(float(weighted.get("f1-score", 0)), 4),
        "strong_classes": strong_classes,
        **top_k_stats,
        **top_k_f1_stats,
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": cm.tolist(),
        "top_confusion_pairs": top_confusion_pairs(labels, cm.tolist(), top_n=12),
        "report_table": report_df.reset_index(names="class").to_dict(orient="records"),
        "pred_key": pred_key,
        "actual_key": actual_key,
        "label_groups": normalized_groups,
        "grouped_classes": len(normalized_groups),
    }


def macro_f1_for_top_support(
    per_class: list[dict[str, Any]],
    *,
    top_n: int = 10,
) -> dict[str, Any]:
    """
    Macro F1 over the top-N classes by reference support in this evaluation run only.
    Not tied to any external dataset or fixed label list.
    """
    if not per_class:
        return {"macro_f1": 0.0, "top_n": 0, "labels": [], "total_support": 0}

    ranked = sorted(per_class, key=lambda row: int(row.get("support") or 0), reverse=True)
    top = ranked[: max(1, top_n)]
    macro_f1 = sum(float(row.get("f1_score") or 0) for row in top) / len(top)
    return {
        "macro_f1": round(macro_f1, 4),
        "macro_precision": round(
            sum(float(row.get("precision") or 0) for row in top) / len(top),
            4,
        ),
        "macro_recall": round(
            sum(float(row.get("recall") or 0) for row in top) / len(top),
            4,
        ),
        "top_n": len(top),
        "labels": [str(row.get("label")) for row in top],
        "total_support": int(sum(int(row.get("support") or 0) for row in top)),
    }


def top_confusion_pairs(
    labels: list[str],
    matrix: list[list[int]],
    *,
    top_n: int = 10,
) -> list[dict[str, Any]]:
    """Off-diagonal confusion counts sorted descending (label drift signals)."""
    pairs: list[dict[str, Any]] = []
    for i, actual in enumerate(labels):
        for j, predicted in enumerate(labels):
            if i == j:
                continue
            count = int(matrix[i][j]) if matrix[i][j] is not None else 0
            if count <= 0:
                continue
            pairs.append(
                {
                    "actual": actual,
                    "predicted": predicted,
                    "count": count,
                }
            )
    pairs.sort(key=lambda item: (-item["count"], item["actual"], item["predicted"]))
    return pairs[:top_n]
