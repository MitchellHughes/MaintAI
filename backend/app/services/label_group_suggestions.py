"""Suggest label merges from classification report signals."""

from __future__ import annotations

from typing import Any

from umec.evaluation.category_matching import category_labels, normalize_label
from umec.evaluation.label_groups import normalize_label_groups
from umec.evaluation.scoped_report import build_scoped_classification_report

# Maintenance-domain clusters — only suggested when 2+ labels exist in the user's config.
SEMANTIC_CLUSTERS: list[tuple[str, list[str]]] = [
    ("failed / faulty", ["faulty", "malfunctioned", "failed", "faulted"]),
    ("structural damage", ["damaged", "broken", "dented", "cracked"]),
    ("surface / wear", ["worn", "corroded", "delaminated"]),
    ("leak / pressure", ["leaking", "low pressure", "discharged"]),
    ("loose / adjustment", ["loose", "out of adjust", "excess play", "unsecure"]),
    ("missing / detached", ["missing", "detached"]),
    ("contamination", ["dirty", "odor"]),
]


def _per_class_map(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {normalize_label(row.get("label")): row for row in report.get("per_class") or []}


def suggest_label_groups(
    report: dict[str, Any],
    categories: list[dict[str, Any]],
    *,
    max_groups: int = 12,
    low_f1_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """Return merge group suggestions from semantic clusters and confusion patterns."""
    user_labels = set(category_labels(categories))
    per_class = _per_class_map(report)
    suggestions: list[dict[str, Any]] = []
    used: set[str] = set()

    for group_label, members in SEMANTIC_CLUSTERS:
        present = [member for member in members if member in user_labels and member not in used]
        if len(present) < 2:
            continue

        low_f1 = [
            member
            for member in present
            if float(per_class.get(member, {}).get("f1_score", 0)) <= low_f1_threshold
        ]
        if len(low_f1) < 2 and len(present) < 3:
            continue

        suggestions.append(
            {
                "group_label": group_label,
                "members": sorted(present),
                "reason": (
                    f"{len(present)} related categories with weak individual F1 "
                    f"(≤ {int(low_f1_threshold * 100)}% for most)."
                ),
                "source": "semantic",
            }
        )
        used.update(present)

    for pair in report.get("top_confusion_pairs") or []:
        actual = normalize_label(pair.get("actual"))
        predicted = normalize_label(pair.get("predicted"))
        count = int(pair.get("count") or 0)
        if not actual or not predicted or actual == predicted:
            continue
        if actual not in user_labels or predicted not in user_labels:
            continue
        if actual in used or predicted in used:
            continue
        if count < 8:
            continue

        actual_f1 = float(per_class.get(actual, {}).get("f1_score", 1))
        predicted_f1 = float(per_class.get(predicted, {}).get("f1_score", 1))
        if actual_f1 > 0.55 and predicted_f1 > 0.55:
            continue

        group_label = f"{actual} / {predicted}"
        suggestions.append(
            {
                "group_label": group_label,
                "members": sorted({actual, predicted}),
                "reason": f"Frequent cross-confusion ({count} rows): actual “{actual}” predicted as “{predicted}”.",
                "source": "confusion",
            }
        )
        used.update({actual, predicted})

        if len(suggestions) >= max_groups:
            break

    return suggestions[:max_groups]


def label_group_preview(
    predictions: list[dict[str, Any]],
    categories: list[dict[str, Any]],
    label_groups: list[dict[str, Any]] | None,
    *,
    pred_key: str = "final_condition",
    actual_key: str = "actual_label",
) -> dict[str, Any]:
    """Compare macro F1 before and after applying label groups."""
    baseline = build_scoped_classification_report(
        predictions,
        categories,
        pred_key=pred_key,
        actual_key=actual_key,
    )
    if baseline.get("error"):
        return {
            "baseline_macro_f1": None,
            "merged_macro_f1": None,
            "baseline_report": baseline,
            "merged_report": baseline,
            "suggested_groups": [],
        }

    normalized = normalize_label_groups(label_groups)
    merged = build_scoped_classification_report(
        predictions,
        categories,
        pred_key=pred_key,
        actual_key=actual_key,
        label_groups=normalized,
    )

    suggested = suggest_label_groups(baseline, categories)
    if not normalized and suggested:
        auto_merged = build_scoped_classification_report(
            predictions,
            categories,
            pred_key=pred_key,
            actual_key=actual_key,
            label_groups=suggested,
        )
    else:
        auto_merged = merged

    return {
        "baseline_macro_f1": baseline.get("macro_f1"),
        "merged_macro_f1": merged.get("macro_f1"),
        "suggested_macro_f1": auto_merged.get("macro_f1"),
        "baseline_classes": baseline.get("target_classes"),
        "merged_classes": merged.get("target_classes"),
        "suggested_classes": auto_merged.get("target_classes"),
        "suggested_groups": suggested,
        "available_labels": category_labels(categories),
        "baseline_report": baseline,
        "merged_report": merged,
    }
