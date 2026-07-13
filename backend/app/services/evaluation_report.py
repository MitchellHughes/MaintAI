"""Build scoped classification reports for API and CLI."""

from __future__ import annotations

from typing import Any

from umec.evaluation.scoped_report import build_scoped_classification_report


def evaluation_report_from_predictions(
    predictions: list[dict[str, Any]],
    custom_categories: list[dict[str, Any]],
    *,
    pred_key: str = "final_condition",
    actual_key: str = "actual_label",
    label_groups: list[dict[str, Any]] | None = None,
    top_support_n: int = 10,
    top_k: int = 3,
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("predictions must be a non-empty list.")
    if not custom_categories:
        raise ValueError("custom_categories must include at least one label.")

    return build_scoped_classification_report(
        predictions,
        custom_categories,
        pred_key=pred_key,
        actual_key=actual_key,
        label_groups=label_groups,
        top_support_n=top_support_n,
        top_k=top_k,
    )
