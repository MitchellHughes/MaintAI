#!/usr/bin/env python
"""
Generate a classification report scoped to your failure-mechanism categories.

Typical workflow after running predictions in the dashboard:
  1. Export findings (or use prediction rows with actual_label filled).
  2. Save your category list as JSON.
  3. Run this script.

Examples
--------
From dashboard findings export (CSV):

  python scripts/classification_report.py \\
    --findings reports/findings.csv \\
    --categories configs/my_site_categories.json \\
    --output reports/classification_report.csv

From a merged dataset (true label + predicted columns in one file):

  python scripts/classification_report.py \\
    --input data/eval_set.csv \\
    --label-column PartCondition \\
    --pred-column FAILURE_MECHANISM \\
    --categories configs/my_site_categories.json

From saved API prediction JSON (results_by_model.UMECClassifier):

  python scripts/classification_report.py \\
    --predictions-json reports/last_run.json \\
    --model UMECClassifier \\
    --categories configs/my_site_categories.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from umec.data.io import read_data
from umec.evaluation.plots import plot_confusion_matrix
from umec.evaluation.scoped_report import build_scoped_classification_report


def _load_categories(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "custom_categories" in data:
        data = data["custom_categories"]
    if not isinstance(data, list):
        raise ValueError("Categories file must be a JSON array or {custom_categories: [...]}.")
    return data


def _load_rows_from_findings(path: Path) -> list[dict]:
    df = read_data(path, file_format=path.suffix.lstrip("."))
    return df.fillna("").astype(str).to_dict(orient="records")


def _load_rows_from_input(
    path: Path,
    *,
    label_column: str,
    pred_column: str,
) -> list[dict]:
    df = read_data(path, file_format=path.suffix.lstrip("."))
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found. Columns: {list(df.columns)}")
    if pred_column not in df.columns:
        raise ValueError(f"Prediction column '{pred_column}' not found. Columns: {list(df.columns)}")
    rows = []
    for i, row in df.iterrows():
        rows.append(
            {
                "row_id": i,
                "actual_label": str(row[label_column]),
                "final_condition": str(row[pred_column]),
                "predicted_condition": str(row[pred_column]),
            }
        )
    return rows


def _load_rows_from_predictions_json(path: Path, model: str) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    by_model = payload.get("results_by_model") or {}
    if model in by_model:
        return by_model[model]
    if isinstance(payload.get("predictions"), list):
        return payload["predictions"]
    raise ValueError(
        f"JSON must contain results_by_model['{model}'] or a top-level 'predictions' list."
    )


def _print_summary(report: dict) -> None:
    if report.get("error"):
        print(f"Error: {report['error']}")
        return
    print(f"Target classes:     {report['target_classes']}")
    print(f"Evaluated rows:     {report['evaluated_rows']}")
    print(f"Skipped rows:       {report['skipped_rows']} (reference not mapped to your categories)")
    print(f"Accuracy:           {report['accuracy']:.2%}")
    print(f"Macro F1:           {report['macro_f1']:.2%}")
    print(f"Macro precision:    {report['macro_precision']:.2%}")
    print(f"Macro recall:       {report['macro_recall']:.2%}")
    print()
    print(f"{'Label':<24} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 66)
    for row in report.get("per_class", []):
        print(
            f"{row['label']:<24} "
            f"{row['precision']:>10.2%} "
            f"{row['recall']:>10.2%} "
            f"{row['f1_score']:>10.2%} "
            f"{row['support']:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classification report over your defined categories (reference labels required)."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--findings", type=Path, help="Findings CSV/XLSX from dashboard export.")
    src.add_argument("--input", type=Path, help="Dataset with label + prediction columns.")
    src.add_argument(
        "--predictions-json",
        type=Path,
        help="Saved predict API JSON (uses results_by_model).",
    )

    parser.add_argument(
        "--categories",
        type=Path,
        required=True,
        help="JSON array of {label, keywords} (same as dashboard categories).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/classification_report.csv"),
        help="Write per-class + summary table to CSV.",
    )
    parser.add_argument(
        "--confusion-matrix-png",
        type=Path,
        help="Optional path to save a confusion matrix heatmap image.",
    )
    parser.add_argument(
        "--label-column",
        help="Reference/true label column (with --input).",
    )
    parser.add_argument(
        "--pred-column",
        default="final_condition",
        help="Predicted label column (with --input). Default: final_condition",
    )
    parser.add_argument(
        "--pred-key",
        default="final_condition",
        help="Field to score in findings/JSON rows. Default: final_condition",
    )
    parser.add_argument(
        "--model",
        default="UMECClassifier",
        help="Model key when using --predictions-json. Default: UMECClassifier",
    )
    args = parser.parse_args()

    categories = _load_categories(args.categories)

    if args.findings:
        rows = _load_rows_from_findings(args.findings)
        actual_key = "actual_label"
    elif args.input:
        if not args.label_column:
            parser.error("--label-column is required with --input")
        rows = _load_rows_from_input(
            args.input,
            label_column=args.label_column,
            pred_column=args.pred_column,
        )
        actual_key = "actual_label"
    else:
        rows = _load_rows_from_predictions_json(args.predictions_json, args.model)
        actual_key = "actual_label"

    report = build_scoped_classification_report(
        rows,
        categories,
        pred_key=args.pred_key,
        actual_key=actual_key,
    )

    _print_summary(report)

    if report.get("error"):
        sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(report.get("report_table", []))
    table.to_csv(args.output, index=False)
    print(f"\nReport table saved to {args.output.resolve()}")

    if args.confusion_matrix_png:
        labels = report.get("labels") or []
        matrix = report.get("confusion_matrix") or []
        if labels and matrix:
            n = len(labels)
            size = max(10, min(28, n * 0.55))
            y_true = []
            y_pred = []
            for i, row in enumerate(matrix):
                for j, count in enumerate(row):
                    c = int(count)
                    if c > 0:
                        y_true.extend([labels[i]] * c)
                        y_pred.extend([labels[j]] * c)
            args.confusion_matrix_png.parent.mkdir(parents=True, exist_ok=True)
            plot_confusion_matrix(
                y_true,
                y_pred,
                labels=labels,
                title="Confusion matrix (actual vs predicted)",
                figsize=(size, size * 0.92),
                save_path=str(args.confusion_matrix_png),
            )
            print(f"Confusion matrix PNG saved to {args.confusion_matrix_png.resolve()}")


if __name__ == "__main__":
    main()
