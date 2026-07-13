from __future__ import annotations

from typing import List

import pandas as pd

from umec.models.base import BaseUnsupervisedClassifier
from umec.models.umec import UMECClassifier


def explain_record(
    df: pd.DataFrame,
    index: int,
    text_column: str,
    label_column: str | None,
    base_classifiers: List[BaseUnsupervisedClassifier],
    umec: UMECClassifier,
    top_k: int = 5,
) -> str:
    if index not in df.index:
        raise KeyError(f"Index {index} not found in dataframe.")

    row = df.loc[[index]]
    lines = []
    lines.append("=" * 100)
    lines.append(f"INDEX: {index}")
    if label_column and label_column in row.columns:
        lines.append(f"ORIGINAL LABEL: {row[label_column].iloc[0]}")
    lines.append(f"TEXT: {row[text_column].iloc[0]}")

    pred, reduction = umec.predict(row, column_name=text_column)
    lines.append(f"UMEC PREDICTION: {pred.iloc[0]}")

    lines.append("\nUMEC REDUCTION STATS (TOP BITS):")
    reduction_row = reduction.iloc[0].sort_values(ascending=False).head(top_k)
    for bit, value in reduction_row.items():
        lines.append(f"  {bit:<25} {value:.4f}")

    lines.append("\nBASE MODEL SCORES:")
    for i, clf in enumerate(base_classifiers):
        _, scores = clf.predict(row, column_name=text_column)
        top = scores.iloc[0].sort_values(ascending=False).head(top_k)
        lines.append(f"  Model {i}:")
        for label, score in top.items():
            lines.append(f"    {label:<18} {score:.4f}")

    return "\n".join(lines)
