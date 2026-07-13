from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score


def filter_defined_classes(
    y_true: pd.Series,
    defined_classes: Iterable[str],
) -> tuple[pd.Series, pd.Series]:
    defined_classes = set(defined_classes)
    mask = y_true.isin(defined_classes)
    return y_true[mask], mask


def top_k_accuracy(score_df: pd.DataFrame, y_true: pd.Series, k: int = 2) -> float:
    y_true = y_true.reset_index(drop=True)
    scores = score_df.reset_index(drop=True)

    hits = 0
    for i in range(len(y_true)):
        row_scores = scores.iloc[i].values
        if np.sum(row_scores) == 0:
            continue
        top_k_idx = np.argsort(row_scores)[-k:][::-1]
        top_k_labels = score_df.columns[top_k_idx].tolist()
        if y_true.iloc[i] in top_k_labels:
            hits += 1
    return hits / max(len(y_true), 1)


def classification_report_df(y_true: pd.Series, y_pred: pd.Series, labels: Iterable[str]) -> pd.DataFrame:
    report = classification_report(
        y_true,
        y_pred,
        labels=list(labels),
        zero_division=0,
        output_dict=True,
    )
    return pd.DataFrame(report).transpose()


def macro_f1(y_true: pd.Series, y_pred: pd.Series, labels: Iterable[str]) -> float:
    return f1_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0)
