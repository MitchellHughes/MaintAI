from __future__ import annotations

from typing import Protocol

import pandas as pd


class BaseUnsupervisedClassifier(Protocol):
    classes: list[str]

    def fit(self, corpus) -> None:
        ...

    def transform(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> pd.DataFrame:
        ...

    def predict(self, df: pd.DataFrame, column_name: str = "processed_discrepancy") -> tuple[pd.Series, pd.DataFrame]:
        ...
