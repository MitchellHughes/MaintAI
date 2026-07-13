from __future__ import annotations

import re
from typing import Dict, Iterable

import pandas as pd


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_non_alnum: bool = True,
    collapse_spaces: bool = True,
) -> str:
    if text is None:
        return ""

    text = str(text)
    if lowercase:
        text = text.lower()
    if remove_non_alnum:
        text = re.sub(r"[^a-z0-9\s\-/]", " ", text)
    if collapse_spaces:
        text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_tokens(text: str, token_map: Dict[str, str] | None) -> str:
    if not token_map:
        return text

    tokens = text.split()
    normalized = [token_map.get(tok, tok) for tok in tokens]
    return " ".join(normalized)


def apply_label_map(series: pd.Series, label_map: Dict[str, str] | None) -> pd.Series:
    if not label_map:
        return series

    return series.map(lambda x: label_map.get(str(x), str(x)))


def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str,
    output_column: str,
    preprocess_cfg: Dict[str, bool],
    token_map: Dict[str, str] | None = None,
) -> pd.DataFrame:
    df = df.copy()
    df[output_column] = (
        df[text_column]
        .fillna("")
        .astype(str)
        .apply(
            lambda x: clean_text(
                x,
                lowercase=preprocess_cfg.get("lowercase", True),
                remove_non_alnum=preprocess_cfg.get("remove_non_alnum", True),
                collapse_spaces=preprocess_cfg.get("collapse_spaces", True),
            )
        )
        .apply(lambda x: normalize_tokens(x, token_map))
    )

    return df
