from __future__ import annotations

import io as _io
from pathlib import Path
from typing import Any

import pandas as pd


def read_data_buffer(buffer: bytes, filename: str, **kwargs: Any) -> pd.DataFrame:
    """Parse an uploaded file (raw bytes + original filename) into a DataFrame.

    Mirrors :func:`read_data` format detection but works on an in-memory buffer
    (e.g. a Flask upload) instead of a path. Used by the /api/upload route so CSV
    and Excel are parsed by the same logic.
    """
    suffix = Path(filename).suffix.lstrip(".").lower()
    if suffix == "csv":
        return pd.read_csv(_io.BytesIO(buffer), encoding="utf-8-sig", **kwargs)
    if suffix in {"xlsx", "xlsm"}:
        return pd.read_excel(_io.BytesIO(buffer), engine="openpyxl", **kwargs)
    if suffix == "xls":
        return pd.read_excel(_io.BytesIO(buffer), engine="xlrd", **kwargs)
    raise ValueError(f"Unsupported file type '.{suffix}'. Use .csv, .xlsx, or .xls.")


def read_data(path: str | Path, file_format: str | None = None, **kwargs: Any) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        df = pd.read_csv(path, **kwargs)
    elif fmt in {"xlsx", "xls"}:
        df = pd.read_excel(path, **kwargs)
    elif fmt == "parquet":
        df = pd.read_parquet(path, **kwargs)
    else:
        raise ValueError(f"Unsupported data format: {fmt}")

    return df


def save_data(df: pd.DataFrame, path: str | Path, file_format: str | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fmt = (file_format or path.suffix.lstrip(".")).lower()
    if fmt == "csv":
        df.to_csv(path, index=False)
    elif fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unsupported output format: {fmt}")
