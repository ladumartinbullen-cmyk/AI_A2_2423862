"""Loading and cleaning the raw CSV (no statistics are fitted here: imputation happens in the pipeline)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_raw(csv_path: Path) -> pd.DataFrame:
    """Read the raw CSV."""
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}. Run run_all.py first.")
    return pd.read_csv(csv_path)


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Drop exact duplicate rows, keep the original CSV position as `row_id`."""
    df = df.reset_index(drop=True)
    df.insert(0, "row_id", df.index)
    cols = [c for c in df.columns if c != "row_id"]
    cleaned = df.drop_duplicates(subset=cols, keep="first").reset_index(drop=True)
    info = {
        "rows_before": int(len(df)),
        "duplicates_removed": int(len(df) - len(cleaned)),
        "rows_after": int(len(cleaned)),
        "missing_after_cleaning": {
            c: int(v) for c, v in cleaned.drop(columns="row_id").isna().sum().items() if v
        },
        "missing_handling": "median imputation fitted on the training split only (inside the sklearn Pipeline)",
    }
    return cleaned, info
