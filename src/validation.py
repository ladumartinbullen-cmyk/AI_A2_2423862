"""Validation of seeds, datasets and single prediction records."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from .config import (
    FEATURES,
    FORBIDDEN_FEATURES,
    MIN_YIELD,
    N_DUPLICATES,
    NUMERIC_FEATURES,
    RANGES,
    REQUIRED_COLUMNS,
    SEASONS,
    expected_base_rows,
    expected_missing,
    parse_student_id,
)


class DataValidationError(ValueError):
    """Raised when the dataset breaks schema, range, row-count or leakage rules."""


class RecordValidationError(ValueError):
    """Raised when a single prediction record is invalid. `.errors` lists every problem."""

    def __init__(self, errors: list[str]):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def validate_seed(student_id: str, seed: int) -> str:
    """Reject a seed that is not LLLL (last four registration digits). Returns registration digits."""
    reg_digits, last4 = parse_student_id(student_id)
    if int(last4) != int(seed):
        raise ValueError(
            f"Invalid seed {seed}: it must equal the last four digits of the registration "
            f"number ({int(last4)}) taken from '{student_id}'."
        )
    return reg_digits


def validate_dataframe(df: pd.DataFrame, seed: int) -> dict[str, Any]:
    """Check columns, ranges, missing values, duplicates and leakage. Returns an evidence report."""
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        raise DataValidationError(f"Missing required columns: {missing_cols}")

    leaked = FORBIDDEN_FEATURES & set(FEATURES)
    if leaked:
        raise DataValidationError(f"Target leakage: forbidden columns used as features: {sorted(leaked)}")

    problems: list[str] = []
    tol = 1e-6
    for col in NUMERIC_FEATURES:
        lo, hi = RANGES[col]
        values = df[col].dropna()
        bad = int(((values < lo - tol) | (values > hi + tol)).sum())
        if bad:
            problems.append(f"{col}: {bad} value(s) outside [{lo}, {hi}]")
    if not set(df["season"].dropna().unique()) <= set(SEASONS):
        problems.append("season contains values other than A/B")
    if int((df["yield_tons"] < MIN_YIELD - tol).sum()):
        problems.append(f"yield_tons has values below {MIN_YIELD}")
    if not set(df["high_risk"].dropna().unique()) <= {0, 1}:
        problems.append("high_risk must be 0/1")
    for col in ("yield_tons", "high_risk", "rainfall_mm", "avg_temp_c", "field_size_ha", "season", "cooperative_support"):
        if df[col].isna().any():
            problems.append(f"{col} has unexpected missing values")

    expected_rows = expected_base_rows(seed) + N_DUPLICATES
    if len(df) != expected_rows:
        problems.append(
            f"row count {len(df)} != expected {expected_rows} "
            f"(base {expected_base_rows(seed)} + {N_DUPLICATES} duplicates)"
        )
    if problems:
        raise DataValidationError("; ".join(problems))

    return {
        "rows_raw": int(len(df)),
        "expected_base_rows": expected_base_rows(seed),
        "expected_missing_per_column": expected_missing(seed),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v},
        "duplicate_rows": int(df.duplicated().sum()),
        "leakage_check": "passed: yield_tons, yield per hectare and high_risk are not model inputs",
        "range_check": "passed",
        "schema_check": "passed",
    }


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_record(record: Any) -> dict[str, Any]:
    """Validate one farm record (missing fields, wrong types, ranges). Returns a clean dict."""
    if not isinstance(record, dict):
        raise RecordValidationError(["Input must be a JSON object with the seven farm fields."])
    errors: list[str] = []
    for field in FEATURES:
        if field not in record:
            errors.append(f"Missing required field: {field}")
    for field in record:
        if field not in FEATURES:
            errors.append(f"Unknown field: {field}")

    clean: dict[str, Any] = {}
    for field in NUMERIC_FEATURES:
        if field not in record:
            continue
        value = record[field]
        if field == "cooperative_support":
            if not _is_number(value) or value not in (0, 1):
                errors.append("cooperative_support must be the number 0 or 1")
            else:
                clean[field] = int(value)
            continue
        if not _is_number(value) or not math.isfinite(value):
            errors.append(f"{field} must be a finite number, got {value!r}")
            continue
        lo, hi = RANGES[field]
        if not lo <= value <= hi:
            errors.append(f"{field}={value} is outside the allowed range [{lo}, {hi}]")
        else:
            clean[field] = float(value)
    if "season" in record:
        if not isinstance(record["season"], str) or record["season"] not in SEASONS:
            errors.append("season must be the text 'A' or 'B'")
        else:
            clean["season"] = record["season"]
    if errors:
        raise RecordValidationError(errors)
    return clean
