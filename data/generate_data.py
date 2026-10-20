"""Reproducible synthetic dataset generator for the Musanze potato cooperative.

Usage:
    python data/generate_data.py --seed 3862 --student-id ug2423862@ines.ac.rw
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import N_DUPLICATES, Paths, parse_student_id, verification_code  # noqa: E402


def generate_dataframe(seed: int) -> pd.DataFrame:
    """Create the raw dataset (with deliberate missing values and duplicates) from `seed`."""
    rng = np.random.default_rng(seed)
    n = 240 + seed % 41  # base rows

    rainfall = rng.uniform(600, 1400, n)
    temp = rng.uniform(12, 26, n)
    soil = rng.uniform(20, 80, n)
    fert = rng.uniform(0, 250, n)
    size = rng.uniform(0.2, 4.0, n)
    season = rng.choice(["A", "B"], n)
    support = rng.integers(0, 2, n)
    noise = rng.normal(0.0, 0.80, n)

    season_effect = np.where(season == "A", 0.50, -0.20)
    yield_tons = (
        0.004 * rainfall
        + 0.06 * soil
        + 0.02 * fert
        + 1.30 * size
        - 0.25 * np.abs(temp - 18)
        + 0.70 * support
        + season_effect
        + noise
    )
    yield_tons = np.clip(yield_tons, 0.50, None)
    yield_per_ha = yield_tons / size
    high_risk = (yield_per_ha < np.median(yield_per_ha)).astype(int)

    df = pd.DataFrame(
        {
            "rainfall_mm": rainfall,
            "avg_temp_c": temp,
            "soil_moisture_pct": soil,
            "fertilizer_kg": fert,
            "field_size_ha": size,
            "season": season,
            "cooperative_support": support,
            "yield_tons": yield_tons,
            "high_risk": high_risk,
        }
    )

    # Deliberate data problems
    m = int(round(0.05 * n))
    soil_pos = rng.choice(n, size=m, replace=False)
    fert_pos = rng.choice(n, size=m, replace=False)
    df.loc[soil_pos, "soil_moisture_pct"] = np.nan
    df.loc[fert_pos, "fertilizer_kg"] = np.nan
    dup_pos = rng.choice(n, size=N_DUPLICATES, replace=False)
    df = pd.concat([df, df.iloc[dup_pos]], ignore_index=True)
    return df


def sha256_file(path: Path) -> str:
    """SHA-256 of the exact CSV bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_dataset(seed: int, csv_path: Path, fingerprint_path: Path) -> str:
    """Write the CSV (4 decimals) and the fingerprint file. Returns the SHA-256 hex digest."""
    df = generate_dataframe(seed)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False, float_format="%.4f", lineterminator="\n")
    digest = sha256_file(csv_path)
    fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(
        f"verification_code: {verification_code(seed)}\n"
        f"file: {csv_path.name}\n"
        f"rows: {len(df)}\n"
        f"sha256: {digest}\n",
        encoding="utf-8",
    )
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    reg_digits, last4 = parse_student_id(args.student_id)
    if int(last4) != args.seed:
        sys.exit(f"Seed {args.seed} must equal the last four registration digits ({last4}).")
    paths = Paths(ROOT, reg_digits)
    digest = write_dataset(args.seed, paths.csv, paths.fingerprint)
    print(f"{verification_code(args.seed)}  wrote {paths.csv}\nSHA-256: {digest}")


if __name__ == "__main__":
    main()
