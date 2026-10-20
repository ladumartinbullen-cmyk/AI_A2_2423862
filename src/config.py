"""Central configuration for the Musanze Harvest Intelligence Service."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

NUMERIC_FEATURES = [
    "rainfall_mm",
    "avg_temp_c",
    "soil_moisture_pct",
    "fertilizer_kg",
    "field_size_ha",
    "cooperative_support",
]
CATEGORICAL_FEATURES = ["season"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

TARGET_REG = "yield_tons"
TARGET_CLF = "high_risk"
REQUIRED_COLUMNS = FEATURES + [TARGET_REG, TARGET_CLF]

# Columns that must never be used as model inputs (target leakage).
FORBIDDEN_FEATURES = {"yield_tons", "yield_per_ha", "yield_per_hectare", "high_risk"}

# Valid ranges = generator ranges. Inference rejects values outside them.
RANGES = {
    "rainfall_mm": (600.0, 1400.0),
    "avg_temp_c": (12.0, 26.0),
    "soil_moisture_pct": (20.0, 80.0),
    "fertilizer_kg": (0.0, 250.0),
    "field_size_ha": (0.2, 4.0),
    "cooperative_support": (0, 1),
}
SEASONS = ("A", "B")
MIN_YIELD = 0.5
TEST_SIZE = 0.25
N_SPLITS_CV = 5
K_RANGE = range(2, 7)
N_DUPLICATES = 3


def parse_student_id(student_id: str) -> tuple[str, str]:
    """Return (registration digits, last four digits) from a username like ug2423862@ines.ac.rw."""
    local = student_id.split("@")[0]
    digits = re.sub(r"\D", "", local)
    if len(digits) < 4:
        raise ValueError(
            f"Student id '{student_id}' must contain at least four digits (e.g. ug2423862@ines.ac.rw)."
        )
    return digits, digits[-4:]


def verification_code(seed: int) -> str:
    """AI-A2-LLLL verification code."""
    return f"AI-A2-{int(seed):04d}"


def expected_base_rows(seed: int) -> int:
    """Base rows = 240 + (LLLL mod 41)."""
    return 240 + int(seed) % 41


def expected_missing(seed: int) -> int:
    """m = round(0.05 x base rows)."""
    return int(round(0.05 * expected_base_rows(seed)))


class Paths:
    """All output locations for one run, rooted at `root`."""

    def __init__(self, root: Path = ROOT, reg_digits: str = "2423862"):
        self.root = Path(root)
        self.reg_digits = reg_digits
        self.data_dir = self.root / "data"
        self.models_dir = self.root / "models"
        self.artifacts_dir = self.root / "artifacts"
        self.plots_dir = self.artifacts_dir / "plots"
        self.csv = self.data_dir / f"AI_A2_{reg_digits}.csv"
        self.fingerprint = self.artifacts_dir / "data_fingerprint.txt"

    def make_dirs(self) -> None:
        for d in (self.data_dir, self.models_dir, self.artifacts_dir, self.plots_dir):
            d.mkdir(parents=True, exist_ok=True)
