"""Shared inference logic used by predict.py and app.py."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from .config import FEATURES, ROOT
from .validation import validate_record

MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = ROOT / "artifacts"
_CACHE: dict[str, dict[str, Any]] = {}


class ArtifactsMissingError(FileNotFoundError):
    """Raised when saved models are not found (run run_all.py first)."""


def load_artifacts(models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Load saved pipelines and metadata (cached)."""
    key = str(models_dir)
    if key in _CACHE:
        return _CACHE[key]
    needed = ["regressor.joblib", "classifier.joblib", "clusterer.joblib", "model_metadata.json"]
    missing = [n for n in needed if not (Path(models_dir) / n).exists()]
    if missing:
        raise ArtifactsMissingError(f"Missing model files {missing} in {models_dir}. Run: python run_all.py ...")
    arts = {
        "regressor": joblib.load(Path(models_dir) / "regressor.joblib"),
        "classifier": joblib.load(Path(models_dir) / "classifier.joblib"),
        "clusterer": joblib.load(Path(models_dir) / "clusterer.joblib"),
        "metadata": json.loads((Path(models_dir) / "model_metadata.json").read_text(encoding="utf-8")),
    }
    _CACHE[key] = arts
    return arts


def predict_record(record: Any, models_dir: Path = MODELS_DIR) -> dict[str, Any]:
    """Validate one farm record and return yield, risk, probability, cluster and verification code."""
    clean = validate_record(record)  # raises RecordValidationError
    arts = load_artifacts(models_dir)
    X = pd.DataFrame([clean], columns=FEATURES)
    proba = float(arts["classifier"].predict_proba(X)[0, 1])
    risk = int(arts["classifier"].predict(X)[0])
    meta = arts["metadata"]
    cluster = int(arts["clusterer"].predict(X)[0])
    return {
        "predicted_yield_tons": round(float(arts["regressor"].predict(X)[0]), 4),
        "high_risk_prediction": risk,
        "high_risk_label": "high production risk" if risk == 1 else "not high production risk",
        "high_risk_probability": round(proba, 4),
        "cluster": cluster,
        "cluster_profile": meta["cluster_profiles"][cluster],
        "verification_code": meta["verification_code"],
        "human_review_required": True,
        "notice": "Synthetic-data classroom prototype. A field officer or agronomist must review this estimate.",
    }


def model_info(models_dir: Path = MODELS_DIR, artifacts_dir: Path = ARTIFACTS_DIR) -> dict[str, Any]:
    """Metadata plus headline metrics for GET /model-info."""
    meta = dict(load_artifacts(models_dir)["metadata"])
    try:
        reg = json.loads((Path(artifacts_dir) / "regression_metrics.json").read_text(encoding="utf-8"))
        clf = json.loads((Path(artifacts_dir) / "classification_metrics.json").read_text(encoding="utf-8"))
        meta["regression_test_metrics"] = reg["models"][reg["selected_model"]]["test"]
        meta["classification_test_metrics"] = {
            k: v for k, v in clf["models"][clf["selected_model"]]["test"].items() if k != "confusion_matrix"
        }
    except (FileNotFoundError, KeyError):
        pass
    return meta
