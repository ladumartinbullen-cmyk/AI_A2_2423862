"""Automated tests: happy path, failure paths, reproducibility and one API response."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "data"))

import generate_data  # noqa: E402
from src import inference  # noqa: E402
from src.config import FEATURES, FORBIDDEN_FEATURES, expected_base_rows, expected_missing  # noqa: E402
from src.train import run_training  # noqa: E402
from src.validation import RecordValidationError, validate_record, validate_seed  # noqa: E402

STUDENT_ID = "ug2423862@ines.ac.rw"
SEED = 3862
VALID = {
    "rainfall_mm": 1000, "avg_temp_c": 18.5, "soil_moisture_pct": 55, "fertilizer_kg": 120,
    "field_size_ha": 1.5, "season": "A", "cooperative_support": 1,
}


@pytest.fixture(scope="session", autouse=True)
def trained_models():
    """Make sure saved models exist (train once if they do not)."""
    if not (ROOT / "models" / "regressor.joblib").exists():
        run_training(STUDENT_ID, SEED, root=ROOT, verbose=False)


def test_valid_inference_returns_all_fields():
    out = inference.predict_record(dict(VALID))
    assert out["verification_code"] == "AI-A2-3862"
    assert out["predicted_yield_tons"] > 0
    assert out["high_risk_prediction"] in (0, 1)
    assert 0.0 <= out["high_risk_probability"] <= 1.0
    assert isinstance(out["cluster"], int)
    assert out["human_review_required"] is True


def test_missing_field_is_rejected():
    bad = dict(VALID)
    del bad["field_size_ha"]
    with pytest.raises(RecordValidationError) as exc:
        inference.predict_record(bad)
    assert any("field_size_ha" in e for e in exc.value.errors)


def test_wrong_type_is_rejected():
    bad = dict(VALID, rainfall_mm="lots")
    with pytest.raises(RecordValidationError) as exc:
        validate_record(bad)
    assert any("rainfall_mm" in e for e in exc.value.errors)
    with pytest.raises(RecordValidationError):
        validate_record(dict(VALID, cooperative_support=True))


def test_range_error_is_rejected():
    with pytest.raises(RecordValidationError) as exc:
        validate_record(dict(VALID, avg_temp_c=40))
    assert any("outside the allowed range" in e for e in exc.value.errors)
    with pytest.raises(RecordValidationError):
        validate_record(dict(VALID, season="C"))


def test_wrong_seed_is_rejected():
    assert validate_seed(STUDENT_ID, 3862) == "2423862"
    with pytest.raises(ValueError):
        validate_seed(STUDENT_ID, 4671)


def test_dataset_signature_rules():
    df = generate_data.generate_dataframe(SEED)
    assert expected_base_rows(SEED) == 248
    assert len(df) == 248 + 3
    assert df.duplicated().sum() == 3
    assert df["soil_moisture_pct"].isna().sum() >= expected_missing(SEED)
    assert df["yield_tons"].min() >= 0.5


def test_no_target_leakage_in_features():
    assert not (FORBIDDEN_FEATURES & set(FEATURES))


def test_reproducibility_same_fingerprint_and_metrics(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    ra = run_training(STUDENT_ID, SEED, root=a, verbose=False)
    rb = run_training(STUDENT_ID, SEED, root=b, verbose=False)
    assert ra["sha256"] == rb["sha256"]
    for name in ("regression_metrics.json", "classification_metrics.json", "clustering_report.json"):
        assert json.loads((a / "artifacts" / name).read_text()) == json.loads((b / "artifacts" / name).read_text())


def test_api_health_and_predict_and_validation_error():
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from app import app

    client = TestClient(app)
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/model-info").json()["verification_code"] == "AI-A2-3862"
    ok = client.post("/predict", json=VALID)
    assert ok.status_code == 200 and ok.json()["verification_code"] == "AI-A2-3862"
    bad = client.post("/predict", json=dict(VALID, rainfall_mm=9999))
    assert bad.status_code == 422
