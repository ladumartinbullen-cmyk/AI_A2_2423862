"""FastAPI service: uvicorn app:app --host 127.0.0.1 --port 8000"""
from __future__ import annotations

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src import inference
from src.config import RANGES
from src.validation import RecordValidationError

app = FastAPI(
    title="Musanze Harvest Intelligence Service",
    version="1.0.0",
    description="Synthetic-data classroom prototype. Outputs need human review.",
)


class FarmRecord(BaseModel):
    """One farm record. Values outside the training ranges are rejected with HTTP 422."""

    model_config = ConfigDict(extra="forbid")

    rainfall_mm: float = Field(..., ge=RANGES["rainfall_mm"][0], le=RANGES["rainfall_mm"][1])
    avg_temp_c: float = Field(..., ge=RANGES["avg_temp_c"][0], le=RANGES["avg_temp_c"][1])
    soil_moisture_pct: float = Field(..., ge=RANGES["soil_moisture_pct"][0], le=RANGES["soil_moisture_pct"][1])
    fertilizer_kg: float = Field(..., ge=RANGES["fertilizer_kg"][0], le=RANGES["fertilizer_kg"][1])
    field_size_ha: float = Field(..., ge=RANGES["field_size_ha"][0], le=RANGES["field_size_ha"][1])
    season: Literal["A", "B"]
    cooperative_support: Literal[0, 1]


class Prediction(BaseModel):
    predicted_yield_tons: float
    high_risk_prediction: int
    high_risk_label: str
    high_risk_probability: float
    cluster: int
    cluster_profile: str
    verification_code: str
    human_review_required: bool
    notice: str


@app.get("/health")
def health() -> dict:
    """Liveness check; reports whether models are loadable."""
    try:
        meta = inference.load_artifacts()["metadata"]
        return {"status": "ok", "verification_code": meta["verification_code"]}
    except inference.ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.get("/model-info")
def model_info() -> dict:
    """Selected models, verification code, dataset fingerprint and test metrics."""
    try:
        return inference.model_info()
    except inference.ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@app.post("/predict", response_model=Prediction)
def predict(record: FarmRecord) -> dict:
    """Predict yield, risk, probability and cluster for one farm."""
    try:
        return inference.predict_record(record.model_dump())
    except RecordValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors)
    except inference.ArtifactsMissingError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
