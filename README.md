# Musanze Harvest Intelligence Service

Student: Bullen Ladu Martin |
Registration number: 24/23862
Institutional username: ug2423862@ines.ac.rw 
Verification code: AI-A2-3862 (seed 3862) 
SWE 3513 Artificial Intelligence, INES-Ruhengeri, Individual Assignment 2 |

## Problem
A Musanze potato cooperative needs a local, evidence-based tool that (1) estimates harvest quantity (regression), (2) flags farms at high production risk, i.e. yield per hectare below the median (classification) and (3) groups farms with similar conditions (clustering). A FastAPI service allows later mobile-app integration. It is a classroom decision-support prototype and **requires human review**; it must not replace an agronomist or decide access to finance, insurance or public services.

## Setup and run
Python 3.12 (3.10+ should work). Tested on Ubuntu 24 (Linux).
```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_all.py --student-id ug2423862@ines.ac.rw --seed 3862
python predict.py --json samples/valid.json
python predict.py --json samples/invalid.json     # prints validation errors, exit code 2
pytest -q
uvicorn app:app --host 127.0.0.1 --port 8000       # docs at http://127.0.0.1:8000/docs
```
`run_all.py` rejects any seed that is not 3862. Optional: `python docs/build_report.py` rebuilds the PDF report from the artifacts (needs `reportlab`, the only extra package, used solely for the report).

## Dataset signature
- Seed `numpy.random.default_rng(3862)`; base rows 240 + (3862 mod 41) = **248**; m = round(0.05 x 248) = 12 missing values in `soil_moisture_pct` and in `fertilizer_kg`; 3 appended duplicates = **251 raw rows**.
- CSV: `data/AI_A2_2423862.csv` (4 decimals). SHA-256: `f3a4210b542054016e2690d417ad4c5a7c89c15c82323fdc2d31503a8f3fc9bd` (also in `artifacts/data_fingerprint.txt`).

## Layout
```
run_all.py  predict.py  app.py  requirements.txt
data/generate_data.py   data/AI_A2_2423862.csv
src/        config, validation, data_prep, modeling, plots, train, inference
models/     regressor / classifier / clusterer pipelines (preprocessing + model) + model_metadata.json
artifacts/  regression_metrics.json, classification_metrics.json, clustering_report.json,
            test_predictions.csv, data_fingerprint.txt, data_validation.json, error_analysis.json,
            plots/residuals.png, confusion_matrix.png, clusters.png
samples/    valid.json, invalid.json        tests/test_system.py
docs/       report (PDF) and build script   evidence/  demo, test log, declaration, AI_USE.md
```

## Results (held-out test set, 62 farms)
- Regression: selected **ridge_regression** (lowest CV RMSE); test MAE 0.867, RMSE 1.117, R2 0.806 (mean baseline RMSE 2.538).
- Classification: selected **logistic_regression** (highest CV F1); test precision 0.939, recall 1.000, F1 0.969, ROC-AUC 1.000 (majority baseline F1 0.000).
- Clustering: K-Means, **K = 2** (silhouette 0.158 - weak structure, reported honestly).

## Design notes
- Imputation, scaling and encoding live inside scikit-learn Pipelines, so they are fit on training data only and reused unchanged at inference.
- `yield_tons`, yield per hectare and `high_risk` are never model inputs (leakage); clustering ignores targets.
- Models are chosen by 5-fold CV on the training split; the test set is only used for reporting.

## API examples
```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/model-info
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"rainfall_mm":1000,"avg_temp_c":18.5,"soil_moisture_pct":55,"fertilizer_kg":120,"field_size_ha":1.5,"season":"A","cooperative_support":1}'
# invalid (rainfall out of range) -> HTTP 422 with a detail message
curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
  -d '{"rainfall_mm":5000,"avg_temp_c":18.5,"soil_moisture_pct":55,"fertilizer_kg":120,"field_size_ha":1.5,"season":"A","cooperative_support":1}'
```

## Known limitations
Synthetic data with hand-coded relationships (scores are optimistic); small sample; inputs must stay within the training ranges; weak clusters; the test-set ROC-AUC of 1.000 comes from only 62 farms; saved `.joblib` files require a compatible scikit-learn version (trained with 1.8.0) - re-run `run_all.py` if loading fails.

## Submission details
- Repository URL: `<<FILL: your repository URL>>`
- Final commit hash: `<<FILL: output of git rev-parse HEAD>>`
- AI assistance is disclosed in `evidence/AI_USE.md`.
