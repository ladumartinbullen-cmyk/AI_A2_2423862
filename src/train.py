"""End-to-end training orchestration: generate -> validate -> clean -> split -> train -> evaluate -> save."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from . import data_prep, plots
from .config import (
    FEATURES,
    RANGES,
    ROOT,
    TARGET_CLF,
    TARGET_REG,
    TEST_SIZE,
    Paths,
    expected_base_rows,
    verification_code,
)
from .modeling import train_classification, train_clustering, train_regression
from .validation import validate_dataframe, validate_seed


def _clean_nan(obj):
    """Replace NaN with None so the JSON files are valid."""
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_nan(v) for v in obj]
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(_clean_nan(obj), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _import_generator(root: Path):
    sys.path.insert(0, str(root / "data"))
    try:
        import importlib

        return importlib.import_module("generate_data")
    finally:
        sys.path.pop(0)


def run_training(student_id: str, seed: int, root: Path = ROOT, verbose: bool = True) -> dict:
    """Run the whole pipeline and write every artifact under `root`."""
    reg_digits = validate_seed(student_id, seed)
    code = verification_code(seed)
    paths = Paths(root, reg_digits)
    paths.make_dirs()
    log = print if verbose else (lambda *a, **k: None)
    log(f"Verification code: {code}   student: {student_id}   seed: {seed}")

    # 1. data
    gen = _import_generator(Path(ROOT))
    digest = gen.write_dataset(seed, paths.csv, paths.fingerprint)
    log(f"Dataset: {paths.csv.name}  SHA-256: {digest}")
    raw = data_prep.load_raw(paths.csv)
    validation = validate_dataframe(raw, seed)
    cleaned, clean_info = data_prep.clean(raw)
    validation.update({"cleaning": clean_info, "verification_code": code, "sha256": digest,
                       "base_rows_rule": f"240 + ({seed} mod 41) = {expected_base_rows(seed)}"})
    _dump(validation, paths.artifacts_dir / "data_validation.json")
    log(f"Rows raw={validation['rows_raw']} dup removed={clean_info['duplicates_removed']} "
        f"missing={validation['missing_values']}")

    # 2. split (fixed, train-only fitting happens inside pipelines)
    X = cleaned[FEATURES]
    y_reg, y_clf = cleaned[TARGET_REG], cleaned[TARGET_CLF]
    X_tr, X_te, yr_tr, yr_te, yc_tr, yc_te = train_test_split(
        X, y_reg, y_clf, test_size=TEST_SIZE, random_state=seed, stratify=y_clf
    )

    # 3. models
    reg_report, reg_fit, reg_sel = train_regression(X_tr, yr_tr, X_te, yr_te, seed)
    clf_report, clf_fit, clf_sel = train_classification(X_tr, yc_tr, X_te, yc_te, seed)
    clu_report, clu_pipe = train_clustering(X_tr, yr_tr, yc_tr, seed)
    split_info = {"random_state": seed, "test_size": TEST_SIZE, "stratified_on": TARGET_CLF,
                  "n_train": int(len(X_tr)), "n_test": int(len(X_te))}
    for rep in (reg_report, clf_report, clu_report):
        rep["verification_code"] = code
        rep["split"] = split_info
    _dump(reg_report, paths.artifacts_dir / "regression_metrics.json")
    _dump(clf_report, paths.artifacts_dir / "classification_metrics.json")
    _dump(clu_report, paths.artifacts_dir / "clustering_report.json")

    # 4. predictions on the held-out test set
    reg_best, clf_best = reg_fit[reg_sel], clf_fit[clf_sel]
    pred_y = reg_best.predict(X_te)
    pred_c = clf_best.predict(X_te)
    prob_c = clf_best.predict_proba(X_te)[:, 1]
    clusters = clu_pipe.predict(X_te)
    preds = pd.DataFrame({
        "row_id": cleaned.loc[X_te.index, "row_id"].to_numpy(),
        "actual_yield_tons": yr_te.to_numpy(),
        "predicted_yield_tons": pred_y,
        "yield_error_tons": pred_y - yr_te.to_numpy(),
        "abs_yield_error_tons": np.abs(pred_y - yr_te.to_numpy()),
        "actual_high_risk": yc_te.to_numpy(),
        "predicted_high_risk": pred_c,
        "high_risk_probability": prob_c,
        "cluster": clusters,
    })
    preds.round(4).to_csv(paths.artifacts_dir / "test_predictions.csv", index=False,
                          float_format="%.4f", lineterminator="\n")

    # 5. error analysis (facts for the written interpretation)
    Xte = X_te.reset_index(drop=True)
    i = int(preds["abs_yield_error_tons"].idxmax())
    wrong = preds[preds["actual_high_risk"] != preds["predicted_high_risk"]].copy()
    wrong["confidence"] = np.where(wrong["predicted_high_risk"] == 1, wrong["high_risk_probability"],
                                   1 - wrong["high_risk_probability"])
    j = int(wrong["confidence"].idxmax()) if len(wrong) else None
    median_ypha = float((cleaned[TARGET_REG] / cleaned["field_size_ha"]).median())
    ana = {
        "noise_std_in_generator": 0.80,
        "largest_regression_error": {
            "row_id": int(preds.loc[i, "row_id"]),
            "features": Xte.loc[i].to_dict(),
            "actual_yield_tons": round(float(preds.loc[i, "actual_yield_tons"]), 4),
            "predicted_yield_tons": round(float(preds.loc[i, "predicted_yield_tons"]), 4),
            "error_tons": round(float(preds.loc[i, "yield_error_tons"]), 4),
            "temperature_distance_from_18c": round(abs(float(Xte.loc[i, "avg_temp_c"]) - 18), 3),
            "error_in_noise_sds": round(abs(float(preds.loc[i, "yield_error_tons"])) / 0.80, 2),
        },
        "median_yield_per_ha_cleaned_data": round(median_ypha, 4),
        "misclassified_test_farms": int(len(wrong)),
    }
    if j is not None:
        ypha = float(preds.loc[j, "actual_yield_tons"]) / float(Xte.loc[j, "field_size_ha"])
        ana["most_confident_misclassification"] = {
            "row_id": int(preds.loc[j, "row_id"]),
            "features": Xte.loc[j].to_dict(),
            "actual_high_risk": int(preds.loc[j, "actual_high_risk"]),
            "predicted_high_risk": int(preds.loc[j, "predicted_high_risk"]),
            "predicted_probability": round(float(preds.loc[j, "high_risk_probability"]), 4),
            "actual_yield_per_ha": round(ypha, 4),
            "distance_to_median_ypha": round(abs(ypha - median_ypha), 4),
        }
    _dump(ana, paths.artifacts_dir / "error_analysis.json")

    # 6. plots
    plots.plot_residuals(yr_te, pred_y, paths.plots_dir / "residuals.png",
                         f"{reg_sel} residuals on held-out test farms ({code})")
    plots.plot_confusion(clf_report["models"][clf_sel]["test"]["confusion_matrix"],
                         paths.plots_dir / "confusion_matrix.png", f"{clf_sel} ({code})")
    Z_tr = clu_pipe.named_steps["prep"].transform(X_tr)
    plots.plot_clusters(Z_tr, clu_pipe.named_steps["kmeans"].labels_,
                        clu_pipe.named_steps["kmeans"].cluster_centers_,
                        paths.plots_dir / "clusters.png",
                        f"K={clu_report['selected_k']} clusters, PCA view ({code})")

    # 7. save models + metadata
    joblib.dump(reg_best, paths.models_dir / "regressor.joblib")
    joblib.dump(clf_best, paths.models_dir / "classifier.joblib")
    joblib.dump(clu_pipe, paths.models_dir / "clusterer.joblib")
    meta = {
        "verification_code": code,
        "student_id": student_id,
        "registration_digits": reg_digits,
        "seed": seed,
        "dataset_file": paths.csv.name,
        "dataset_sha256": digest,
        "features": FEATURES,
        "ranges": {k: list(v) for k, v in RANGES.items()},
        "selected_regression_model": reg_sel,
        "selected_classification_model": clf_sel,
        "selected_k": clu_report["selected_k"],
        "cluster_profiles": [p["plain_language_profile"] for p in clu_report["profiles"]],
        "sklearn_version_trained": sklearn.__version__,
        "synthetic_data": True,
        "intended_use": "Classroom decision-support prototype. Not a replacement for an agronomist.",
    }
    _dump(meta, paths.models_dir / "model_metadata.json")

    log(f"Regression selected: {reg_sel}  test RMSE={reg_report['models'][reg_sel]['test']['rmse']:.3f}")
    log(f"Classification selected: {clf_sel}  test F1={clf_report['models'][clf_sel]['test']['f1']:.3f}")
    log(f"Clustering selected K={clu_report['selected_k']}")
    return {"verification_code": code, "sha256": digest, "regression": reg_report,
            "classification": clf_report, "clustering": clu_report, "paths": paths}
