"""Pipelines, candidate models, metrics and model selection for the three ML tasks."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    make_scorer,
    silhouette_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    CATEGORICAL_FEATURES,
    FEATURES,
    FORBIDDEN_FEATURES,
    K_RANGE,
    N_SPLITS_CV,
    NUMERIC_FEATURES,
)

assert not (FORBIDDEN_FEATURES & set(FEATURES)), "target leakage in feature list"


def build_preprocessor() -> ColumnTransformer:
    """Impute + scale numeric columns, impute + one-hot encode categorical columns."""
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)]
    )


def make_pipeline(estimator) -> Pipeline:
    """Preprocessing and model in one object so training and inference share transformations."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def regression_candidates(seed: int) -> dict[str, Any]:
    return {
        "mean_baseline": DummyRegressor(strategy="mean"),
        "ridge_regression": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def classification_candidates(seed: int) -> dict[str, Any]:
    return {
        "majority_baseline": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=seed, n_jobs=1
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=seed),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "r2": float(r2_score(y_true, y_pred)),
    }


def classification_metrics(y_true, y_pred, proba) -> dict[str, Any]:
    out: dict[str, Any] = {
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, proba))
    else:
        out["roc_auc"] = None  # not defined with a single class in y_true
    return out


def train_regression(X_train, y_train, X_test, y_test, seed: int):
    """Compare baseline + candidates (CV on train, score on held-out test). Returns (report, fitted dict, selected)."""
    cv = KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=seed)
    scoring = {"rmse": "neg_root_mean_squared_error", "mae": "neg_mean_absolute_error", "r2": "r2"}
    report: dict[str, Any] = {"models": {}}
    fitted: dict[str, Pipeline] = {}
    for name, est in regression_candidates(seed).items():
        pipe = make_pipeline(est)
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        report["models"][name] = {
            "is_baseline": name == "mean_baseline",
            "cv_rmse_mean": float(-cvres["test_rmse"].mean()),
            "cv_mae_mean": float(-cvres["test_mae"].mean()),
            "cv_r2_mean": float(cvres["test_r2"].mean()),
            "test": regression_metrics(y_test, pipe.predict(X_test)),
        }
    candidates = {k: v for k, v in report["models"].items() if not v["is_baseline"]}
    selected = min(candidates, key=lambda k: candidates[k]["cv_rmse_mean"])
    base = report["models"]["mean_baseline"]
    sel = report["models"][selected]
    report["selected_model"] = selected
    report["selection_reason"] = (
        f"{selected} has the lowest 5-fold cross-validated RMSE on the training split "
        f"({sel['cv_rmse_mean']:.3f} t vs baseline {base['cv_rmse_mean']:.3f} t). "
        f"The test set was not used for selection; its held-out RMSE is {sel['test']['rmse']:.3f} t "
        f"and R2 is {sel['test']['r2']:.3f}."
    )
    return report, fitted, selected


def train_classification(X_train, y_train, X_test, y_test, seed: int):
    """Compare majority baseline + candidates; select by CV F1 on the training split."""
    cv = StratifiedKFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=seed)
    scoring = {"f1": make_scorer(f1_score, zero_division=0), "precision": make_scorer(precision_score, zero_division=0), "recall": make_scorer(recall_score, zero_division=0)}
    report: dict[str, Any] = {"positive_class": "high_risk = 1 (yield per hectare below the median)", "models": {}}
    fitted: dict[str, Pipeline] = {}
    for name, est in classification_candidates(seed).items():
        pipe = make_pipeline(est)
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        report["models"][name] = {
            "is_baseline": name == "majority_baseline",
            "cv_f1_mean": float(cvres["test_f1"].mean()),
            "test": classification_metrics(y_test, pred, proba),
        }
    candidates = {k: v for k, v in report["models"].items() if not v["is_baseline"]}
    selected = max(candidates, key=lambda k: candidates[k]["cv_f1_mean"])
    sel = report["models"][selected]
    report["selected_model"] = selected
    report["selection_reason"] = (
        f"{selected} has the highest 5-fold cross-validated F1 on the training split "
        f"({sel['cv_f1_mean']:.3f}). Held-out test F1 is {sel['test']['f1']:.3f}, precision "
        f"{sel['test']['precision']:.3f}, recall {sel['test']['recall']:.3f}, ROC-AUC "
        f"{sel['test']['roc_auc']:.3f}. The test set was not used for selection."
    )
    return report, fitted, selected


_PHRASES = {
    "rainfall_mm": ("higher rainfall", "lower rainfall"),
    "avg_temp_c": ("warmer conditions", "cooler conditions"),
    "soil_moisture_pct": ("wetter soil", "drier soil"),
    "fertilizer_kg": ("heavier fertilizer use", "lighter fertilizer use"),
    "field_size_ha": ("larger fields", "smaller fields"),
    "cooperative_support": ("mostly supported by the cooperative", "mostly without cooperative support"),
    "season_A_share": ("mostly season A", "mostly season B"),
}


def train_clustering(X_train: pd.DataFrame, y_train: pd.Series, risk_train: pd.Series, seed: int):
    """Scale features (no targets), test K=2..6, choose K by silhouette, build plain-language profiles."""
    prep = build_preprocessor()
    Z = prep.fit_transform(X_train)
    scores: dict[str, float] = {}
    models: dict[int, KMeans] = {}
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=seed)
        labels = km.fit_predict(Z)
        scores[str(k)] = float(silhouette_score(Z, labels))
        models[k] = km
    best_k = int(max(scores, key=lambda k: scores[k]))
    pipe = Pipeline([("prep", prep), ("kmeans", models[best_k])])
    labels = models[best_k].labels_

    raw = X_train.copy()
    raw["season_A_share"] = (raw.pop("season") == "A").astype(float)
    overall_mean, overall_std = raw.mean(), raw.std(ddof=0)
    profiles = []
    for c in range(best_k):
        mask = labels == c
        means = raw[mask].mean()
        z = ((means - overall_mean) / overall_std)
        strong = z.reindex(z.abs().sort_values(ascending=False).index)
        descriptors = [
            _PHRASES[f][0 if v > 0 else 1] for f, v in strong.items() if abs(v) >= 0.25
        ][:3]
        text = "; ".join(descriptors) if descriptors else "close to the average farm on every feature"
        profiles.append(
            {
                "cluster": c,
                "size_train": int(mask.sum()),
                "feature_means": {k: round(float(v), 4) for k, v in means.items()},
                "feature_z_vs_overall": {k: round(float(v), 3) for k, v in z.items()},
                "plain_language_profile": f"Cluster {c}: {text}.",
                "posthoc_mean_yield_tons": round(float(y_train[mask].mean()), 4),
                "posthoc_high_risk_share": round(float(risk_train[mask].mean()), 4),
            }
        )
    sil = scores[str(best_k)]
    strength = "weak" if sil < 0.25 else "moderate" if sil < 0.5 else "strong"
    report = {
        "silhouette_scores": scores,
        "selected_k": best_k,
        "selection_reason": f"K={best_k} has the highest silhouette score ({sil:.3f}).",
        "scaling": "median imputation + StandardScaler + one-hot season, fitted on the training split only; targets are not used",
        "profiles": profiles,
        "interpretation": (
            f"The silhouette of {sil:.3f} indicates {strength} cluster structure. The farm features are "
            "independent random draws, so the clusters are regions of a continuous space rather than "
            "naturally separate groups; use them as a descriptive grouping for field-officer visits, "
            "not as proof of distinct farm types. Yield and risk shares are shown after clustering "
            "(post-hoc) only to interpret clusters; they were never inputs."
        ),
    }
    return report, pipe
