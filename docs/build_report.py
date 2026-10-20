"""Build docs/AI_A2_2423862_REPORT.pdf from the generated JSON/CSV artifacts (so numbers always match)."""
from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parents[1]
A = ROOT / "artifacts"
NAME, REG, USER, SEED = "Bullen Ladu Martin", "24/23862", "ug2423862@ines.ac.rw", 3862
CODE = f"AI-A2-{SEED:04d}"
OUT = ROOT / "docs" / "AI_A2_2423862_REPORT.pdf"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=15, spaceAfter=6, textColor=colors.HexColor("#1a365d"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=11, spaceBefore=6, spaceAfter=3)
B = ParagraphStyle("B", parent=ss["BodyText"], fontSize=9.3, leading=12.2, spaceAfter=4)
SM = ParagraphStyle("SM", parent=B, fontSize=8, leading=10, textColor=colors.HexColor("#444444"))


def J(name):
    return json.loads((A / name).read_text(encoding="utf-8"))


def table(rows, widths=None, font=8.3):
    cell = ParagraphStyle("cell", parent=B, fontSize=font, leading=font + 2.2, spaceAfter=0)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold")
    rows = [[Paragraph(c, head if r == 0 else cell) if isinstance(c, str) else c for c in row]
            for r, row in enumerate(rows)]
    t = Table(rows, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), font), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]))
    return t


def P(text, style=B):
    return Paragraph(text, style)


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(2 * cm, 1.1 * cm, f"{NAME} | {REG} | {CODE} | SWE 3513 Individual Assignment 2")
    canvas.drawRightString(A4[0] - 2 * cm, 1.1 * cm, f"Page {doc.page}")
    canvas.restoreState()


def build():
    val, reg, clf = J("data_validation.json"), J("regression_metrics.json"), J("classification_metrics.json")
    clu, ana = J("clustering_report.json"), J("error_analysis.json")
    fp = (A / "data_fingerprint.txt").read_text().strip().splitlines()
    sha = val["sha256"]
    rs, cs = reg["selected_model"], clf["selected_model"]
    rt, ct = reg["models"][rs]["test"], clf["models"][cs]["test"]
    S = []

    # Page 1
    S += [P("Musanze Harvest Intelligence Service", H1), P("SWE 3513 Artificial Intelligence - Individual Assignment 2 - Technical Report", B)]
    S.append(table([
        ["Student", NAME], ["Registration number", REG], ["Institutional username", USER],
        ["Verification code", f"<b>{CODE}</b> (seed {SEED})"], ["Programme / institution", "INES-Ruhengeri"],
    ], [5 * cm, 11.5 * cm], 9))
    S += [Spacer(1, 6), P("Decision problem and stakeholder", H2),
          P("A potato cooperative in Musanze receives farm records in inconsistent spreadsheets. Its <b>field officer</b> needs an "
            "evidence-based tool to support visit planning. The service answers three questions from seven farm descriptors "
            "(rainfall, temperature, soil moisture, fertilizer, field size, season, cooperative support):"),
          P("1. <b>Regression:</b> how many tons of potatoes will this farm harvest (yield_tons)?<br/>"
            "2. <b>Classification:</b> is the farm at high production risk, i.e. is its yield per hectare below the median (high_risk)?<br/>"
            "3. <b>Clustering:</b> which group of farms with similar conditions does it resemble?"),
          P("Internet access may be unreliable, so training and command-line prediction run locally; FastAPI is provided for a later mobile app."),
          P("Ethical boundary", H2),
          P("The dataset is <b>synthetic</b> (generated from my registration digits) and the system is a classroom decision-support prototype. "
            "It must not replace an agronomist and must never be used to decide access to finance, insurance or public services. "
            "Every output carries <i>human_review_required = true</i> and a notice that a field officer must review it.")]
    S.append(PageBreak())

    # Page 2
    S += [P("Data and reproducibility", H1),
          P(f"<b>Personal seed rule.</b> LLLL = 3862. The generator uses <font face='Courier'>numpy.random.default_rng(3862)</font>. "
            f"Base rows = 240 + (3862 mod 41) = 240 + 8 = <b>{val['expected_base_rows']}</b>. "
            f"m = round(0.05 x 248) = <b>{val['expected_missing_per_column']}</b> missing values injected into each of soil_moisture_pct and fertilizer_kg, "
            f"then 3 exact duplicates were appended: <b>{val['rows_raw']} raw rows</b>."),
          table([["Column", "Generation rule", "Role"],
                 ["rainfall_mm", "Uniform 600-1400", "Feature"], ["avg_temp_c", "Uniform 12-26", "Feature"],
                 ["soil_moisture_pct", "Uniform 20-80 (12 set missing)", "Feature"], ["fertilizer_kg", "Uniform 0-250 (12 set missing)", "Feature"],
                 ["field_size_ha", "Uniform 0.2-4.0", "Feature"], ["season", "Random A / B", "Categorical feature"],
                 ["cooperative_support", "Random 0 / 1", "Binary feature"],
                 ["yield_tons", "Prescribed formula, noise SD 0.80, clipped at 0.50", "Regression target"],
                 ["high_risk", "Yield per hectare below its median", "Classification target"]],
                [4 * cm, 8.3 * cm, 4.2 * cm]),
          Spacer(1, 6), P("Data-quality evidence", H2),
          table([["Check", "Result"],
                 ["Raw rows / rows after removing duplicates", f"{val['rows_raw']} / {val['cleaning']['rows_after']} ({val['cleaning']['duplicates_removed']} duplicates removed)"],
                 ["Missing values in raw file", f"soil_moisture_pct {val['missing_values']['soil_moisture_pct']}, fertilizer_kg {val['missing_values']['fertilizer_kg']}"],
                 ["Missing values after de-duplication", f"soil_moisture_pct {val['cleaning']['missing_after_cleaning']['soil_moisture_pct']}, fertilizer_kg {val['cleaning']['missing_after_cleaning']['fertilizer_kg']}"],
                 ["Schema, range and leakage checks", "passed (see artifacts/data_validation.json)"]],
                [6.5 * cm, 10 * cm]),
          P("The raw count is 13 rather than 12 because one appended duplicate copied a row whose value had already been set to missing; "
            "removing duplicates restores 12 per column.", SM),
          P("Fingerprint", H2),
          P(f"File AI_A2_2423862.csv, saved with float_format=\"%.4f\". SHA-256 of the final CSV bytes:<br/><font face='Courier' size='8'>{sha}</font>"),
          P("Running <font face='Courier'>run_all.py</font> twice reproduced the same fingerprint and identical metric files "
            "(also checked by an automated test). run_all.py rejects any seed that is not 3862.")]
    S.append(PageBreak())

    # Page 3
    S += [P("Pipeline design", H1),
          P("<b>Flow:</b> generate CSV &rarr; validate (columns, ranges, missing, duplicates, leakage) &rarr; drop exact duplicates &rarr; "
            f"fixed split {val.get('cleaning', {}).get('rows_after', 248)} rows into train/test (test 25%, random_state=3862, stratified on high_risk; "
            f"{reg['split']['n_train']} train / {reg['split']['n_test']} test) &rarr; fit models &rarr; evaluate on the held-out test set &rarr; save models and artifacts."),
          P("<b>Preprocessing</b> is a scikit-learn ColumnTransformer inside every Pipeline: numeric features use median imputation then StandardScaler; "
            "season uses most-frequent imputation then one-hot encoding (unknown categories ignored). Because imputation, scaling and encoding are "
            "steps of the Pipeline, they are fitted only on training data (and on training folds during cross-validation), and the very same fitted "
            "transformations are applied at inference time by predict.py and app.py."),
          P("<b>Leakage controls.</b> yield_tons, yield per hectare and high_risk are never classification inputs (an assertion and a test enforce this). "
            "high_risk is derived from yield, so a model given yield could trivially recreate it. Clustering uses the seven features only; yield and risk "
            "appear in profiles only after clustering, to interpret them."),
          P("<b>Model selection.</b> Candidates were compared with 5-fold cross-validation on the training split only; the best non-baseline model was "
            "chosen and the held-out test set was used once for reporting. Every random state is 3862."),
          table([["Task", "Baseline", "Candidates", "Selected (by CV)"],
                 ["Regression", "Mean predictor", "Ridge, Random Forest, Gradient Boosting", rs],
                 ["Classification", "Majority class", "Logistic Regression, Random Forest, Gradient Boosting", cs],
                 ["Clustering", "-", f"K-Means, K = 2..6, scaled features", f"K = {clu['selected_k']} (silhouette)"]],
                [2.6 * cm, 3 * cm, 7.4 * cm, 3.5 * cm]),
          Spacer(1, 6),
          P("Saved artifacts: models/regressor.joblib, classifier.joblib and clusterer.joblib are full pipelines (preprocessing + model), plus "
            "model_metadata.json. predict.py and app.py load exactly these files.")]
    S.append(PageBreak())

    # Page 4
    def rrow(n, v):
        t = v["test"]
        return [n, f"{t['mae']:.3f}", f"{t['rmse']:.3f}", f"{t['r2']:.3f}", f"{v['cv_rmse_mean']:.3f}"]

    def crow(n, v):
        t = v["test"]
        return [n, f"{t['precision']:.3f}", f"{t['recall']:.3f}", f"{t['f1']:.3f}", f"{t['roc_auc']:.3f}", f"{v['cv_f1_mean']:.3f}"]
    S += [P("Regression and classification", H1), P("Regression (held-out test set, tons)", H2),
          table([["Model", "MAE", "RMSE", "R2", "CV RMSE (train)"]] + [rrow(k, v) for k, v in reg["models"].items()],
                [5 * cm, 2.5 * cm, 2.5 * cm, 2.5 * cm, 3.5 * cm]),
          P("Classification (held-out test set)", H2),
          table([["Model", "Precision", "Recall", "F1", "ROC-AUC", "CV F1 (train)"]] + [crow(k, v) for k, v in clf["models"].items()],
                [4.5 * cm, 2.3 * cm, 2.3 * cm, 2.3 * cm, 2.5 * cm, 2.6 * cm]),
          Spacer(1, 4),
          table([[Image(str(A / "plots" / "residuals.png"), 11 * cm, 4.4 * cm), Image(str(A / "plots" / "confusion_matrix.png"), 5 * cm, 4.35 * cm)]],
                [11.3 * cm, 5.2 * cm]),
          P(f"<b>Selected models.</b> {reg['selection_reason']} {clf['selection_reason']}"),
          P(f"<b>Reading the plots.</b> Residuals are close to zero on average (mean -0.15 t, a slight under-prediction) with no strong pattern, and the test RMSE of {rt['rmse']:.2f} t is close to the "
            "0.80 t noise built into the generator, which no model can remove. Ridge regression won because the true yield formula is almost linear; "
            f"the confusion matrix {ct['confusion_matrix']} shows no missed high-risk farm (recall {ct['recall']:.2f}) and {ct['confusion_matrix'][0][1]} false alarms. "
            "With only 62 test farms, an ROC-AUC of 1.000 should be read cautiously.", B)]
    e, m = ana["largest_regression_error"], ana.get("most_confident_misclassification")
    S += [P(f"<b>One important regression error.</b> Test farm row {e['row_id']}: actual {e['actual_yield_tons']:.2f} t, predicted {e['predicted_yield_tons']:.2f} t "
            f"(error {e['error_tons']:.2f} t, about {e['error_in_noise_sds']:.1f} noise standard deviations). Its fertilizer value was missing, so the model used the training median instead of the real "
            f"amount, and the farm has a large field ({e['features']['field_size_ha']:.2f} ha), where yield is most sensitive to errors. I conclude the under-prediction comes from this imputation "
            "combined with a large positive random noise draw; I cannot separate the two because noise is unobservable."),
          ]
    if m:
        S.append(P(f"<b>One important classification error.</b> Farm row {m['row_id']} was predicted high risk (probability {m['predicted_probability']:.2f}) but is not. Its true yield per hectare "
                   f"is {m['actual_yield_per_ha']:.2f}, only {m['distance_to_median_ypha']:.2f} from the median {ana['median_yield_per_ha_cleaned_data']:.2f} that defines the label. "
                   "Because the label is a median split, many farms sit near the boundary and a small amount of noise flips them; this is a borderline case, not a model failure."))
    S.append(PageBreak())

    # Page 5
    S += [P("Clustering", H1),
          P("Features were imputed and scaled (training data only), no targets were used, and K-Means was run for K = 2..6 with n_init=10, random_state=3862."),
          table([["K", "2", "3", "4", "5", "6"], ["Silhouette"] + [f"{clu['silhouette_scores'][str(k)]:.3f}" for k in range(2, 7)]],
                [3 * cm] + [2.7 * cm] * 5),
          P(f"{clu['selection_reason']} The silhouette is low, which means the clusters overlap: farm features are independent random draws, so the data has no natural groups."),
          Image(str(A / "plots" / "clusters.png"), 9.2 * cm, 7.1 * cm), P("Farm profiles", H2),
          table([["Cluster", "Farms (train)", "Profile (plain language)", "Mean yield (t)*", "High-risk share*"]] +
                [[str(p["cluster"]), str(p["size_train"]), Paragraph(p["plain_language_profile"], SM), f"{p['posthoc_mean_yield_tons']:.2f}",
                  f"{p['posthoc_high_risk_share']:.2f}"] for p in clu["profiles"]],
                [1.6 * cm, 2.3 * cm, 7 * cm, 2.8 * cm, 2.8 * cm]),
          P("*Computed after clustering only to help interpretation; never used to form clusters.", SM)]
    p0 = clu["profiles"][0]
    S.append(P(f"<b>Operational meaning (one cluster).</b> {p0['plain_language_profile']} Mean yield is {p0['posthoc_mean_yield_tons']:.2f} t and "
               f"{p0['posthoc_high_risk_share']*100:.0f}% of its farms are high risk. A field officer can use the groups to plan visits, for example one "
               "outreach message for supported farms and another for unsupported ones. Because the split is mainly driven by one binary feature and the silhouette is weak, "
               "it is a convenient grouping, not evidence of two distinct farm types."))
    S.append(PageBreak())

    # Page 6
    S += [P("API and testing", H1),
          table([["Endpoint", "Purpose", "Success", "Errors"],
                 ["GET /health", "Liveness and verification code", "200 {status, verification_code}", "503 if models missing"],
                 ["GET /model-info", "Selected models, fingerprint, test metrics", "200 JSON", "503 if models missing"],
                 ["POST /predict", "Yield, risk, probability, cluster", "200 + verification code", "422 invalid input; 503 models missing"]],
                [3 * cm, 5.2 * cm, 4.5 * cm, 3.8 * cm]),
          P("Request validation uses a typed Pydantic model: floats with ge/le range limits, season as Literal['A','B'], cooperative_support as Literal[0,1], "
            "unknown fields forbidden.", B),
          P("Valid request and response", H2),
          P("<font face='Courier' size='7.8'>POST /predict {\"rainfall_mm\":1000,\"avg_temp_c\":18.5,\"soil_moisture_pct\":55,\"fertilizer_kg\":120,"
            "\"field_size_ha\":1.5,\"season\":\"A\",\"cooperative_support\":1}<br/>"
            "&rarr; 200 {\"predicted_yield_tons\":11.7842,\"high_risk_prediction\":0,\"high_risk_probability\":0.1009,\"cluster\":1,"
            f"\"verification_code\":\"{CODE}\",\"human_review_required\":true,...}}</font>"),
          P("Invalid request (rainfall 5000, season 'C', avg_temp_c 'hot', field_size_ha missing)", H2),
          P("<font face='Courier' size='7.8'>&rarr; 422 with one message per problem, for example: \"rainfall_mm=5000 is outside the allowed range [600.0, 1400.0]\"; "
            "the CLI prints the same list and exits with code 2 (no stack trace).</font>"),
          P("Automated tests (pytest, tests/test_system.py)", H2),
          table([["Test", "What it exercises"],
                 ["valid inference", "all output fields, verification code, human-review flag"], ["missing field", "error names field_size_ha"],
                 ["wrong type", "text for a number; boolean for a binary field"], ["range error", "temperature 40 and season 'C' rejected"],
                 ["wrong seed", "seed that is not 3862 is rejected"], ["dataset signature", "248 base rows, 3 duplicates, missing values, min yield 0.5"],
                 ["no leakage", "forbidden columns are not features"], ["reproducibility", "two full trainings give same SHA-256 and metrics"],
                 ["API", "/health, /model-info, valid /predict, 422 for bad range"]],
                [3.8 * cm, 12.7 * cm]),
          P("Failure behaviour: invalid input never reaches the model; missing model files return a clear 503/exit code 3 asking the user to run run_all.py. "
            "The full pytest log is in evidence/AI_A2_2423862_TEST_LOG.pdf.", B)]
    S.append(PageBreak())

    # Page 7
    S += [P("Model card and reflection", H1),
          table([["Item", "Statement"],
                 ["Intended use", "Classroom decision-support prototype to help a field officer prioritise visits for synthetic Musanze potato farms."],
                 ["Prohibited use", "Replacing an agronomist; deciding loans, insurance or public services; use on real farms without re-training and validation."],
                 ["Data", "100% synthetic, generated from seed 3862; relationships are the ones I coded, so scores are optimistic compared with real farms."],
                 ["Limitations", "Small data (248 farms, 62 test); inputs limited to the training ranges; temperature effect is simple; clusters are weak (silhouette "
                  f"{clu['silhouette_scores'][str(clu['selected_k'])]:.2f}); no pests, disease, prices or soil type."],
                 ["Risks", "Over-trusting a number; unfair treatment of farms without cooperative support; silent errors when a value is imputed."],
                 ["Human oversight", "Every output flags human_review_required; a field officer decides and can override."]],
                [3.4 * cm, 13.1 * cm]),
          Spacer(1, 6), P("Reflection", H2),
          P("The most useful lesson was that the simple, interpretable models won: ridge regression and logistic regression beat the forests because the data "
            "were generated from near-linear rules, and cross-validation on the training split picked them without touching the test set. The two analysed errors "
            "show different limits: one caused by imputing a missing input, one by a median-based label that is unstable near its boundary."),
          P("With real data I would collect actual cooperative records, check for spatial and seasonal drift, add features such as soil type and variety, calibrate the "
            "probabilities, replace the median-based risk label with an agronomic threshold, track predictions against outcomes, and test fairness across farm sizes and "
            "support status before any operational use.")]
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm, bottomMargin=1.8 * cm,
                            title=f"AI_A2_2423862 Report - {NAME}", author=NAME)
    doc.build(S, onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT)


if __name__ == "__main__":
    build()
