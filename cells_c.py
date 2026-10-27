CELLS_C = []
def md(t): CELLS_C.append(("md", t))
def code(t, kind="code"): CELLS_C.append((kind, t.strip("\n")))

md("""## 12-13. Features, chronological split and leakage-safe pipelines
* **Split by time** (no shuffling): train < 2023, validation = 2023, test ≥ 2024 (if the file ends before 2025, e.g. the old Kaggle copy, the timeline is cut 70/15/15 instead). The future is never used to predict the past.
* Imputation, scaling and one-hot encoding live **inside** a `ColumnTransformer` that is fitted on training rows only.
* `Month` is a categorical feature (one-hot) so December and January stay "close" in meaning without extra code; `RainToday` is today's observation, not tomorrow's.""")
code("""NUM_F = NUM
CAT_F = ["Location", "WindGustDir", "WindDir9am", "WindDir3pm", "RainToday", "Month"]
FEATURES = NUM_F + CAT_F                                   # the exact input schema of the saved model
assert not {"RISK_MM", "RainTomorrow", "RainTomorrow_bin", "Date", "Year", "Season"} & set(FEATURES), "leakage / non-input column in FEATURES"

X = df[FEATURES].copy(); X["Month"] = X["Month"].astype(str); y = df["RainTomorrow_bin"]
VAL_START, TEST_START = pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")   # Rattle's weatherAUS.csv runs to 2026
if df.Date.max() < pd.Timestamp("2025-01-01"):      # older Kaggle copy (ends 2017): split the timeline 70/15/15 instead
    VAL_START, TEST_START = df.Date.quantile(0.70), df.Date.quantile(0.85)
tr = df.Date < VAL_START; va = (df.Date >= VAL_START) & (df.Date < TEST_START); te = df.Date >= TEST_START
X_train, y_train, X_val, y_val, X_test, y_test = X[tr], y[tr], X[va], y[va], X[te], y[te]
for nm, m_ in [("train", tr), ("validation", va), ("test", te)]:
    print(f"{nm:10s} rows={m_.sum():6d} | {df.Date[m_].min().date()} -> {df.Date[m_].max().date()} | rain rate={y[m_].mean():.3f}")

def make_prep():
    num = Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([("num", num, NUM_F), ("cat", cat, CAT_F)])

# Expensive models (KNN, SVM) are trained on a reproducible random subset of the training rows; their validation/test sets are untouched.
KNN_N, SVM_N = 60000, 15000
models = {
    "Logistic Regression": (LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED), None),
    "Decision Tree": (DecisionTreeClassifier(max_depth=8, min_samples_leaf=50, class_weight="balanced", random_state=SEED), None),
    "Random Forest": (RandomForestClassifier(n_estimators=100, max_depth=14, min_samples_leaf=20, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED), None),
    "KNN": (KNeighborsClassifier(n_neighbors=25, n_jobs=-1), KNN_N),
    "SVM (RBF)": (SVC(kernel="rbf", C=1.0, class_weight="balanced", random_state=SEED), SVM_N),
    # Gradient Boosting: the histogram version is much faster than GradientBoostingClassifier on ~100k rows
    "Gradient Boosting": (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, max_depth=6, class_weight="balanced", random_state=SEED), None)}
print("Majority-class baseline accuracy on test:", round(1 - y_test.mean(), 4), "-> accuracy alone is misleading; we focus on rain recall / F1")""")

md("""## 14. Class imbalance (only ~1 day in 4-5 is rainy)
Training data is **never resampled** and the test set is never touched. Instead: `class_weight="balanced"` (LR, tree, forest, SVM) and, for every model, a **decision threshold tuned on the validation year** to maximise the F1 of the "Rain" class. KNN has no class weights, so the tuned threshold is its imbalance remedy.""")
code("""def get_scores(pipe, X_):
    return pipe.predict_proba(X_)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X_)

def tune_threshold(y_true, s):
    p, r, t = precision_recall_curve(y_true, s); f = 2 * p * r / (p + r + 1e-12)
    return float(t[np.nanargmax(f[:-1])])

def evaluate(y_true, s, thr):
    pred = (s >= thr).astype(int)
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred, zero_division=0),
            "Recall": recall_score(y_true, pred), "F1": f1_score(y_true, pred), "Balanced Acc": balanced_accuracy_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, s), "PR-AUC": average_precision_score(y_true, s)}

fitted, scores, rows = {}, {}, []
for name, (clf, n_sub) in models.items():
    idx = X_train.sample(n_sub, random_state=SEED).index if n_sub else X_train.index
    pipe = Pipeline([("prep", make_prep()), ("clf", clf)])
    t0 = time.time(); pipe.fit(X_train.loc[idx], y_train.loc[idx]); fit_s = time.time() - t0
    s_val, s_te = get_scores(pipe, X_val), get_scores(pipe, X_test)
    default_thr = 0.5 if hasattr(pipe, "predict_proba") else 0.0
    thr = tune_threshold(y_val, s_val); m = evaluate(y_test, s_te, thr); m_def = evaluate(y_test, s_te, default_thr)
    rows.append({"Model": name, "Train rows": len(idx), "Fit sec": round(fit_s, 1), "Threshold": thr,
                 "Val F1 (tuned)": f1_score(y_val, (s_val >= thr).astype(int)), **m,
                 "Recall @default thr": m_def["Recall"], "F1 @default thr": m_def["F1"]})
    fitted[name], scores[name] = pipe, s_te
    print(f"{name:20s} done in {fit_s:6.1f}s | test F1={m['F1']:.3f} recall={m['Recall']:.3f} AUC={m['ROC-AUC']:.3f}")""")

md("""## 15. Model comparison and best-model selection
**Selection criterion (fixed before looking at results):** the model with the highest **F1-score of the "Rain" class on the validation year**, using each model's own tuned threshold. F1 balances precision and recall for the minority class, so accuracy never decides. Test-set Recall and ROC-AUC are shown beside it as a sanity check; the test set is only used to *report* results, never to choose.""")
code("""res = pd.DataFrame(rows).set_index("Model"); display(res.round(4))
res.round(5).reset_index().to_csv("model_metrics.csv", index=False)
best_name = res["Val F1 (tuned)"].idxmax()        # selection uses the VALIDATION year only; test is reported, never used to choose
best_pipe, best_thr = fitted[best_name], float(res.loc[best_name, "Threshold"])
print(f"\\nBest model (highest validation F1 for the Rain class): {best_name} | threshold = {best_thr:.4f}")
print("Test-set ranking by F1:", list(res.F1.sort_values(ascending=False).index))
print("Test-set ranking by ROC-AUC:", list(res["ROC-AUC"].sort_values(ascending=False).index))

pred_best = (scores[best_name] >= best_thr).astype(int)
print(f"\\nClassification report - {best_name} on the 2024+ test set"); print(classification_report(y_test, pred_best, target_names=["No Rain", "Rain"], digits=3))

# charts: metric comparison, ROC curves, confusion matrices
res[["Precision", "Recall", "F1", "ROC-AUC"]].plot.bar(figsize=(11, 4.5), rot=15); plt.ylim(0, 1); plt.title("Model comparison on the chronological test set"); plt.legend(loc="lower right")
save_fig("14_model_comparison")
plt.figure(figsize=(6.5, 5.5))
for name in models:
    fpr, tpr, _ = roc_curve(y_test, scores[name]); plt.plot(fpr, tpr, label=f"{name} (AUC={res.loc[name, 'ROC-AUC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curves (test)"); plt.legend(fontsize=8)
save_fig("15_roc_curves")
fig, ax = plt.subplots(2, 3, figsize=(15, 8))
for a, name in zip(ax.ravel(), models):
    cm = confusion_matrix(y_test, (scores[name] >= res.loc[name, "Threshold"]).astype(int))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=a, xticklabels=["No Rain", "Rain"], yticklabels=["No Rain", "Rain"])
    a.set_title(name); a.set_xlabel("Predicted"); a.set_ylabel("Actual")
save_fig("16_confusion_matrices")

# interpretability of the best model, when available
clf_ = best_pipe.named_steps["clf"]; names_ = best_pipe.named_steps["prep"].get_feature_names_out()
imp = getattr(clf_, "feature_importances_", None)
if imp is None and hasattr(clf_, "coef_"): imp = np.abs(clf_.coef_[0])
if imp is not None and len(imp) == len(names_):
    top = pd.Series(imp, index=names_).sort_values().tail(15)
    top.plot.barh(figsize=(8, 5), color="seagreen"); plt.title(f"Top 15 features - {best_name}"); save_fig("17_feature_importance")""")

md("""## 16-17. Save artifacts
* `cleaned_weather.csv`, `model_metrics.csv` (already written), graphs in `outputs/`.
* `best_rainfall_model.joblib` = the **complete fitted Pipeline** (imputers + encoder + scaler + classifier), trained on the training years only so it matches the reported test metrics.
* `feature_schema.json` = exact input columns, allowed categories, plausible ranges, tuned threshold and three real test rows as GUI examples. The GUIs read this file, so they cannot drift from the model.""")
code("""UNITS = {"MinTemp":"°C","MaxTemp":"°C","Rainfall":"mm","Evaporation":"mm","Sunshine":"hours","WindGustSpeed":"km/h","WindSpeed9am":"km/h","WindSpeed3pm":"km/h",
         "Humidity9am":"%","Humidity3pm":"%","Pressure9am":"hPa","Pressure3pm":"hPa","Cloud9am":"oktas (0-9)","Cloud3pm":"oktas (0-9)","Temp9am":"°C","Temp3pm":"°C"}
HARD = {"MinTemp":(-20,55),"MaxTemp":(-20,60),"Rainfall":(0,1000),"Evaporation":(0,100),"Sunshine":(0,24),"WindGustSpeed":(0,250),"WindSpeed9am":(0,250),"WindSpeed3pm":(0,250),
        "Humidity9am":(0,100),"Humidity3pm":(0,100),"Pressure9am":(900,1100),"Pressure3pm":(900,1100),"Cloud9am":(0,9),"Cloud3pm":(0,9),"Temp9am":(-20,60),"Temp3pm":(-20,60)}
def clean(v): return None if (v is None or (isinstance(v, float) and np.isnan(v))) else (float(v) if isinstance(v, (int, float, np.number)) else str(v))
ex_idx = [X_test[y_test == 1].sample(1, random_state=SEED).index[0], X_test[y_test == 0].sample(1, random_state=SEED).index[0], X_test[y_test == 1].sample(1, random_state=SEED + 1).index[0]]
schema = {
    "model_name": best_name, "features": FEATURES, "numeric_features": NUM_F, "categorical_features": CAT_F,
    "numeric": {c: {"unit": UNITS[c], "hard_min": HARD[c][0], "hard_max": HARD[c][1], "train_min": float(X_train[c].min()), "train_max": float(X_train[c].max()), "train_median": float(X_train[c].median())} for c in NUM_F},
    "categorical": {c: sorted(X_train[c].dropna().unique().tolist(), key=(lambda v: int(v)) if c == "Month" else str) for c in CAT_F},
    "threshold": best_thr, "has_predict_proba": bool(hasattr(best_pipe, "predict_proba")),
    "split": {"validation_start": str(pd.Timestamp(VAL_START).date()), "test_start": str(pd.Timestamp(TEST_START).date())},
    "test_metrics": {k: float(v) for k, v in res.loc[best_name, ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].items()},
    "examples": [{"actual_RainTomorrow": "Yes" if y_test.loc[i] else "No", **{c: clean(X_test.loc[i, c]) for c in FEATURES}} for i in ex_idx]}
json.dump(schema, open("feature_schema.json", "w"), indent=2)
joblib.dump(best_pipe, "best_rainfall_model.joblib", compress=3)
print("Saved best_rainfall_model.joblib:", round(os.path.getsize("best_rainfall_model.joblib") / 1e6, 1), "MB | feature_schema.json | model_metrics.csv | cleaned_weather.csv | outputs/:", len(os.listdir("outputs")), "files")
print("Model input columns:", list(best_pipe.feature_names_in_) == FEATURES)""")

md("""## 17b. Save a report for the Streamlit dashboard
`model_report.json` stores the dataset overview, the selection criterion and every model's confusion matrix, all computed above, so the app can show them without re-running anything.""")
code("""report = {
    "selection_criterion": "Highest F1-score of the Rain class on the validation year (threshold tuned on that year); test metrics are reported only.",
    "best_model": best_name,
    "dataset": {"source_file": str(path), "rows_raw": int(len(df_raw)), "rows_used": int(len(df)), "columns_raw": int(df_raw.shape[1]),
                "date_min": str(df.Date.min().date()), "date_max": str(df.Date.max().date()), "locations": int(df.Location.nunique()),
                "rain_tomorrow_rate": float(df.RainTomorrow_bin.mean()), "removed_leaky_columns": leaky,
                "missing_pct": {c: float(round(df[c].isna().mean() * 100, 2)) for c in NUM_F},
                "rows_train": int(tr.sum()), "rows_validation": int(va.sum()), "rows_test": int(te.sum()),
                "test_rain_rate": float(y_test.mean()), "majority_class_accuracy_test": float(1 - y_test.mean())},
    "confusion_matrices": {n: confusion_matrix(y_test, (scores[n] >= res.loc[n, "Threshold"]).astype(int)).tolist() for n in models},
    "correlation_with_target": df[NUM].corrwith(df.RainTomorrow_bin).round(4).to_dict()}
json.dump(report, open("model_report.json", "w"), indent=2)
print("Saved model_report.json")""")

md("""## 17c. Extra exports for the SkySense app (explainability, calibration, profiles, data)
* **Permutation importance** - how much the test ROC-AUC drops when one input column is shuffled (works for any model; shows how the *model* uses information, **not** physical causation).
* **Score calibration** - among past test days with a given model score, how often did rain really follow? This makes a probability-style score meaningful.
* **Rainy vs dry profile** - median of each measurement before rainy / dry days (training years only).
* `weather_insights.csv.gz` - the cleaned data *without* imputed values, used by the app's Insights pages.""")
code("""from sklearn.inspection import permutation_importance
rep = json.load(open("model_report.json"))

# 1) permutation importance of the saved best model (subsample keeps SVM/KNN fast)
sub_ = X_test.sample(min(3000, len(X_test)), random_state=SEED)
pi = permutation_importance(best_pipe, sub_, y_test.loc[sub_.index], scoring="roc_auc", n_repeats=3, random_state=SEED, n_jobs=1)
rep["permutation_importance"] = {"scoring": "drop in test ROC-AUC when the column is shuffled", "rows_used": int(len(sub_)),
    "values": {c: {"mean": float(m), "std": float(s)} for c, m, s in zip(FEATURES, pi.importances_mean, pi.importances_std)}}
display(pd.Series(pi.importances_mean, index=FEATURES).sort_values(ascending=False).head(8).round(4))

# 2) calibration table (only if the model outputs probabilities)
sc = np.asarray(scores[best_name]); edges = [0, .2, .4, .6, .8, 1.0001]
if hasattr(best_pipe, "predict_proba"):
    b = np.digitize(sc, edges[1:-1])
    rep["calibration"] = [{"low": edges[i], "high": min(edges[i + 1], 1.0), "n": int((b == i).sum()),
                           "rain_rate": float(y_test.values[b == i].mean()) if (b == i).sum() else None} for i in range(5)]
    display(pd.DataFrame(rep["calibration"]).round(3))
else:
    rep["calibration"] = None

# 3) typical values before rainy vs dry days (training years, raw data incl. NaN ignored)
trd = df[tr]
rep["profile"] = {c: {"dry_median": float(trd.loc[trd.RainTomorrow_bin == 0, c].median()), "rain_median": float(trd.loc[trd.RainTomorrow_bin == 1, c].median()),
                      "p1": float(trd[c].quantile(.01)), "p99": float(trd[c].quantile(.99))} for c in NUM_F}

# 4) ROC curve points for every model (downsampled)
rep["roc"] = {}
for n_ in models:
    fpr, tpr, _ = roc_curve(y_test, scores[n_]); ix = np.linspace(0, len(fpr) - 1, min(150, len(fpr))).astype(int)
    rep["roc"][n_] = {"fpr": fpr[ix].round(4).tolist(), "tpr": tpr[ix].round(4).tolist()}
rep["models_info"] = {n_: {"params": {k: str(v) for k, v in models[n_][0].get_params().items() if k in ("C", "max_depth", "n_estimators", "n_neighbors", "max_iter", "learning_rate", "kernel", "min_samples_leaf", "class_weight")},
                           "train_rows": int(KNN_N if n_ == "KNN" else SVM_N if n_ == "SVM (RBF)" else tr.sum())} for n_ in models}
json.dump(rep, open("model_report.json", "w"), indent=2)

# 5) data for the Insights pages (real values, NaN kept, no imputation)
keep = ["Date", "Location"] + NUM + WIND + ["RainToday", "RainTomorrow_bin", "Year", "Month", "Season"]
df[keep].assign(Date=df.Date.dt.strftime("%Y-%m-%d")).to_csv("weather_insights.csv.gz", index=False, compression="gzip")
print("Updated model_report.json and saved weather_insights.csv.gz", round(os.path.getsize("weather_insights.csv.gz") / 1e6, 1), "MB")""")

md("## 18. Reload the saved model and verify predictions")
code("""loaded = joblib.load("best_rainfall_model.joblib"); sch = json.load(open("feature_schema.json"))
assert list(loaded.feature_names_in_) == sch["features"], "schema mismatch"
sample = X_test.sample(8, random_state=SEED)
s_loaded = get_scores(loaded, sample); s_orig = get_scores(best_pipe, sample)
assert np.allclose(s_loaded, s_orig), "reloaded model differs from in-memory model"
out = pd.DataFrame({"Location": sample.Location, "Month": sample.Month, "Score": s_loaded.round(3), "Predicted": np.where(s_loaded >= sch["threshold"], "Rain Tomorrow", "No Rain Tomorrow"), "Actual": y_test.loc[sample.index].map({1: "Rain", 0: "No Rain"})})
display(out)
# missing values are handled by the pipeline's imputers (same behaviour the GUI relies on)
sparse_row = pd.DataFrame([{c: np.nan for c in sch["features"]} | {"Location": "Sydney", "Month": "7", "Humidity3pm": 85.0, "Pressure3pm": 1006.0}])
print("Row with only 4 known inputs ->", get_scores(loaded, sparse_row).round(3), "(no error, NaNs imputed from training statistics)")
print("Reload check passed: predictions identical to the in-memory pipeline.")""")

code("""# Optional: bundle everything for download (Colab) - works locally too
import shutil
shutil.make_archive("project_artifacts", "zip", ".", "outputs")
try:
    from google.colab import files
    for f in ["best_rainfall_model.joblib", "feature_schema.json", "model_metrics.csv", "model_report.json", "weather_insights.csv.gz", "cleaned_weather.csv", "project_artifacts.zip"]:
        files.download(f)
except ImportError:
    print("Not in Colab - files are already in the current folder.")""")
