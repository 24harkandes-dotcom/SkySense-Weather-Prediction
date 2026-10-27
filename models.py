"""Model Insights: what was trained, how it was compared and what the numbers mean."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from skysense import content, data, theme

CONCEPTS = [
    ("1. Dataset loading and inspection", "§1–2", "Home (facts), Weather Insights"),
    ("2. Data cleaning, missing values", "§3 and pipeline imputers (§12–13)", "Model Insights → Advanced"),
    ("3–4. EDA; univariate, bivariate, multivariate analysis", "§4", "Weather Insights (all views)"),
    ("5. Mean, median, mode, variance, SD, quartiles, outliers", "§5", "Insights → Statistics lab → Describe; Distributions & outliers"),
    ("6. Probability and distributions", "§6", "Insights → Statistics lab → Probability"),
    ("7. Sampling and Central Limit Theorem", "§7–9", "Insights → Statistics lab → Central Limit Theorem"),
    ("8. Hypothesis tests and confidence intervals", "§11", "Insights → Statistics lab → Hypothesis test; Factors → Advanced"),
    ("9. Correlation and interpretation", "§10", "Insights → Correlations"),
    ("10. Encoding and scaling", "§12–13 (OneHotEncoder, StandardScaler)", "Model Insights → Advanced"),
    ("11. Train/test split and leakage prevention", "§7, §12–13 (chronological split, RISK_MM removed)", "Model Insights"),
    ("12–13. Model training, comparison, metrics", "§14–15", "Model Insights"),
    ("14. Best-model selection", "§15 (validation F1)", "Model Insights"),
    ("15. Joblib saving and reloading", "§16–18", "Predictor (loads the saved file)"),
    ("16. End-to-end prediction", "§18", "Rainfall Predictor"),
]


def render():
    theme.title("🧠 Model Insights", "The science behind the prediction, in plain language.")
    if not data.require(data.MODEL_FILES, "The model results"):
        return
    loaded = data.safe_model()
    if loaded is None:
        return
    _, schema = loaded; report = data.read_json("model_report.json"); M = data.read_metrics(); ds = report["dataset"]
    best, tm = schema["model_name"], schema["test_metrics"]
    st.markdown(f'<div class="result rain"><span class="icon">🏆</span><div class="big">{best}</div><div class="sub">Chosen from six models. Criterion: {report["selection_criterion"]}</div></div>', unsafe_allow_html=True)
    st.markdown(f"**In plain English** (on {ds['rows_test']:,} unseen test days from {schema['split']['test_start']} onward):")
    st.markdown(f"- Of every 100 days the model called rainy, about **{tm['Precision'] * 100:.0f}** really were. *(Precision)*\n"
                f"- It caught about **{tm['Recall'] * 100:.0f}** of every 100 days that were followed by rain. *(Recall)*\n"
                f"- Overall it was right on **{tm['Accuracy'] * 100:.0f}%** of days, yet always guessing “no rain” would score **{ds['majority_class_accuracy_test'] * 100:.0f}%**, which is why accuracy alone is misleading here. *(Accuracy)*")
    c = st.columns(5)
    for col, (k, h) in zip(c, [("Accuracy", "How often the prediction was correct overall."), ("Precision", "How often predicted rainy days were really rainy."), ("Recall", "How many of the real rainy days it found."),
                               ("F1", "A balance between precision and recall."), ("ROC-AUC", "How well it separates rainy from dry days across all thresholds (0.5 = chance, 1 = perfect).")]):
        col.metric(k, f"{tm[k]:.2f}", help=h)

    theme.title("Six models, one fair test")
    long = M.melt(id_vars="Model", value_vars=["Precision", "Recall", "F1", "ROC-AUC"], var_name="Metric")
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(data=long, x="Model", y="value", hue="Metric", palette=[theme.SKY, theme.BLUE, theme.NAVY, theme.AMBER], ax=ax)
    ax.set_ylim(0, 1); ax.set_ylabel("score (higher is better)"); ax.set_xlabel(""); ax.set_title("Model comparison on the test years"); plt.setp(ax.get_xticklabels(), rotation=15, ha="right"); ax.legend(ncol=4, fontsize=8)
    theme.show(fig); theme.what_it_shows("each model's scores on data it never saw. The winner is chosen on a separate validation year, not on this chart.")
    show = M.set_index("Model")[["F1", "Recall", "Precision", "ROC-AUC", "Accuracy"]]
    st.dataframe(show.style.format("{:.3f}").highlight_max(axis=0, color="#cfe8cf"))

    c1, c2 = st.columns(2)
    with c1:
        names = list(report["confusion_matrices"]); pick = st.selectbox("Confusion matrix for", names, index=names.index(best), key="cm_pick")
        cm = np.array(report["confusion_matrices"][pick])
        fig, ax = plt.subplots(figsize=(4.6, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax, xticklabels=["No rain", "Rain"], yticklabels=["No rain", "Rain"]); ax.set_xlabel("Model said"); ax.set_ylabel("What happened"); ax.grid(False); plt.setp(ax.get_yticklabels(), rotation=0); ax.set_title(pick)
        theme.show(fig)
        st.caption(f"Top-left: correct dry days ({cm[0, 0]:,}). Bottom-right: rainy days caught ({cm[1, 1]:,}). Top-right: **false alarms** ({cm[0, 1]:,}). Bottom-left: **missed rain** ({cm[1, 0]:,}).")
    with c2:
        fig, ax = plt.subplots(figsize=(4.8, 4.2))
        for n, r in report.get("roc", {}).items():
            ax.plot(r["fpr"], r["tpr"], lw=2.6 if n == best else 1.3, label=n)
        ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("false alarm rate"); ax.set_ylabel("rainy days caught"); ax.set_title("ROC curves"); ax.legend(fontsize=7)
        theme.show(fig); st.caption("Curves closer to the top-left corner separate rainy from dry days better.")

    theme.title("Honest limitations")
    for t in ["It learned from **historical Australian stations only**; it should not be applied to other countries.",
              "It is **not a live forecast**: it sees only the numbers you give it, with no radar, satellite or atmospheric model.",
              "Rain is uncertain; even the best model misses some rainy days and raises some false alarms (see the confusion matrix).",
              "Importance and patterns describe **association**, not cause and effect.",
              "KNN and SVM were trained on a random subset of the training rows for speed; the others used all of them.",
              "Climate and measurement practices change over time, so performance on future years may differ."]:
        st.markdown(f"- {t}")

    with st.expander("⚙️ Advanced model details"):
        st.dataframe(M.set_index("Model").style.format("{:.4f}", subset=M.select_dtypes("number").columns.difference(["Train rows"])))
        st.markdown(f"**Split (chronological, no shuffling):** train {ds['rows_train']:,} rows · validation {ds['rows_validation']:,} (from {schema['split']['validation_start']}) · test {ds['rows_test']:,} (from {schema['split']['test_start']}). "
                    f"Removed to prevent leakage: {ds['removed_leaky_columns'] or 'none found'}. Preprocessing (median/mode imputation, standardisation, one-hot encoding) is fitted on training rows only inside one scikit-learn Pipeline. "
                    "Class imbalance is handled with class weights, and each model's decision threshold is tuned on the validation year.")
        st.markdown("**Model settings:**"); st.json(report.get("models_info", {}))
        st.markdown(f"**Input features ({len(schema['features'])}):** {', '.join(schema['features'])}")
    with st.expander("🎓 Project concept map (Data Science & Statistics practicals)"):
        st.table(pd.DataFrame(CONCEPTS, columns=["Concept", "Notebook section", "Where it appears in the app"]).set_index("Concept"))
