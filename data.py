"""Loading of the files created by the notebook. Everything is cached, nothing is trained here."""
import json

import pandas as pd
import streamlit as st

import predictor

BASE = predictor.BASE
MODEL_FILES = ["best_rainfall_model.joblib", "feature_schema.json", "model_metrics.csv", "model_report.json"]
DATA_FILES = ["weather_insights.csv.gz"]


def missing(files):
    return [f for f in files if not (BASE / f).exists()]


@st.cache_data
def read_json(name):
    return json.load(open(BASE / name, encoding="utf-8"))


@st.cache_data
def read_metrics():
    return pd.read_csv(BASE / "model_metrics.csv")


@st.cache_resource
def load_model():
    return predictor.load()


@st.cache_data
def load_weather():
    df = pd.read_csv(BASE / "weather_insights.csv.gz", parse_dates=["Date"])
    df["Outcome"] = df["RainTomorrow_bin"].map({1: "Rain tomorrow", 0: "No rain tomorrow"})
    return df


def setup_instructions():
    st.markdown("""**One-time setup**
1. Put `weatherAUS.csv` in the `data/` folder.
2. Open `weather_rainfall_prediction.ipynb` and run **all** cells (Kernel → Restart & Run All).
3. Keep the files it creates next to `app.py`, then refresh this page.""")


def require(files, what):
    """Show a friendly empty state and return False if needed files are missing."""
    gone = missing(files)
    if gone:
        st.markdown(f'<div class="note"><b>☁️ {what} is not ready yet.</b><br>Missing: {", ".join(gone)}</div>', unsafe_allow_html=True)
        setup_instructions()
        return False
    return True


def safe_model():
    """(pipeline, schema) or None after showing a friendly message, so a broken model file never crashes a page."""
    try:
        return load_model()
    except Exception as e:
        st.error(f"The saved model could not be loaded: {e}")
        st.info("Most common cause: it was trained with a different scikit-learn version. Re-run the notebook in this same environment, then restart the app.")
        return None
