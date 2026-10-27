"""Shared prediction logic for the Streamlit app (app.py) and any other front end.

Reads ONLY two files produced by the notebook, so the app can never drift from the trained model:
    best_rainfall_model.joblib  - the complete fitted sklearn Pipeline (imputers + encoder + scaler + classifier)
    feature_schema.json         - exact input columns, categories, ranges, tuned threshold, example rows
Set the environment variable RAIN_MODEL_DIR to load them from another folder.
"""
import json, math, os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("RAIN_MODEL_DIR", Path(__file__).resolve().parent))
MODEL_PATH, SCHEMA_PATH = BASE / "best_rainfall_model.joblib", BASE / "feature_schema.json"
UNKNOWN = "(unknown)"
MIN_NUMERIC_INPUTS = 3          # fewer measurements than this -> the prediction would be mostly imputed guesses
_cache = {}


def load():
    """Load (once) and return (pipeline, schema). Raises a clear error if the notebook has not been run yet."""
    if not _cache:
        missing = [p.name for p in (MODEL_PATH, SCHEMA_PATH) if not p.exists()]
        if missing:
            raise FileNotFoundError(f"Missing {missing} in {BASE}. Run the notebook first (it creates them), "
                                    "then put those files next to app.py.")
        model, schema = joblib.load(MODEL_PATH), json.load(open(SCHEMA_PATH, encoding="utf-8"))
        if list(model.feature_names_in_) != schema["features"]:
            raise ValueError("Model and feature_schema.json disagree - re-run the notebook and copy BOTH files.")
        _cache.update(model=model, schema=schema)
    return _cache["model"], _cache["schema"]


def _is_blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and v.strip() in ("", UNKNOWN))


def validate(inputs, schema):
    """inputs: dict feature -> value (blank/None = unknown). Returns (row dict, errors, warnings)."""
    row, errors, warnings = {}, [], []
    extra = set(inputs) - set(schema["features"])
    if extra:
        errors.append(f"Unexpected inputs (not model features): {sorted(extra)}")
    n_given = 0
    for c in schema["numeric_features"]:
        v, info = inputs.get(c), schema["numeric"][c]
        if _is_blank(v):
            row[c] = np.nan; continue
        try:
            x = float(v)
        except (TypeError, ValueError):
            errors.append(f"{c}: '{v}' is not a number."); continue
        if not np.isfinite(x) or not (info["hard_min"] <= x <= info["hard_max"]):
            errors.append(f"{c}: {v} {info['unit']} is outside the physically plausible range {info['hard_min']} to {info['hard_max']}."); continue
        if not (info["train_min"] <= x <= info["train_max"]):
            warnings.append(f"{c}: {x:g} is outside the range seen in training ({info['train_min']:g} to {info['train_max']:g}); treat the result with caution.")
        row[c] = x; n_given += 1
    for c in schema["categorical_features"]:
        v = inputs.get(c)
        if _is_blank(v):
            row[c] = np.nan; continue
        if c == "Month":
            try: v = str(int(float(v)))
            except (TypeError, ValueError): pass
        v = str(v)
        if v not in schema["categorical"][c]:
            errors.append(f"{c}: '{v}' is not one of {schema['categorical'][c]}.")
        else:
            row[c] = v
    if n_given < MIN_NUMERIC_INPUTS:
        errors.append(f"Please enter at least {MIN_NUMERIC_INPUTS} numeric measurements (e.g. Humidity3pm, Pressure3pm, Sunshine).")
    if not errors:
        if not np.isnan(row["MinTemp"]) and not np.isnan(row["MaxTemp"]) and row["MinTemp"] > row["MaxTemp"]:
            warnings.append("MinTemp is higher than MaxTemp - please double-check.")
        missing = [c for c in schema["features"] if (isinstance(row[c], float) and np.isnan(row[c]))]
        if missing:
            warnings.append(f"{len(missing)} of {len(schema['features'])} inputs left blank; they are filled with typical training values: {', '.join(missing)}.")
    return row, errors, warnings


def _score(model, schema, X):
    """Probability of rain if the model has predict_proba, otherwise the SVM-style decision score."""
    return model.predict_proba(X)[:, 1] if schema["has_predict_proba"] else model.decision_function(X)


def score_sweep(inputs, feature, values):
    """Scenario explorer: scores when ONE numeric feature is varied and every other input is held fixed."""
    model, schema = load()
    row, errors, _ = validate(inputs, schema)
    if errors:
        raise ValueError("; ".join(errors))
    X = pd.DataFrame([row] * len(values), columns=schema["features"])
    X[feature] = list(values)
    return _score(model, schema, X)


def explain(inputs):
    """Local sensitivity: change in score when each provided input is replaced by 'unknown' (the pipeline then imputes the
    training median / most frequent value). Positive = this input pushed the model toward rain. Describes the MODEL, not the atmosphere."""
    model, schema = load()
    row, errors, _ = validate(inputs, schema)
    if errors:
        raise ValueError("; ".join(errors))
    given = [c for c in schema["features"] if not (isinstance(row[c], float) and np.isnan(row[c]))]
    rows = [row] + [{**row, c: np.nan} for c in given]
    s = _score(model, schema, pd.DataFrame(rows, columns=schema["features"]))
    return {c: float(s[0] - s[i + 1]) for i, c in enumerate(given)}


def predict(inputs):
    """Returns dict: ok, errors, warnings, score, probability (or None), threshold, label, rain (bool), score_kind."""
    model, schema = load()
    row, errors, warnings = validate(inputs, schema)
    if errors:
        return {"ok": False, "errors": errors, "warnings": warnings}
    X = pd.DataFrame([row], columns=schema["features"])        # exact column order the pipeline was fitted with
    score = float(_score(model, schema, X)[0])
    kind = "rain score (model probability)" if schema["has_predict_proba"] else "decision score"
    rain = score >= schema["threshold"]
    return {"ok": True, "errors": [], "warnings": warnings, "score": score, "threshold": schema["threshold"], "rain": bool(rain),
            "probability": score if schema["has_predict_proba"] else None,      # None for models without predict_proba (e.g. SVM)
            "label": "Rain tomorrow" if rain else "No rain tomorrow", "score_kind": kind, "model_name": schema["model_name"]}
