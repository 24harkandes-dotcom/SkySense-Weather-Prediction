# 🌦️ SkySense: Weather Intelligence & Rainfall Insights

*Predict the rain. Understand its impact. Prepare for what comes next.*

An educational Streamlit platform built on the **WeatherAUS** dataset (Australian daily weather) with a full Data Science & Statistics notebook behind it.
It predicts `RainTomorrow` (Yes/No), explains the weather factors linked to rain, and teaches how rainfall affects farms, roads, water, nature, health and the economy.

> **Historical data, not a live forecast.** SkySense never fetches live weather and never invents current readings. It is not an official warning service.

## App sections
| Section | What you get |
|---|---|
| 🏠 Home | Hero, navigation cards, data-calculated discoveries, glossary, quick tour |
| 🌧️ Rainfall Predictor | Grouped inputs with units and tooltips, reset, result card, rain score + historical hit-rate, "what pushed the prediction", comparison with typical rainy/dry days, planning tip, scenario explorer |
| 🌍 Impact Explorer | 6 topics × 4 illustrative rainfall levels: benefits, risks, preparedness, and how often each level occurs in the dataset |
| 📊 Weather Insights | Station/year filters; 8 views incl. correlations, distributions, seasons, wind rose and a **Statistics lab** (describe, probability, CLT, hypothesis test) |
| 🔬 Rainfall Factors | Humidity, pressure, temperature, sunshine, wind, recent rain, location & season, plus permutation feature importance |
| 🧠 Model Insights | Six models compared, metrics in plain English, confusion matrices, ROC, limitations, advanced details, viva concept map |

## Project files
| File | Purpose |
|---|---|
| `weather_rainfall_prediction.ipynb` | All analysis and training (run top to bottom). **Creates the model files.** |
| `app.py` | Streamlit entry point (navigation) |
| `skysense/` | `theme.py` (CSS + chart style), `data.py` (loading), `analysis.py` (statistics), `content.py` (text), `views/` (one file per section) |
| `predictor.py` | Validation, prediction, scenario sweep and local explanation (reads the saved model only) |
| `assets/style.css`, `.streamlit/config.toml` | Look and feel |
| `tools/` | Source of the notebook cells (`python tools/build_notebook.py` rebuilds the `.ipynb`) |
| `data/` | Put `weatherAUS.csv` here |

Created by running the notebook (not included in the ZIP, because they must come from the real data):
`best_rainfall_model.joblib`, `feature_schema.json`, `model_metrics.csv`, `model_report.json`, `weather_insights.csv.gz`, `cleaned_weather.csv`, `outputs/` (graphs).

## Run on Windows
1. **Extract** the ZIP. Open the `weather-rainfall-ml` folder, click the address bar, type `cmd` and press Enter.
2. **Virtual environment**
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
3. **Install:** `pip install -r requirements.txt`
4. **Dataset:** download https://rattle.togaware.com/weatherAUS.csv (or the Kaggle "Rain in Australia" file) and save it as `data\weatherAUS.csv`.
5. **Train:** `jupyter notebook` → open `weather_rainfall_prediction.ipynb` → **Kernel → Restart & Run All** (several minutes).
6. **Launch:** `streamlit run app.py` → open http://localhost:8501.

If the files from step 5 are missing the app shows setup instructions instead of crashing. After retraining, press Ctrl+C and start the app again.

## Google Colab (training only)
Upload the notebook → **Runtime → Run all** (upload `weatherAUS.csv` if the download fails). The last cell downloads the model files; copy them **next to `app.py`** on your laptop and continue at step 6. Streamlit is meant to run locally.

## Where each concept is covered
| Concept | Notebook | App |
|---|---|---|
| 1 Loading & inspection | §1–2 | Home facts |
| 2 Cleaning, missing values | §3, pipeline imputers (§12–13) | Model Insights → Advanced |
| 3–4 EDA; uni-, bi-, multivariate | §4 | Weather Insights (all views) |
| 5 Mean, median, mode, variance, SD, quartiles, outliers | §5 | Insights → Statistics lab → Describe; Distributions & outliers |
| 6 Probability & distributions | §6 | Statistics lab → Probability (binomial, conditional probability, Wilson CI) |
| 7 Sampling & CLT | §7–9 | Statistics lab → Central Limit Theorem (interactive) |
| 8 Hypothesis tests & confidence intervals | §11 | Statistics lab → Hypothesis test; Factors → Advanced |
| 9 Correlation | §10 | Insights → Correlations |
| 10 Encoding & scaling | §12–13 | Model Insights → Advanced |
| 11 Splitting & leakage | §7, §12–13 (chronological split, `RISK_MM` removed) | Model Insights |
| 12–13 Training, comparison, metrics | §14–15 | Model Insights |
| 14 Best-model selection | §15: highest **validation F1 of the Rain class**; test set only reports | Model Insights |
| 15 Joblib save/reload | §16–18 | Predictor loads the saved file |
| 16 End-to-end prediction | §18 | Rainfall Predictor |

## Reliability notes
* `RISK_MM` is dropped; preprocessing lives inside one scikit-learn `Pipeline` fitted on training years only; split is chronological; unseen categories are ignored; invalid inputs are rejected with messages; class imbalance uses class weights plus a tuned threshold.
* The "rain score" is not a calibrated probability. The app therefore also shows the **historical hit rate** of past test days with a similar score (computed in the notebook). Models without probabilities (SVM) show only a yes/no answer.
* "What pushed this prediction?" and feature importance describe how the **model** behaves, not what causes rain.
* Impact-explorer content is general educational information, not measured by the dataset. Intensity bands are illustrative.

## What was and was not tested
Tested (sandbox, Python 3.12, pandas 3.0, scikit-learn 1.8): the whole notebook ran end to end on **synthetic data shaped like WeatherAUS**; every app page, selector option, filter, the prediction flow, the SVM (no-probability) path, missing files and a corrupted model file were executed against a **stand-in for Streamlit**, and the generated charts were inspected.
**Not tested:** the real `weatherAUS.csv` (so no real metrics exist yet), the real Streamlit rendering (CSS, layout, mobile, widget behaviour), Windows and Colab runs. Charts use Matplotlib (themed), not Plotly.
