CELLS_A = []
def md(t): CELLS_A.append(("md", t))
def code(t, kind="code"): CELLS_A.append((kind, t.strip("\n")))

md("""# Weather Data Analysis and Rainfall Prediction Using Machine Learning
**Dataset:** WeatherAUS (Rattle / Kaggle) · **Target:** `RainTomorrow` (Yes/No) · **Run order:** top to bottom (Runtime → Run all).

Every number below is computed from the real data when the notebook runs. Nothing is hard-coded.""")

md("## 1. Install / import libraries and load the dataset")
code("""import os, json, time, warnings, urllib.request
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
import joblib, sklearn

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, balanced_accuracy_score, confusion_matrix,
                             classification_report, precision_recall_curve, roc_curve)
try:
    from IPython.display import display
except ImportError:
    display = print

warnings.filterwarnings("ignore")
SEED = 42
np.random.seed(SEED); rng = np.random.default_rng(SEED)      # reproducibility
os.makedirs("outputs", exist_ok=True)                          # all graphs are saved here
sns.set_theme(style="whitegrid")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

def save_fig(name):
    plt.tight_layout(); plt.savefig(f"outputs/{name}.png", dpi=100, bbox_inches="tight"); plt.show(); plt.close("all")

print("pandas", pd.__version__, "| numpy", np.__version__, "| scikit-learn", sklearn.__version__)""")

code("""# Load the data: local file -> download -> manual upload (practical fallbacks if the URL fails)
DATA_URL = "https://rattle.togaware.com/weatherAUS.csv"
candidates = ["weatherAUS.csv", "data/weatherAUS.csv", "/content/weatherAUS.csv", "/mnt/user-data/uploads/weatherAUS.csv"]
path = next((p for p in candidates if os.path.exists(p)), None)
if path is None:
    try:
        urllib.request.urlretrieve(DATA_URL, "weatherAUS.csv"); path = "weatherAUS.csv"
    except Exception as e:
        print("Download failed:", e)
        try:
            from google.colab import files            # Colab fallback: upload the Kaggle CSV by hand
            up = files.upload(); path = list(up)[0]
        except ImportError:
            raise FileNotFoundError("Put weatherAUS.csv next to this notebook (Kaggle link in README.md).")
df_raw = pd.read_csv(path)
print("Loaded:", path, "->", df_raw.shape)""")

md("## 2. First look: head, tail, shape, columns, dtypes, info, missing values, duplicates")
code("""display(df_raw.head()); display(df_raw.tail())
print("Shape:", df_raw.shape); print("Columns:", list(df_raw.columns)); print("\\nDtypes:"); print(df_raw.dtypes)
print("\\nInfo:"); df_raw.info()
miss = pd.DataFrame({"missing": df_raw.isna().sum(), "missing_%": (df_raw.isna().mean()*100).round(2)})
print("\\nMissing values:"); display(miss[miss.missing > 0].sort_values("missing", ascending=False))
print("Duplicate rows:", df_raw.duplicated().sum())
print("\\nDate range:", df_raw.Date.min(), "->", df_raw.Date.max(), "| locations:", df_raw.Location.nunique())
print("\\nTarget distribution (incl. missing):"); print(df_raw.RainTomorrow.value_counts(dropna=False))""")

md("""## 3. Data cleaning and preparation
* `Date` → datetime; exact duplicates dropped; rows **without a target** dropped (cannot be used for supervised learning).
* **`RISK_MM` removed**: it is the *next day's* rainfall amount, i.e. the target in disguise (target leakage).
* Remaining features are all observations of day *t* (9 am / 3 pm readings, today's rainfall), so a forecast for day *t+1* can legitimately use them.
* `df` keeps NaNs on purpose: the ML pipeline imputes them **using training data only** (no preprocessing leakage).
* `df_clean` (median-by-location imputation, binary/label encodings) is for EDA and for the saved `cleaned_weather.csv` only.""")
code("""df = df_raw.copy()
df["Date"] = pd.to_datetime(df["Date"])
n0 = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=["RainTomorrow"])
leaky = [c for c in ["RISK_MM"] if c in df.columns]
df = df.drop(columns=leaky)
df = df.sort_values(["Date", "Location"]).reset_index(drop=True)      # chronological order
df["Year"], df["Month"] = df.Date.dt.year, df.Date.dt.month
df["Season"] = df.Month.map({12:"Summer",1:"Summer",2:"Summer",3:"Autumn",4:"Autumn",5:"Autumn",
                             6:"Winter",7:"Winter",8:"Winter",9:"Spring",10:"Spring",11:"Spring"})
df["RainTomorrow_bin"] = (df.RainTomorrow == "Yes").astype(int)
df["RainToday_bin"] = df.RainToday.map({"No": 0, "Yes": 1})
print(f"Rows: {n0} -> {len(df)} | removed leaky columns: {leaky} | 'RISK_MM' still present? {'RISK_MM' in df.columns}")

NUM = ["MinTemp","MaxTemp","Rainfall","Evaporation","Sunshine","WindGustSpeed","WindSpeed9am","WindSpeed3pm",
       "Humidity9am","Humidity3pm","Pressure9am","Pressure3pm","Cloud9am","Cloud3pm","Temp9am","Temp3pm"]
WIND = ["WindGustDir", "WindDir9am", "WindDir3pm"]

# ---- cleaned copy for EDA / export (imputation here is NOT used for model evaluation) ----
df_clean = df.copy()
df_clean[NUM] = df_clean.groupby("Location")[NUM].transform(lambda s: s.fillna(s.median()))
df_clean[NUM] = df_clean[NUM].fillna(df_clean[NUM].median())                 # locations that never measure a variable
for c in WIND + ["RainToday"]:
    df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])
df_clean["RainToday_bin"] = (df_clean.RainToday == "Yes").astype(int)
for c in ["Location"] + WIND:                                                 # label encoding of categoricals
    df_clean[c + "_code"] = df_clean[c].astype("category").cat.codes
df_clean.to_csv("cleaned_weather.csv", index=False)
print("Saved cleaned_weather.csv", df_clean.shape, "| remaining NaNs in df_clean:", int(df_clean.isna().sum().sum()))""")

md("## 4. Exploratory Data Analysis (graphs saved in `outputs/`)")
code("""# 4a Histograms
cols = ["MinTemp","MaxTemp","Humidity3pm","Pressure3pm","WindGustSpeed","Temp9am"]
fig, ax = plt.subplots(2, 3, figsize=(15, 7))
for a, c in zip(ax.ravel(), cols):
    sns.histplot(df[c].dropna(), bins=40, kde=True, ax=a, color="steelblue"); a.set_title(f"Histogram of {c}")
save_fig("01_histograms")

# 4b Box plots: key variables split by tomorrow's rain
fig, ax = plt.subplots(1, 4, figsize=(16, 4))
for a, c in zip(ax, ["Humidity3pm", "Pressure3pm", "Sunshine", "MaxTemp"]):
    sns.boxplot(data=df, x="RainTomorrow", y=c, ax=a, palette="Set2"); a.set_title(f"{c} vs RainTomorrow")
save_fig("02_boxplots")""")
code("""# 4c Bar charts: rain probability by location and by month
fig, ax = plt.subplots(1, 2, figsize=(16, 9), gridspec_kw={"width_ratios": [1.3, 1]})
loc_rate = df.groupby("Location").RainTomorrow_bin.mean().sort_values()
loc_rate.plot.barh(ax=ax[0], color="teal"); ax[0].set_title("P(RainTomorrow = Yes) by location"); ax[0].set_xlabel("proportion")
df.groupby("Month").RainTomorrow_bin.mean().plot.bar(ax=ax[1], color="coral"); ax[1].set_title("P(RainTomorrow = Yes) by month"); ax[1].set_ylabel("proportion")
save_fig("03_bar_charts")

# 4d Pie charts
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
for a, (c, t) in zip(ax, [("RainTomorrow", "RainTomorrow"), ("RainToday", "RainToday"), ("Season", "Rows by season")]):
    df[c].value_counts().plot.pie(autopct="%1.1f%%", ax=a, startangle=90); a.set_ylabel(""); a.set_title(t)
save_fig("04_pie_charts")""")
code("""# 4e Scatter plots (random 6000-row sample for readability)
s = df.sample(6000, random_state=SEED)
fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(data=s, x="Humidity3pm", y="Temp3pm", hue="RainTomorrow", alpha=.5, s=14, ax=ax[0])
sns.scatterplot(data=s, x="Pressure3pm", y="Humidity3pm", hue="RainTomorrow", alpha=.5, s=14, ax=ax[1])
save_fig("05_scatter_plots")

# 4f Line chart: monthly averages over time
m = df.set_index("Date")[["Rainfall", "MaxTemp", "RainTomorrow_bin"]].resample("MS").mean()
fig, ax = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
for a, (c, col) in zip(ax, [("MaxTemp", "tomato"), ("Rainfall", "royalblue"), ("RainTomorrow_bin", "seagreen")]):
    m[c].plot(ax=a, color=col); a.set_ylabel(c)
ax[0].set_title("Monthly average over all stations")
save_fig("06_line_chart")

# 4g Correlation heatmap (pairwise-complete observations)
plt.figure(figsize=(12, 9))
sns.heatmap(df[NUM + ["RainTomorrow_bin"]].corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f", annot_kws={"size": 7})
plt.title("Correlation heatmap"); save_fig("07_correlation_heatmap")
print(df[NUM].corrwith(df.RainTomorrow_bin).sort_values(ascending=False).round(3).to_string())""")

md("## 5. Descriptive statistics")
code("""def iqr_outliers(x):
    q1, q3 = x.quantile([.25, .75]); i = q3 - q1
    return int(((x < q1 - 1.5*i) | (x > q3 + 1.5*i)).sum())
desc = pd.DataFrame({
    "mean": df[NUM].mean(), "median": df[NUM].median(), "mode": [df[c].mode().iloc[0] for c in NUM],
    "variance": df[NUM].var(), "std": df[NUM].std(), "min": df[NUM].min(), "max": df[NUM].max(),
    "Q1": df[NUM].quantile(.25), "Q2": df[NUM].quantile(.5), "Q3": df[NUM].quantile(.75)})
desc["IQR"] = desc.Q3 - desc.Q1
desc["IQR_outliers"] = [iqr_outliers(df[c].dropna()) for c in NUM]
display(desc.round(3))
print("\\ndescribe():"); display(df[NUM].describe().T.round(3))""")

md("## 6. Probability distributions (real data vs simulated)")
code("""# --- Normal: real Pressure3pm vs a simulated Normal(mu, sd) ---
x = df["Pressure3pm"].dropna(); mu, sd = x.mean(), x.std()
sim_n = rng.normal(mu, sd, len(x))
print(f"Pressure3pm: mean={mu:.2f}, sd={sd:.2f}, skew={stats.skew(x):.3f}, excess kurtosis={stats.kurtosis(x):.3f}")
print("Share within 1/2/3 sd (real):", [round(((x > mu-k*sd) & (x < mu+k*sd)).mean(), 4) for k in (1, 2, 3)], "| theory: [0.6827, 0.9545, 0.9973]")
sh = stats.shapiro(x.sample(5000, random_state=SEED)); print(f"Shapiro-Wilk (n=5000 sample): W={sh.statistic:.4f}, p={sh.pvalue:.3g}")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5)); g = np.linspace(x.min(), x.max(), 300)
for a, d, t in [(ax[0], x, "Real Pressure3pm"), (ax[1], sim_n, "Simulated Normal")]:
    sns.histplot(d, bins=50, stat="density", ax=a, color="lightsteelblue"); a.plot(g, stats.norm.pdf(g, mu, sd), "r"); a.set_title(t)
stats.probplot(x.sample(5000, random_state=SEED), dist="norm", plot=ax[2]); ax[2].set_title("Q-Q plot (real)")
save_fig("08_normal_distribution")""")
code("""# --- Uniform: simulated U(lo, hi) vs real MaxTemp (KS test against a uniform) ---
y = df["MaxTemp"].dropna(); lo, hi = y.min(), y.max()
sim_u = rng.uniform(lo, hi, len(y))
ks_sim = stats.kstest(sim_u, "uniform", args=(lo, hi - lo)); ks_real = stats.kstest(y, "uniform", args=(lo, hi - lo))
print(f"KS vs Uniform -> simulated: D={ks_sim.statistic:.4f}, p={ks_sim.pvalue:.3g} | real MaxTemp: D={ks_real.statistic:.4f}, p={ks_real.pvalue:.3g}")
cnt = df.Month.value_counts().sort_index(); chi = stats.chisquare(cnt)
print(f"Rows per month vs uniform: chi2={chi.statistic:.1f}, p={chi.pvalue:.3g} (observations are roughly, not perfectly, uniform in time)")
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
sns.histplot(sim_u, bins=40, ax=ax[0], color="orchid").set_title("Simulated Uniform"); sns.histplot(y, bins=40, ax=ax[1], color="gold").set_title("Real MaxTemp (not uniform)")
cnt.plot.bar(ax=ax[2], color="gray"); ax[2].set_title("Observations per month")
save_fig("09_uniform_distribution")

# --- Binomial: rainy days in blocks of 7 days at Sydney vs Binomial(7, p) ---
r = df[df.Location == "Sydney"].sort_values("Date").RainTomorrow_bin.values
p_hat = r.mean(); blocks = len(r) // 7; k = r[:blocks*7].reshape(blocks, 7).sum(1)
obs = np.bincount(k, minlength=8) / blocks; theo = stats.binom.pmf(np.arange(8), 7, p_hat); sim_b = np.bincount(rng.binomial(7, p_hat, blocks), minlength=8) / blocks
print(f"Sydney: p(rain day)={p_hat:.3f}, blocks={blocks} | observed var={k.var(ddof=1):.3f} vs binomial var=np(1-p)={7*p_hat*(1-p_hat):.3f}")
print("-> variance above the binomial value = rain days cluster (days are not independent).")
w = .27; plt.figure(figsize=(9, 4.5)); plt.bar(np.arange(8)-w, obs, w, label="Observed (real)"); plt.bar(np.arange(8), theo, w, label="Binomial pmf"); plt.bar(np.arange(8)+w, sim_b, w, label="Simulated")
plt.xlabel("rainy days in a 7-day block"); plt.ylabel("probability"); plt.title("Binomial model for weekly rainy days (Sydney)"); plt.legend()
save_fig("10_binomial_distribution")""")
