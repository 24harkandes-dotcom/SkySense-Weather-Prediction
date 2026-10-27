"""Pure pandas / scipy helpers. Every number the app shows about the data comes from one of these functions."""
import numpy as np
import pandas as pd
from scipy import stats

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def rain_rate(df, by):
    """% of days followed by rain, for each group."""
    return df.groupby(by)["RainTomorrow_bin"].mean() * 100


def binned_rate(df, col, q=8):
    """Split a measurement into equal-sized groups (quantile bins) and give the % of rainy-tomorrow days in each."""
    d = df[[col, "RainTomorrow_bin"]].dropna()
    if d[col].nunique() < 3:
        return pd.DataFrame()
    d["bin"] = pd.qcut(d[col], q=min(q, d[col].nunique()), duplicates="drop")
    g = d.groupby("bin", observed=True).agg(rate=("RainTomorrow_bin", "mean"), n=("RainTomorrow_bin", "size"), mid=(col, "median"))
    g["rate"] *= 100
    return g.reset_index()


def wilson_ci(k, n, z=1.96):
    """95% confidence interval for a proportion (Wilson score)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def describe(s):
    """Mean, median, mode, variance, std, quartiles and outliers (1.5 x IQR rule) of one variable."""
    s = s.dropna()
    if s.empty:
        return {}
    q1, q3 = s.quantile(.25), s.quantile(.75)
    iqr = q3 - q1
    out = ((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr))
    mode = s.round(1).mode()
    return {"Count": len(s), "Mean": s.mean(), "Median": s.median(), "Mode": mode.iloc[0] if len(mode) else np.nan,
            "Variance": s.var(), "Std dev": s.std(), "Min": s.min(), "Q1 (25%)": q1, "Q3 (75%)": q3, "Max": s.max(),
            "IQR": iqr, "Skewness": stats.skew(s), "Outliers (n)": int(out.sum()), "Outliers (%)": out.mean() * 100}


def compare_groups(df, col):
    """Rainy-tomorrow vs dry-tomorrow: medians, Welch t-test, Mann-Whitney U and a 95% CI for the difference in means."""
    a = df.loc[df.RainTomorrow_bin == 1, col].dropna()
    b = df.loc[df.RainTomorrow_bin == 0, col].dropna()
    if len(a) < 30 or len(b) < 30:
        return None
    t = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    pooled = np.sqrt((a.var() + b.var()) / 2)
    d = df[[col, "RainTomorrow_bin"]].dropna()
    return {"rain_mean": a.mean(), "dry_mean": b.mean(), "rain_median": a.median(), "dry_median": b.median(), "diff": diff,
            "ci": (diff - 1.96 * se, diff + 1.96 * se), "t_p": t.pvalue, "u_p": u.pvalue, "cohen_d": diff / pooled if pooled else np.nan,
            "r_pb": d[col].corr(d.RainTomorrow_bin), "n_rain": len(a), "n_dry": len(b)}


def fmt_p(p):
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def facts(df):
    """'Did you know?' discoveries calculated from the loaded data. Returns [(icon, headline, detail)]."""
    out = []
    n = len(df)
    base = df.RainTomorrow_bin.mean() * 100
    out.append(("📅", f"{n:,} daily records from {df.Location.nunique()} weather stations",
                f"{df.Date.min():%b %Y} to {df.Date.max():%b %Y}. Rain followed on {base:.0f}% of days, so dry days are the norm."))
    loc = rain_rate(df, "Location")[df.groupby("Location").size() > 500].sort_values()
    if len(loc) >= 2:
        out.append(("🏙️", f"{loc.index[-1]} is the rainiest station; {loc.index[0]} the driest",
                    f"Rain followed on {loc.iloc[-1]:.0f}% of days at {loc.index[-1]} versus {loc.iloc[0]:.0f}% at {loc.index[0]}."))
    mon = rain_rate(df, "Month")
    if len(mon) >= 2:
        out.append(("🗓️", f"{MONTHS[int(mon.idxmax()) - 1]} is the wettest month on average, {MONTHS[int(mon.idxmin()) - 1]} the driest",
                    f"{mon.max():.0f}% vs {mon.min():.0f}% of days were followed by rain (all stations combined; it varies by region)."))
    h = df.dropna(subset=["Humidity3pm"])
    hi, lo = h[h.Humidity3pm >= 70].RainTomorrow_bin.mean() * 100, h[h.Humidity3pm < 40].RainTomorrow_bin.mean() * 100
    if h[h.Humidity3pm >= 70].shape[0] > 30 and h[h.Humidity3pm < 40].shape[0] > 30:
        out.append(("💧", f"Humid afternoons ({hi:.0f}%) are followed by rain far more often than dry ones ({lo:.0f}%)",
                    "Afternoon humidity of 70% or more versus under 40%. A strong clue, but never a guarantee."))
    rt = df.dropna(subset=["RainToday"])
    y, nn = rt[rt.RainToday == "Yes"].RainTomorrow_bin.mean() * 100, rt[rt.RainToday == "No"].RainTomorrow_bin.mean() * 100
    if len(rt):
        out.append(("🔁", f"Rain tends to come in spells: {y:.0f}% after a rainy day vs {nn:.0f}% after a dry day",
                    "Probability of rain tomorrow given whether it rained today."))
    s = df.dropna(subset=["Sunshine"])
    if (s.Sunshine < 3).sum() > 30 and (s.Sunshine >= 9).sum() > 30:
        out.append(("☀️", f"After gloomy days (under 3 h of sun) rain followed {s[s.Sunshine < 3].RainTomorrow_bin.mean() * 100:.0f}% of the time",
                    f"After very sunny days (9 h or more) it was only {s[s.Sunshine >= 9].RainTomorrow_bin.mean() * 100:.0f}%."))
    return out


def learned(df):
    """'What did we learn?' bullet points for the (filtered) data."""
    pts, base = [], df.RainTomorrow_bin.mean() * 100
    pts.append(f"In this selection, rain followed on **{base:.1f}%** of {len(df):,} days.")
    rows = []
    for c in ["Humidity3pm", "Humidity9am", "Sunshine", "Pressure3pm", "Cloud3pm", "WindGustSpeed", "Rainfall", "MaxTemp", "MinTemp"]:
        d = df[[c, "RainTomorrow_bin"]].dropna()
        if len(d) > 100:
            rows.append((c, d[c].corr(d.RainTomorrow_bin)))
    rows.sort(key=lambda x: -abs(x[1]))
    for c, r in rows[:3]:
        pts.append(f"**{c}** has one of the strongest links with next-day rain (correlation {r:+.2f}): {'higher' if r > 0 else 'lower'} values tend to come before rainy days.")
    mon = rain_rate(df, "Month")
    if len(mon) > 1:
        pts.append(f"Seasonality: the wettest month here is **{MONTHS[int(mon.idxmax()) - 1]}** ({mon.max():.0f}%) and the driest **{MONTHS[int(mon.idxmin()) - 1]}** ({mon.min():.0f}%).")
    pts.append("These are **associations** in historical data. They help prediction but do not prove that one measurement *causes* rain.")
    return pts
