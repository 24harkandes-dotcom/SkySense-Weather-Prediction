"""Rainfall Factors: what goes with rain in the data, and how the model uses the inputs."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from skysense import analysis, content, data, theme

OPTIONS = list(content.FACTORS) + ["Location & season"]


def _bin_chart(d, v):
    g = analysis.binned_rate(d, v)
    if g.empty:
        st.info("Not enough data for this measurement."); return None
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.plot(g.mid, g.rate, marker="o", lw=2.6, color=theme.BLUE); ax.axhline(d.RainTomorrow_bin.mean() * 100, color=theme.AMBER, ls="--", label="overall average")
    ax.set_ylim(0, max(g.rate.max() * 1.15, 10)); ax.set_xlabel(content.axis_label(v)); ax.set_ylabel("% of days followed by rain"); ax.set_title(f"Chance of rain tomorrow vs {content.label(v).lower()}"); ax.legend()
    theme.show(fig)
    return g


def _dist_chart(d, v):
    x = d[[v, "Outcome"]].dropna(); lo, hi = x[v].quantile(.005), x[v].quantile(.995); bins = np.linspace(lo, hi, 32)
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    for k, col in (("Rain tomorrow", theme.RAIN_C), ("No rain tomorrow", theme.DRY_C)):
        ax.hist(x.loc[x.Outcome == k, v].clip(lo, hi), bins=bins, density=True, alpha=.55, color=col, label=k)
    ax.set_xlabel(content.axis_label(v)); ax.set_ylabel("density"); ax.set_title("Rainy-tomorrow vs dry-tomorrow days"); ax.legend()
    theme.show(fig)


def _loc_season(d):
    top = d.groupby("Location").size().sort_values(ascending=False).head(14).index
    t = d[d.Location.isin(top)].pivot_table(index="Location", columns="Month", values="RainTomorrow_bin", aggfunc="mean") * 100
    t = t.reindex(columns=range(1, 13)); t = t.loc[t.mean(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.heatmap(t, cmap="Blues", annot=True, fmt=".0f", annot_kws={"size": 7}, xticklabels=analysis.MONTHS, cbar_kws={"label": "% of days followed by rain"}, ax=ax)
    ax.set_title("Rain-tomorrow frequency by station and month"); ax.set_xlabel(""); ax.grid(False)
    theme.show(fig); theme.what_it_shows("darker = rain follows more often. Each station has its own wet and dry season, so location and time of year are both strong clues.")


def _importance(report):
    theme.title("How does the model use these measurements?", "Permutation importance on the held-out test years.")
    pi = report.get("permutation_importance")
    if not pi:
        theme.note("Re-run the notebook (section 17c) to create the feature-importance data.", "info"); return
    s = pd.DataFrame(pi["values"]).T.sort_values("mean").tail(12)
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.barh([content.label(k) for k in s.index], s["mean"], xerr=s["std"], color=theme.BLUE, error_kw={"ecolor": theme.NAVY, "lw": 1})
    ax.set_xlabel("drop in ROC-AUC when this input is shuffled\n(bigger = the model relies on it more)"); ax.set_title("Most influential inputs for the saved model")
    theme.show(fig)
    theme.what_it_shows("we scramble one input at a time and see how much the model gets worse. A big drop means the model leans on that input. It describes the model's behaviour, **not** physical causation, and correlated inputs (e.g. 9am and 3pm humidity) can share credit.")
    with st.expander("🎓 Advanced: how permutation importance works"):
        st.markdown(f"Computed on {pi['rows_used']:,} test rows; scoring = test ROC-AUC; 3 repeats, with the bar showing mean ± standard deviation across repeats. "
                    "It is model-agnostic, so it works for any of the six models. Impurity-based importances from trees are biased toward high-cardinality features, which is why we do not use them for comparison.")


def render():
    theme.title("🔬 Rainfall Factors", "What influences rain? Real patterns from the data, explained simply.")
    if not data.require(data.DATA_FILES, "The insights dataset"):
        return
    df = data.load_weather()
    pick = st.radio("Choose a factor", OPTIONS, horizontal=True, key="factor_pick", format_func=lambda k: f"{content.FACTORS[k]['icon']} {k}" if k in content.FACTORS else "🗺️ Location & season")
    if pick == "Location & season":
        st.markdown(theme.card("🗺️", "Location & season", "Australia spans tropical monsoon, desert and temperate climates. The same humidity or pressure means different things in Darwin and Hobart, and the wet season shifts through the year."), unsafe_allow_html=True)
        _loc_season(df)
    else:
        f = content.FACTORS[pick]
        st.markdown(theme.card(f["icon"], pick, f["text"]), unsafe_allow_html=True)
        v = st.selectbox("Measurement", f["vars"], format_func=content.label, key=f"factor_var_{pick}") if len(f["vars"]) > 1 else f["vars"][0]
        if v not in f["vars"]: v = f["main"]
        c1, c2 = st.columns(2)
        with c1: g = _bin_chart(df, v)
        with c2: _dist_chart(df, v)
        if g is not None:
            theme.what_it_shows(f"left: days grouped by {content.label(v).lower()} and how often rain followed in each group. Right: the spread of values before rainy and dry days. "
                                f"Even in the most rain-prone group only **{g.rate.max():.0f}%** of days were followed by rain, so no single measurement guarantees rain.")
        if pick == "Recent rainfall":
            r = df.dropna(subset=["RainToday"]); rate = analysis.rain_rate(r, "RainToday")
            st.markdown(f"After a rainy day ({content.UNITS['Rainfall']} > 1 mm) rain followed on **{rate.get('Yes', np.nan):.0f}%** of days, versus **{rate.get('No', np.nan):.0f}%** after a dry day.")
        with st.expander("🎓 Advanced explanation for students"):
            st.markdown(f["advanced"])
            r = analysis.compare_groups(df, v)
            if r:
                st.markdown(f"**{content.label(v)}**: median before rain {r['rain_median']:.1f} vs {r['dry_median']:.1f} {content.UNITS[v]} before no rain. Welch t-test p {analysis.fmt_p(r['t_p'])}, "
                            f"Mann-Whitney p {analysis.fmt_p(r['u_p'])}, Cohen's d {r['cohen_d']:+.2f}, point-biserial correlation {r['r_pb']:+.2f}.")
    if not data.missing(["model_report.json"]):
        _importance(data.read_json("model_report.json"))
