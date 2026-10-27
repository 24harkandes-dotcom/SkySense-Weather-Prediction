"""Weather Insights: filters + interactive exploration of the real data, and a Statistics Lab."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from skysense import analysis, content, data, theme

VIEWS = ["Where it rains", "Rainy vs dry days", "Seasons", "Temperature & humidity", "Pressure, sun & wind", "Correlations", "Distributions & outliers", "Statistics lab"]
NUM = list(content.UNITS)
OUT_COLORS = {"Rain tomorrow": theme.RAIN_C, "No rain tomorrow": theme.DRY_C}


def _var(label_, key, default="Humidity3pm"):
    return st.selectbox(label_, NUM, index=NUM.index(default), format_func=content.label, key=key)


def _where(d):
    r = analysis.rain_rate(d, "Location")[d.groupby("Location").size() > 100].sort_values()
    fig, ax = plt.subplots(figsize=(7, max(3, .3 * len(r))))
    ax.barh(r.index, r.values, color=theme.BLUE)
    ax.axvline(d.RainTomorrow_bin.mean() * 100, color=theme.AMBER, ls="--", label="all selected stations")
    ax.set_xlabel("% of days followed by rain"); ax.set_title("Rain-tomorrow frequency by weather station"); ax.legend()
    theme.show(fig); theme.what_it_shows("how often a day at each station was followed by rain. Stations differ a lot, so location matters.")


def _rvd(d):
    v = _var("Measurement", "ins_rvd")
    x = d[[v, "Outcome"]].dropna()
    hi = x[v].quantile(.995); lo = x[v].quantile(.005)
    bins = np.linspace(lo, hi, 36)
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        for k, col in OUT_COLORS.items():
            ax.hist(x.loc[x.Outcome == k, v].clip(lo, hi), bins=bins, density=True, alpha=.55, color=col, label=k)
        ax.set_xlabel(content.axis_label(v)); ax.set_ylabel("share of days (density)"); ax.set_title(f"{content.label(v)} before rainy and dry days"); ax.legend()
        theme.show(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(4, 3.8))
        bp = ax.boxplot([x.loc[x.Outcome == k, v] for k in OUT_COLORS], patch_artist=True, showfliers=False, medianprops={"color": theme.NAVY})
        ax.set_xticklabels(["Rain\ntomorrow", "No rain\ntomorrow"])
        for p, col in zip(bp["boxes"], OUT_COLORS.values()): p.set_facecolor(col); p.set_alpha(.6)
        ax.set_ylabel(content.axis_label(v)); ax.set_title("Box plot (no outlier dots)")
        theme.show(fig)
    m = x.groupby("Outcome")[v].median()
    theme.what_it_shows(f"the same measurement on days followed by rain (blue) or no rain (amber). Median: {m.get('Rain tomorrow', np.nan):.1f} vs {m.get('No rain tomorrow', np.nan):.1f} {content.UNITS[v]}. More separation means a more useful clue.")


def _seasons(d):
    opt = {"% of days followed by rain": ("RainTomorrow_bin", 100), "Average rain recorded that day (mm)": ("Rainfall", 1), "Average maximum temperature (°C)": ("MaxTemp", 1), "Average afternoon humidity (%)": ("Humidity3pm", 1)}
    k = st.radio("Show", list(opt), horizontal=True, key="ins_season")
    col, mult = opt[k]
    g = d.groupby("Month")[col].mean().reindex(range(1, 13)) * mult
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar(analysis.MONTHS, g.values, color=theme.BLUE); ax.set_ylabel(k); ax.set_title(f"{k} by month")
    theme.show(fig)
    theme.what_it_shows("the yearly cycle for the selected stations. Southern-hemisphere seasons are reversed relative to the north, and the pattern differs between tropical and temperate regions: try filtering a single station.")


def _temp_hum(d):
    s = d[["MaxTemp", "Humidity3pm", "Outcome"]].dropna()
    s = s.sample(min(4000, len(s)), random_state=1)
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for k, col in OUT_COLORS.items():
        z = s[s.Outcome == k]; ax.scatter(z.MaxTemp, z.Humidity3pm, s=9, alpha=.35, color=col, label=k)
    ax.set_xlabel("Maximum temperature (°C)"); ax.set_ylabel("Afternoon humidity (%)"); ax.set_title("Temperature vs humidity, coloured by what happened next"); ax.legend(markerscale=2)
    theme.show(fig)
    theme.what_it_shows(f"a random sample of {len(s):,} days. Blue (rain next day) clusters toward higher humidity at any temperature; temperature alone separates the colours poorly.")


def _pss(d):
    t1, t2, t3 = st.tabs(["Pressure", "Sunshine", "Wind direction"])
    for tab, v in ((t1, "Pressure3pm"), (t2, "Sunshine")):
        with tab:
            g = analysis.binned_rate(d, v)
            if g.empty: st.info("Not enough data."); continue
            fig, ax = plt.subplots(figsize=(6.5, 3.5))
            ax.plot(g.mid, g.rate, marker="o", color=theme.BLUE, lw=2.4); ax.set_xlabel(content.axis_label(v)); ax.set_ylabel("% of days followed by rain"); ax.set_title(f"Rain-tomorrow frequency vs {content.label(v).lower()}")
            theme.show(fig); theme.what_it_shows(f"days were split into equal-sized groups by {content.label(v).lower()}; each dot is the share of rainy-tomorrow days in a group.")
    with t3:
        w = d.dropna(subset=["WindGustDir"]); r = analysis.rain_rate(w, "WindGustDir").reindex(analysis.COMPASS)
        ang = np.linspace(0, 2 * np.pi, 16, endpoint=False)
        fig = plt.figure(figsize=(5.2, 5.2)); ax = fig.add_subplot(111, projection="polar")
        ax.bar(ang, r.fillna(0).values, width=2 * np.pi / 16 * .85, color=theme.BLUE, alpha=.8); ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        ax.set_xticks(ang); ax.set_xticklabels(analysis.COMPASS, fontsize=8); ax.set_title("% of days followed by rain, by gust direction", pad=18)
        theme.show(fig); theme.what_it_shows("each spoke is the direction the strongest gust blew FROM; longer = rain followed more often. Meaning depends on where the coast is.")


def _corr(d):
    m = st.radio("Method", ["Pearson (straight-line)", "Spearman (rank-based)"], horizontal=True, key="ins_corr")
    x = d[NUM + ["RainTomorrow_bin"]].rename(columns=lambda c: "RainTomorrow" if c == "RainTomorrow_bin" else c)
    if len(x) > 30000: x = x.sample(30000, random_state=1)
    c = x.corr(method="pearson" if m.startswith("P") else "spearman")
    fig, ax = plt.subplots(figsize=(9, 7))
    import seaborn as sns
    sns.heatmap(c, mask=np.triu(np.ones_like(c, dtype=bool), 1), cmap="RdBu_r", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f", annot_kws={"size": 7}, linewidths=.4, cbar_kws={"label": "correlation"}, ax=ax)
    ax.set_title("Correlation heatmap"); ax.grid(False)
    theme.show(fig)
    theme.what_it_shows("how strongly pairs of measurements move together (+1 together, −1 opposite, 0 unrelated). Look at the bottom row for links to next-day rain. **Correlation is not causation.**")


def _dist(d):
    v = _var("Measurement", "ins_dist")
    s = d[v].dropna(); st_ = analysis.describe(s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5), gridspec_kw={"height_ratios": [3, 1]}, sharex=True)
    a1.hist(s, bins=40, color=theme.BLUE, alpha=.8); a1.axvline(st_["Mean"], color=theme.AMBER, lw=2, label=f"mean {st_['Mean']:.1f}"); a1.axvline(st_["Median"], color=theme.NAVY, ls="--", lw=2, label=f"median {st_['Median']:.1f}")
    a1.set_ylabel("number of days"); a1.set_title(f"Distribution of {content.label(v).lower()}"); a1.legend()
    a2.boxplot(s, vert=False, widths=.6, flierprops={"markersize": 2, "alpha": .3}); a2.set_yticks([]); a2.set_xlabel(content.axis_label(v)); a2.grid(False)
    theme.show(fig)
    theme.what_it_shows(f"how {content.label(v).lower()} is spread. {st_['Outliers (%)']:.1f}% of values are outliers by the 1.5×IQR rule; when mean and median differ a lot the data is skewed.")


def _lab(d):
    st.markdown("Hands-on versions of the statistics practicals. The full calculations live in the notebook.")
    t = st.tabs(["Describe", "Probability", "Central Limit Theorem", "Hypothesis test"])
    with t[0]:
        v = _var("Measurement", "lab_desc", "Rainfall")
        s = analysis.describe(d[v])
        if s: st.dataframe(pd.DataFrame(s, index=[content.UNITS[v]]).T.style.format("{:,.2f}"))
        st.caption("Mean, median, mode, variance, standard deviation, quartiles, IQR and outliers (values beyond 1.5×IQR from the quartiles).")
    with t[1]:
        n, k = len(d), int(d.RainTomorrow_bin.sum()); p = k / n; lo, hi = analysis.wilson_ci(k, n)
        st.markdown(f"**P(rain tomorrow) = {p:.1%}**, 95% confidence interval {lo:.1%} to {hi:.1%} (n = {n:,}).")
        rt = d.dropna(subset=["RainToday"]); a = rt[rt.RainToday == "Yes"].RainTomorrow_bin; b = rt[rt.RainToday == "No"].RainTomorrow_bin
        if len(a) > 30 and len(b) > 30:
            st.markdown(f"**Conditional probability:** P(rain tomorrow | rain today) = {a.mean():.1%}, versus P(rain tomorrow | dry today) = {b.mean():.1%}.")
        N = st.slider("Days in a month (n)", 7, 31, 30, key="lab_n"); xs = np.arange(N + 1)
        fig, ax = plt.subplots(figsize=(6.5, 3.2)); ax.bar(xs, stats.binom.pmf(xs, N, p), color=theme.BLUE); ax.set_xlabel(f"rainy-tomorrow days in {N} days"); ax.set_ylabel("probability"); ax.set_title(f"Binomial model (n={N}, p={p:.2f})")
        theme.show(fig)
        theme.what_it_shows("what a month would look like if each day were an independent coin flip with the observed rain chance. Real rain comes in spells (see the conditional probabilities), so real months are more clustered than this model suggests.")
    with t[2]:
        v = _var("Population variable", "lab_clt", "Rainfall"); pop = d[v].dropna().values
        n = st.slider("Sample size n", 1, 200, 30, key="lab_clt_n")
        rng = np.random.default_rng(0); means = rng.choice(pop, (1000, n)).mean(axis=1)
        mu, sd = pop.mean(), pop.std(ddof=1)
        fig, ax = plt.subplots(figsize=(6.8, 3.4)); ax.hist(means, bins=40, density=True, color=theme.SKY, alpha=.8, label="1000 sample means")
        xs = np.linspace(means.min(), means.max(), 200); ax.plot(xs, stats.norm.pdf(xs, mu, sd / np.sqrt(n)), color=theme.NAVY, lw=2, label="Normal curve predicted by CLT")
        ax.set_xlabel(f"sample mean of {content.label(v).lower()} ({content.UNITS[v]})"); ax.set_title(f"Distribution of sample means (n = {n})"); ax.legend()
        theme.show(fig)
        theme.what_it_shows(f"even though the raw data is skewed (skewness {stats.skew(pop):.1f}), means of larger samples look bell-shaped. Standard deviation of the means: {means.std(ddof=1):.3f} vs theory σ/√n = {sd / np.sqrt(n):.3f}.")
    with t[3]:
        v = _var("Compare this measurement", "lab_ht", "Humidity3pm"); r = analysis.compare_groups(d, v)
        if r is None: st.info("Not enough data."); return
        st.markdown(f"**H₀:** the average {content.label(v).lower()} is the same before rainy and dry days. **H₁:** it differs.")
        c = st.columns(3); c[0].metric("Mean before rain", f"{r['rain_mean']:.1f}"); c[1].metric("Mean before no rain", f"{r['dry_mean']:.1f}"); c[2].metric("Difference (95% CI)", f"{r['diff']:+.1f}", f"{r['ci'][0]:+.1f} to {r['ci'][1]:+.1f}", delta_color="off")
        st.markdown(f"Welch t-test p-value **{analysis.fmt_p(r['t_p'])}**, Mann-Whitney U p-value **{analysis.fmt_p(r['u_p'])}**, effect size (Cohen's d) **{r['cohen_d']:+.2f}**.")
        st.markdown("**Reading it:** " + (f"the difference is statistically significant at the 5% level" if r["t_p"] < .05 else "no significant difference at the 5% level") +
                    f"; the effect size says how *big* it is ({'small' if abs(r['cohen_d']) < .5 else 'medium' if abs(r['cohen_d']) < .8 else 'large'}).")
        theme.note("Assumptions: with thousands of days the t-test is robust (CLT), and Mann-Whitney needs no normality. But consecutive days are not independent, so real p-values are less certain than shown. With big data tiny p-values are easy to get, so look at the effect size too.", "info")


def render():
    theme.title("📊 Weather Insights", "Explore the real dataset. Pick filters, then pick what to look at.")
    if not data.require(data.DATA_FILES, "The insights dataset"):
        return
    df = data.load_weather()
    with st.container(border=True):
        c1, c2 = st.columns([3, 2])
        locs = c1.multiselect("Weather stations (leave empty for all)", sorted(df.Location.unique()), key="ins_locs")
        y0, y1 = int(df.Year.min()), int(df.Year.max())
        yrs = c2.slider("Years", y0, y1, (y0, y1), key="ins_years")
    d = df[(df.Year >= yrs[0]) & (df.Year <= yrs[1])]
    if locs: d = d[d.Location.isin(locs)]
    if len(d) < 200:
        st.warning("That selection has too few days to analyse. Widen the filters."); return
    st.caption(f"{len(d):,} days · {d.Location.nunique()} station(s) · {yrs[0]}–{yrs[1]}")
    view = st.radio("What would you like to explore?", VIEWS, horizontal=True, key="ins_view")
    {"Where it rains": _where, "Rainy vs dry days": _rvd, "Seasons": _seasons, "Temperature & humidity": _temp_hum, "Pressure, sun & wind": _pss,
     "Correlations": _corr, "Distributions & outliers": _dist, "Statistics lab": _lab}[view](d)
    theme.title("What did we learn?", "Computed from your current selection.")
    for p in analysis.learned(d): st.markdown(f"- {p}")
