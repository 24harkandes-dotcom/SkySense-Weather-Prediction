CELLS_B = []
def md(t): CELLS_B.append(("md", t))
def code(t, kind="code"): CELLS_B.append((kind, t.strip("\n")))

md("## 7. Sampling techniques (population = all labelled rows)")
code("""N, n = len(df), 2000
srs = df.sample(n=n, random_state=SEED)                                            # simple random
step = N // n; start = int(rng.integers(0, step)); systematic = df.iloc[start::step]  # systematic: every k-th row
stratified = df.groupby("Location", group_keys=False).sample(frac=n/N, random_state=SEED)  # proportional strata = Location
clusters = rng.choice(df.Location.unique(), size=5, replace=False)                 # location-based cluster sampling:
cluster = df[df.Location.isin(clusters)]                                            # whole clusters (stations) are selected
print(f"Population N={N} | SRS n={len(srs)} | systematic (k={step}, start={start}) n={len(systematic)} | stratified n={len(stratified)} | cluster n={len(cluster)}")
print("Selected clusters:", list(clusters))""")

md("## 8. Sample vs population statistics")
code("""vars_ = ["MaxTemp", "Humidity3pm", "Rainfall", "RainTomorrow_bin"]
groups = {"Population": df, "Simple random": srs, "Systematic": systematic, "Stratified": stratified, "Cluster": cluster}
tab = pd.DataFrame({name: [d[v].mean() for v in vars_] + [d[v].std() for v in vars_] + [len(d)] for name, d in groups.items()},
                   index=[f"mean {v}" for v in vars_] + [f"std {v}" for v in vars_] + ["n"]).T
display(tab.round(3))
err = pd.DataFrame({name: [abs(d[v].mean() - df[v].mean()) / abs(df[v].mean()) * 100 for v in vars_] for name, d in list(groups.items())[1:]}, index=vars_)
print("\\nAbsolute % error of the sample mean relative to the population mean:"); display(err.round(2))
err.T.plot.bar(figsize=(10, 4.5), rot=0); plt.ylabel("% error of mean"); plt.title("Sampling methods: error vs population")
save_fig("11_sampling_error")""")

md("## 9. Central Limit Theorem (population = Rainfall, strongly right-skewed)")
code("""pop = df["Rainfall"].dropna().values; mu, sigma = pop.mean(), pop.std()
print(f"Population: N={len(pop)}, mean={mu:.3f}, sd={sigma:.3f}, skew={stats.skew(pop):.2f}")
sizes, reps, rows = [10, 30, 50, 100], 3000, []
fig, ax = plt.subplots(2, 3, figsize=(16, 8)); ax = ax.ravel()
ax[0].hist(np.clip(pop, 0, np.percentile(pop, 99)), bins=50, color="gray"); ax[0].set_title("Population Rainfall (clipped at 99th pct)")
for a, s in zip(ax[1:5], sizes):
    means = rng.choice(pop, size=(reps, s)).mean(axis=1)                      # `reps` samples of size s
    se_emp, se_th = means.std(ddof=1), sigma / np.sqrt(s)
    rows.append({"n": s, "mean of sample means": means.mean(), "empirical SE": se_emp, "theoretical SE (sigma/sqrt n)": se_th, "skew of sampling dist": stats.skew(means)})
    sns.histplot(means, bins=40, stat="density", kde=True, ax=a, color="skyblue")
    xs = np.linspace(means.min(), means.max(), 200); a.plot(xs, stats.norm.pdf(xs, mu, se_th), "r"); a.set_title(f"Sample means, n={s}")
clt = pd.DataFrame(rows); display(clt.round(4))
ax[5].plot(clt.n, clt["empirical SE"], "o-", label="empirical"); ax[5].plot(clt.n, clt["theoretical SE (sigma/sqrt n)"], "s--", label="sigma/sqrt(n)")
ax[5].set_xlabel("sample size n"); ax[5].set_ylabel("standard error"); ax[5].legend(); ax[5].set_title("Standard error shrinks with n")
save_fig("12_central_limit_theorem")""")

md("## 10. Covariance, Pearson and Spearman correlation")
code("""pairs = [("MinTemp", "MaxTemp"), ("Humidity9am", "Humidity3pm"), ("Temp3pm", "Humidity3pm"), ("Pressure3pm", "Humidity3pm"), ("Sunshine", "Humidity3pm"), ("Pressure9am", "Pressure3pm")]
rows = []
for a, b in pairs:
    d = df[[a, b]].dropna(); pr = stats.pearsonr(d[a], d[b]); sp = stats.spearmanr(d[a], d[b])
    rows.append({"pair": f"{a} vs {b}", "n": len(d), "covariance": d[a].cov(d[b]), "Pearson r": pr[0], "Pearson p": pr[1], "Spearman rho": sp[0], "Spearman p": sp[1]})
corr_tab = pd.DataFrame(rows); display(corr_tab.round(4))
print("p-values printed as 0 are below floating-point precision (very large n).")
fig, ax = plt.subplots(2, 3, figsize=(16, 9)); samp = df.sample(5000, random_state=SEED)
for i, (a_, (a, b)) in enumerate(zip(ax.ravel(), pairs)):
    sns.regplot(data=samp, x=a, y=b, ax=a_, scatter_kws={"s": 6, "alpha": .3}, line_kws={"color": "red"})
    a_.set_title(f"{a} vs {b}  (r={corr_tab['Pearson r'].iloc[i]:.2f})")
save_fig("13_correlation_scatter")""")

md("## 11. Confidence intervals and hypothesis tests (alpha = 0.05)")
code("""ALPHA = 0.05
def report(title, H0, H1, stat_name, stat, p, extra=""):
    dec = "Reject H0" if p < ALPHA else "Fail to reject H0"
    interp = "The data give statistically significant evidence for H1." if p < ALPHA else "There is not enough evidence to support H1."
    print(f"\\n=== {title} ===\\nH0: {H0}\\nH1: {H1}\\n{stat_name} = {stat:.4f} | p-value = {p:.4g} | alpha = {ALPHA}\\nDecision: {dec}\\nInterpretation: {interp} {extra}")

# ---- Confidence intervals ----
v = df["MaxTemp"].dropna(); s100 = v.sample(100, random_state=SEED)
ci100 = stats.t.interval(0.95, len(s100)-1, loc=s100.mean(), scale=stats.sem(s100))
ci_all = stats.t.interval(0.95, len(v)-1, loc=v.mean(), scale=stats.sem(v))
print(f"95% CI for mean MaxTemp, sample n=100 : ({ci100[0]:.3f}, {ci100[1]:.3f}) | contains population mean {v.mean():.3f}? {ci100[0] <= v.mean() <= ci100[1]}")
print(f"95% CI for mean MaxTemp, all n={len(v)}: ({ci_all[0]:.3f}, {ci_all[1]:.3f})  <- narrower: SE falls with n")
pr_, nn = df.RainTomorrow_bin.mean(), len(df); se_p = np.sqrt(pr_*(1-pr_)/nn)
print(f"95% CI for P(RainTomorrow=Yes): ({pr_-1.96*se_p:.4f}, {pr_+1.96*se_p:.4f}) (point estimate {pr_:.4f})")""")
code("""# ---- One-sample t-test: Sydney MaxTemp vs benchmark mu0 = 22 C (benchmark chosen by us for demonstration) ----
syd = df[df.Location == "Sydney"].MaxTemp.dropna(); mu0 = 22
print(f"Assumption check (Sydney MaxTemp, n={len(syd)}): skew={stats.skew(syd):.2f}; n is large, so the t-test is robust (CLT)")
t = stats.ttest_1samp(syd, mu0)
report("One-sample t-test", f"mean Sydney MaxTemp = {mu0}", f"mean Sydney MaxTemp != {mu0}", "t", t.statistic, t.pvalue, f"(sample mean = {syd.mean():.3f})")

# ---- Independent (Welch) t-test: Humidity3pm, rainy tomorrow vs dry tomorrow ----
h_yes = df[df.RainTomorrow == "Yes"].Humidity3pm.dropna(); h_no = df[df.RainTomorrow == "No"].Humidity3pm.dropna()
lev = stats.levene(h_yes, h_no); print(f"\\nLevene equal-variance test: W={lev.statistic:.2f}, p={lev.pvalue:.3g} -> Welch's t-test (no equal-variance assumption) is used")
t = stats.ttest_ind(h_yes, h_no, equal_var=False)
sp_ = np.sqrt(((len(h_yes)-1)*h_yes.var() + (len(h_no)-1)*h_no.var()) / (len(h_yes)+len(h_no)-2)); d_ = (h_yes.mean() - h_no.mean()) / sp_
mw = stats.mannwhitneyu(h_yes, h_no)
report("Independent t-test (Welch)", "mean Humidity3pm (rain tomorrow) = mean Humidity3pm (no rain)", "the two means differ", "t", t.statistic, t.pvalue,
       f"(means {h_yes.mean():.2f} vs {h_no.mean():.2f}; Cohen's d = {d_:.2f}; Mann-Whitney p = {mw.pvalue:.3g})")""")
code("""# ---- Paired t-test: Temp9am vs Temp3pm on the same day/station ----
pt = df[["Temp9am", "Temp3pm"]].dropna(); diff = pt.Temp3pm - pt.Temp9am
sh = stats.shapiro(diff.sample(5000, random_state=SEED))
print(f"Assumption check (differences, n={len(diff)}): skew={stats.skew(diff):.2f}, Shapiro p (5000 sample)={sh.pvalue:.3g}; large n -> t-test still reasonable")
t = stats.ttest_rel(pt.Temp3pm, pt.Temp9am); wil = stats.wilcoxon(diff.sample(5000, random_state=SEED))
report("Paired t-test", "mean(Temp3pm - Temp9am) = 0", "mean(Temp3pm - Temp9am) != 0", "t", t.statistic, t.pvalue, f"(mean difference = {diff.mean():.3f} C; Wilcoxon p = {wil.pvalue:.3g})")

# ---- One-way ANOVA: Humidity3pm across five capital cities ----
want = ["Sydney", "Melbourne", "Brisbane", "Perth", "Adelaide"]
cities = [c for c in want if c in set(df.Location)] or list(df.Location.value_counts().index[:5])
grp = [df[df.Location == c].Humidity3pm.dropna() for c in cities]
print(f"\\nCities: {cities}, group sizes: {[len(g) for g in grp]}")
print("Shapiro p per city (5000-sample):", [f"{stats.shapiro(g.sample(min(5000, len(g)), random_state=SEED)).pvalue:.2g}" for g in grp])
lev = stats.levene(*grp); print(f"Levene: W={lev.statistic:.2f}, p={lev.pvalue:.3g} (small p = unequal variances; Kruskal-Wallis is reported as a robust check)")
f = stats.f_oneway(*grp); kw = stats.kruskal(*grp)
report("One-way ANOVA", "all city means of Humidity3pm are equal", "at least one city mean differs", "F", f.statistic, f.pvalue, f"(Kruskal-Wallis p = {kw.pvalue:.3g})")
if f.pvalue < ALPHA: print("\\nTukey HSD post-hoc:"); print(stats.tukey_hsd(*grp))""")
code("""# ---- Chi-square test of independence: RainToday vs RainTomorrow ----
ct = pd.crosstab(df.RainToday, df.RainTomorrow); chi2, p, dof, exp = stats.chi2_contingency(ct)
cramers = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
print("Contingency table:"); display(ct); print(f"Assumption: smallest expected count = {exp.min():.1f} (>= 5 required) -> satisfied: {exp.min() >= 5}")
report("Chi-square test of independence", "RainToday and RainTomorrow are independent", "RainToday and RainTomorrow are associated", "chi2", chi2, p, f"(dof = {dof}; Cramer's V = {cramers:.3f})")
print("P(rain tomorrow | rain today) =", round(ct.loc["Yes", "Yes"] / ct.loc["Yes"].sum(), 3), "| P(rain tomorrow | no rain today) =", round(ct.loc["No", "Yes"] / ct.loc["No"].sum(), 3))""")
