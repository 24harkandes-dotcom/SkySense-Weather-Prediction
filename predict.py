"""Rainfall Predictor: grouped inputs -> result card -> explanations -> scenario explorer."""
import math
from datetime import date

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import predictor
from skysense import analysis, content, data, theme

GROUPS = [("📍 Where & when", ["Location", "Month", "RainToday"]),
          ("🌡️ Temperature", ["MinTemp", "MaxTemp", "Temp9am", "Temp3pm"]),
          ("💧 Humidity & clouds", ["Humidity9am", "Humidity3pm", "Cloud9am", "Cloud3pm"]),
          ("🧭 Pressure & sunshine", ["Pressure9am", "Pressure3pm", "Sunshine", "Evaporation"]),
          ("🌬️ Wind", ["WindGustSpeed", "WindSpeed9am", "WindSpeed3pm", "WindGustDir", "WindDir9am", "WindDir3pm"]),
          ("🌧️ Recent rainfall", ["Rainfall"])]
STEPS = {"°C": .5, "mm": .2, "hours": .1, "km/h": 1, "%": 1, "hPa": .1, "oktas (0-9)": 1}
KEY_VARS = ["Humidity3pm", "Pressure3pm", "Sunshine", "Cloud3pm", "WindGustSpeed", "Rainfall"]
NOT_SURE = predictor.UNKNOWN


def _range(f, schema, prof):
    """Slider range from the data: about the 1st-99th percentile plus a 25% margin, never beyond what was seen in training."""
    n, p = schema["numeric"][f], prof.get(f, {})
    step = STEPS.get(n["unit"], .5)
    lo, hi = p.get("p1", n["train_min"]), p.get("p99", n["train_max"])
    pad = .25 * (hi - lo)
    lo, hi = max(n["train_min"], lo - pad), min(n["train_max"], hi + pad)
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    if f.startswith("Humidity"): hi = min(hi, 100)
    if f.startswith("Cloud"): lo, hi = 0, 9
    if hi <= lo: hi = lo + step
    default = min(max(round(n["train_median"] / step) * step, lo), hi)
    return lo, hi, step, default


def _default(f, schema, prof):
    if f in schema["numeric_features"]:
        lo, hi, step, d = _range(f, schema, prof)
        return int(d) if step == 1 else float(d)
    cats = schema["categorical"][f]
    if f == "Location": return "Sydney" if "Sydney" in cats else cats[0]
    if f == "Month": return str(date.today().month) if str(date.today().month) in cats else cats[0]
    if f == "RainToday": return "No" if "No" in cats else cats[0]
    return NOT_SURE


def _reset(schema, prof):
    for f in schema["features"]:
        st.session_state[f"in_{f}"] = _default(f, schema, prof)
    st.session_state.pop("last", None)


def _widget(f, schema, prof, col):
    key = f"in_{f}"
    if f in schema["numeric_features"]:
        lo, hi, step, _ = _range(f, schema, prof)
        unit = schema["numeric"][f]["unit"].replace(" (0-9)", "")
        isint = step == 1
        cast = int if isint else float
        return col.slider(f"{content.label(f)} ({unit})", cast(lo), cast(hi), step=cast(step), key=key, help=content.tip(f))
    cats = schema["categorical"][f]
    if f == "Month":
        return col.selectbox(content.label(f), cats, key=key, format_func=lambda m: analysis.MONTHS[int(m) - 1], help=content.tip(f))
    if f.startswith("Wind"):
        return col.selectbox(content.label(f), [NOT_SURE] + cats, key=key, format_func=lambda v: "Not sure" if v == NOT_SURE else v, help=content.tip(f))
    return col.selectbox(content.label(f), cats, key=key, help=content.tip(f))


def _result_card(res, schema, report):
    rain = res["rain"]
    head = "Rain is more likely tomorrow" if rain else "Rain is less likely tomorrow"
    st.markdown(f'<div class="result {"rain" if rain else "dry"}"><span class="icon">{"🌧️" if rain else "🌤️"}</span><div class="big">{head}</div>'
                f'<div class="sub">A machine-learning estimate from the weather you entered. Predicted by <b>{res["model_name"]}</b>, trained on historical Australian data.</div></div>',
                unsafe_allow_html=True)
    if res["probability"] is not None:
        p, thr = res["probability"], res["threshold"]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown(f"**Model rain score: {p:.0%}**  ·  rain is predicted at {thr:.0%} or above")
            st.markdown(f'<div class="meter"><div class="fill" style="width:{p * 100:.0f}%"></div><div class="thr" style="left:{thr * 100:.0f}%"></div></div>'
                        f'<small>0% &nbsp;···&nbsp; orange marker = decision threshold &nbsp;···&nbsp; 100%</small>', unsafe_allow_html=True)
        with c2:
            cal = report.get("calibration")
            if cal:
                b = next((c for c in cal if c["low"] <= p < c["high"] + 1e-9 and c["rain_rate"] is not None), None)
                if b and b["n"] >= 30:
                    st.metric("In past test data", f"{b['rain_rate']:.0%}", help=f"Share of the {b['n']:,} held-out test days with a model score of {b['low']:.0%}-{b['high']:.0%} that were really followed by rain.")
                    st.caption(f"…of {b['n']:,} similar days (score {b['low']:.0%}–{b['high']:.0%}) were followed by rain. This is the honest 'real-world' hit rate for scores like yours.")
        st.caption("The score is a model output, not a guaranteed chance of rain; the model deliberately favours catching rainy days, so scores run higher than raw frequencies. The historical hit rate on the right is the safer number to quote.")
    else:
        st.info(f"{res['model_name']} gives a yes/no decision without a meaningful probability, so no percentage is shown.")


def _plan(res):
    near = res["probability"] is not None and abs(res["probability"] - res["threshold"]) < .08
    if res["rain"]:
        txt = "Carry an umbrella or rain jacket, allow extra travel time and have an indoor plan for outdoor events. Check your official forecast for timing and any warnings."
    else:
        txt = "Rain looks less likely, but it is not ruled out. Keep a light rain layer handy for important plans and check your official forecast."
    if near:
        txt += " <b>This result is close to the decision line, so treat it as a toss-up.</b>"
    st.markdown(f'<div class="card"><h4>🧳 Simple planning suggestion</h4><p>{txt}</p></div>', unsafe_allow_html=True)


def _push_chart(expl, res):
    items = sorted(expl.items(), key=lambda kv: -abs(kv[1]))[:8][::-1]
    scale, unit = (100, "percentage points") if res["probability"] is not None else (1, "decision-score units")
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.barh([content.label(k) for k, _ in items], [v * scale for _, v in items], color=[theme.RAIN_C if v > 0 else theme.DRY_C for _, v in items])
    ax.axvline(0, color=theme.NAVY, lw=1)
    ax.set_xlabel(f"Effect on the rain score ({unit})\n← toward dry  ·  toward rain →")
    ax.set_title("What pushed this prediction?")
    theme.show(fig)
    theme.what_it_shows("for each input, how the score changes if that input were unknown (replaced by a typical value). Blue bars nudge the model toward rain, amber toward dry. This describes how the **model** reacted, not what causes rain.")


def _compare_chart(inputs, prof):
    rows = [k for k in KEY_VARS if k in prof and inputs.get(k) is not None]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    closer_rain = 0
    for i, k in enumerate(rows):
        p, v = prof[k], float(inputs[k])
        span = (p["p99"] - p["p1"]) or 1
        pos = lambda x: (x - p["p1"]) / span
        ax.plot([pos(p["dry_median"]), pos(p["rain_median"])], [i, i], color="#c5d6ea", lw=4, zorder=1)
        ax.scatter(pos(p["dry_median"]), i, s=90, color=theme.DRY_C, zorder=2, label="Typical day before NO rain" if i == 0 else None)
        ax.scatter(pos(p["rain_median"]), i, s=90, color=theme.RAIN_C, zorder=2, label="Typical day before rain" if i == 0 else None)
        ax.scatter(min(max(pos(v), -.08), 1.08), i, s=130, marker="D", color=theme.NAVY, zorder=3, label="Your input" if i == 0 else None)
        ax.text(1.14, i, f"{v:g}", va="center", fontsize=9)
        closer_rain += abs(v - p["rain_median"]) < abs(v - p["dry_median"])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([content.label(k) for k in rows]); ax.set_xlim(-.12, 1.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["low", "high"]); ax.set_xlabel("position within the typical range seen in the data"); ax.legend(loc="lower center", bbox_to_anchor=(.45, 1.0), ncol=3, fontsize=8)
    ax.set_title("How your day compares", pad=28)
    theme.show(fig)
    theme.what_it_shows(f"your inputs (diamonds) against the typical day before rain (blue) and before no rain (amber), from the training years. {closer_rain} of {len(rows)} key measurements look closer to the rainy pattern.")


def _scenario(last, schema, prof):
    theme.title("🎛️ Scenario explorer", "Change just one measurement and watch how the model's score responds.")
    feats = [f for f in schema["numeric_features"]]
    f = st.selectbox("Measurement to vary", feats, index=feats.index("Humidity3pm") if "Humidity3pm" in feats else 0, format_func=content.label, key="scn_feat")
    lo, hi, step, _ = _range(f, schema, prof)
    xs = np.linspace(lo, hi, 40)
    try:
        ys = predictor.score_sweep(last["inputs"], f, xs)
    except Exception as e:
        st.error(f"Could not run the scenario: {e}"); return
    proba = schema["has_predict_proba"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    k = 100 if proba else 1
    ax.plot(xs, ys * k, color=theme.BLUE, lw=2.6)
    ax.axhline(schema["threshold"] * k, color=theme.AMBER, ls="--", lw=1.6, label="decision threshold")
    cur = float(last["inputs"][f]); ax.axvline(cur, color=theme.NAVY, lw=1, ls=":"); ax.scatter([cur], [float(predictor.score_sweep(last["inputs"], f, [cur])[0]) * k], color=theme.NAVY, zorder=3, label="your input")
    ax.set_xlabel(f"{content.label(f)} ({schema['numeric'][f]['unit']})"); ax.set_ylabel("Rain score (%)" if proba else "Decision score")
    ax.set_title(f"Model rain score as {content.label(f).lower()} changes"); ax.legend()
    theme.show(fig)
    theme.what_it_shows("the model's score when only this measurement changes and everything else stays as you entered it. Above the dashed line the model says 'rain'. It reveals how the model behaves; it is not a real-world cause-and-effect experiment.")


def render():
    theme.title("🌧️ Rainfall Predictor", "Describe a day's weather, then see whether rain is more or less likely tomorrow.")
    if not data.require(data.MODEL_FILES, "The prediction model"):
        return
    loaded = data.safe_model()
    if loaded is None:
        return
    _, schema = loaded
    report = data.read_json("model_report.json")
    prof = report.get("profile", {})
    theme.note("<b>This is a historical-data model, not a live forecast.</b> The sliders start at a <i>typical</i> day from the dataset. They are <b>not</b> today's weather. Enter real readings you trust (for example from a home weather station) to get a meaningful result.", "info")
    for f in schema["features"]:                     # first visit: fill the defaults
        st.session_state.setdefault(f"in_{f}", _default(f, schema, prof))
    st.button("↺ Reset to a typical day", on_click=_reset, args=(schema, prof))

    with st.form("weather_form"):
        for name, feats in GROUPS:
            feats = [f for f in feats if f in schema["features"]]
            if not feats: continue
            with st.container(border=True):
                st.markdown(f"**{name}**")
                cols = st.columns(min(len(feats), 3) if name.startswith("🌬") else min(len(feats), 4))
                for i, f in enumerate(feats):
                    _widget(f, schema, prof, cols[i % len(cols)])
        go = st.form_submit_button("🔮 Predict tomorrow's rain", type="primary")
    st.caption("Sliders cover the typical range seen in the data (about the 1st–99th percentile plus a margin).")

    if go:
        inputs = {f: st.session_state[f"in_{f}"] for f in schema["features"]}
        with st.spinner("Reading the weather signs…"):
            try:
                res = predictor.predict(inputs)
                expl = predictor.explain(inputs) if res["ok"] else None
            except Exception as e:
                st.error(f"Something went wrong while predicting: {e}"); return
        st.session_state["last"] = {"inputs": inputs, "res": res, "expl": expl}

    last = st.session_state.get("last")
    if not last:
        st.markdown('<div class="note info">☁️ <b>No prediction yet.</b> Adjust the inputs above and press <b>Predict tomorrow\'s rain</b>.</div>', unsafe_allow_html=True)
        return
    res = last["res"]
    if not res["ok"]:
        st.error("The prediction could not be made. Please fix:")
        for e in res["errors"]: st.markdown(f"- {e}")
        return
    st.divider()
    _result_card(res, schema, report)
    for w in res["warnings"]:
        st.warning(w)
    c1, c2 = st.columns(2)
    with c1: _push_chart(last["expl"], res)
    with c2:
        if prof: _compare_chart(last["inputs"], prof)
    _plan(res)
    theme.note("<b>Please remember:</b> this is an uncertain machine-learning estimate from historical patterns. It is <b>not</b> a guaranteed forecast or an official weather warning. "
               "For real forecasts and warnings, use the Bureau of Meteorology or your local weather service.")
    st.caption("✏️ Want to try something else? Change the inputs above and press Predict again, or play with the scenario explorer below.")
    _scenario(last, schema, prof)
