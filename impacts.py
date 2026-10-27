"""Rainfall Impact Explorer: choose a topic and an illustrative rainfall level."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from skysense import content, data, theme


def _bullets(items):
    return "".join(f"<li>{i}</li>" for i in items)


def _potential_chart(info, chosen):
    xs = np.arange(len(content.LEVELS))
    ben = [info["scores"][l][0] for l in content.LEVELS]; risk = [info["scores"][l][1] for l in content.LEVELS]
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    ax.bar(xs - .2, ben, .4, color=theme.TEAL, label="Benefit potential")
    ax.bar(xs + .2, risk, .4, color=theme.CORAL, label="Risk potential")
    i = content.LEVELS.index(chosen)
    ax.axvspan(i - .45, i + .45, color=theme.SKY, alpha=.18, zorder=0)
    ax.set_xticks(xs); ax.set_xticklabels(content.LEVELS); ax.set_ylim(0, 5.2); ax.set_yticks(range(0, 6))
    ax.set_yticklabels(["none", "", "", "", "", "very high"][:6]); ax.set_title("Benefit vs risk as rainfall intensifies (illustrative)", pad=26); ax.legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=8)
    theme.show(fig)
    theme.what_it_shows("a qualitative 0–5 judgement for this topic, not measured data. Benefits usually peak at moderate rain; risks climb steeply toward extreme rain.")


def _band_chart(df, station):
    d = df if station == "All stations" else df[df.Location == station]
    r = d["Rainfall"].dropna()
    if len(r) < 30:
        st.info("Not enough rainfall readings for this selection."); return
    shares = [((r >= lo) & (r < hi)).mean() * 100 for _, lo, hi in content.LEVEL_BANDS]
    labels = [b[0] for b in content.LEVEL_BANDS]
    fig, ax = plt.subplots(figsize=(6.4, 3.3))
    bars = ax.barh(labels[::-1], shares[::-1], color=[theme.AMBER, theme.SKY, theme.BLUE, "#123a73", theme.NAVY][::-1])
    for b, v in zip(bars, shares[::-1]):
        ax.text(v + .8, b.get_y() + b.get_height() / 2, f"{v:.1f}%" if v >= .1 else "<0.1%", va="center", fontsize=9)
    ax.set_xlabel("% of recorded days"); ax.set_xlim(0, max(shares) * 1.18); ax.set_title(f"How often each rainfall level occurred: {station}")
    theme.show(fig)
    theme.what_it_shows(f"the share of {len(r):,} daily records in each illustrative daily-rainfall band. Heavy and extreme days are rare, but rare is not never.")


def render():
    theme.title("🌍 Rainfall Impact Explorer", "Rain is life-giving and risky. What matters is how much, how fast, and how ready we are.")
    theme.note(content.IMPACT_DISCLAIMER, "info")
    topic = st.radio("Choose a topic", list(content.IMPACTS), horizontal=True, key="impact_topic")
    level = st.select_slider("How much rain? (illustrative daily totals)", options=content.LEVELS, value="Moderate", key="impact_level")
    st.caption(f"**{level}**: {content.LEVEL_RANGES[level]}")
    info = content.IMPACTS[topic]

    st.markdown(f'<div class="hero" style="padding:1.3rem 1.6rem"><h1 style="font-size:1.8rem">{info["icon"]} {topic}</h1><p>{info["tagline"]}</p></div>', unsafe_allow_html=True)
    c = st.columns(2)
    c[0].markdown(f'<div class="card"><h4>🌱 The benefits</h4><ul>{_bullets(info["good"])}</ul></div>', unsafe_allow_html=True)
    c[1].markdown(f'<div class="card"><h4>⚠️ The risks</h4><ul>{_bullets(info["bad"])}</ul></div>', unsafe_allow_html=True)
    st.markdown("")
    st.markdown(f'<div class="fact" style="border-left-color:#f5a623"><b>At {level.lower()} rainfall</b> ({content.LEVEL_RANGES[level]})<br><span>{info["levels"][level]}</span></div>', unsafe_allow_html=True)
    st.markdown(f"**It depends on:** {info['depends']}. The same rain can be a blessing in one place and a hazard in another.")

    c1, c2 = st.columns(2)
    with c1: _potential_chart(info, level)
    with c2:
        if data.missing(data.DATA_FILES):
            theme.note("Run the notebook to see how often each rainfall level occurs in the dataset.", "info")
        else:
            df = data.load_weather()
            station = st.selectbox("Station for the frequency chart", ["All stations"] + sorted(df.Location.unique()), key="impact_station")
            _band_chart(df, station)

    st.markdown('<div class="card"><h4>🧰 Preparedness tips</h4><ul>' + _bullets(info["tips"]) + "</ul></div>", unsafe_allow_html=True)
    with st.expander(f"See every topic at {level.lower()} rainfall"):
        st.table(pd.DataFrame({"Topic": [f"{v['icon']} {k}" for k, v in content.IMPACTS.items()], f"What can happen ({level})": [v["levels"][level] for v in content.IMPACTS.values()]}).set_index("Topic"))
    st.caption("Sources: general educational knowledge about rainfall, floods and drought. For local warnings and thresholds, follow your national weather service and emergency authorities.")
