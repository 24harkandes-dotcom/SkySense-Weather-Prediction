"""Look and feel: CSS injection, small HTML helpers and one shared Matplotlib style so every chart matches the app."""
from pathlib import Path

import matplotlib.pyplot as plt
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
NAVY, BLUE, SKY, CLOUD, AMBER, CORAL, TEAL = "#0b1f3a", "#1e6fd9", "#5fb3f5", "#f3f8fd", "#f5a623", "#e4572e", "#2a9d8f"
RAIN_C, DRY_C = BLUE, AMBER            # the same two colours mean "rain" and "no rain" everywhere in the app

HERO_SVG = """<svg viewBox="0 0 360 220" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
<g fill="#ffffff" opacity=".95"><ellipse cx="170" cy="95" rx="95" ry="48"/><ellipse cx="115" cy="110" rx="62" ry="38"/><ellipse cx="235" cy="108" rx="66" ry="36"/></g>
<g stroke="#9fd0ff" stroke-width="6" stroke-linecap="round"><line x1="110" y1="165" x2="98" y2="195"/><line x1="160" y1="170" x2="148" y2="205"/><line x1="210" y1="165" x2="198" y2="195"/><line x1="260" y1="170" x2="248" y2="200"/></g>
<circle cx="300" cy="40" r="26" fill="#f5a623" opacity=".9"/></svg>"""


def inject_css():
    css = (ROOT / "assets" / "style.css").read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def apply_mpl():
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#b8c9dd", "axes.labelcolor": NAVY, "text.color": NAVY,
        "xtick.color": "#33506f", "ytick.color": "#33506f", "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e3ecf6", "grid.linewidth": .8, "axes.axisbelow": True,
        "axes.titlesize": 13, "axes.titleweight": "bold", "axes.titlelocation": "left", "axes.labelsize": 10.5,
        "font.size": 10, "legend.frameon": False, "figure.dpi": 100, "savefig.bbox": "tight"})


def show(fig):
    """Display a Matplotlib figure and free its memory."""
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


def title(text, sub=None):
    st.markdown(f'<div class="section-title">{text}</div>' + (f'<div class="section-sub">{sub}</div>' if sub else ""), unsafe_allow_html=True)


def card(icon, head, body, cls=""):
    return f'<div class="card {cls}"><div class="icon">{icon}</div><h4>{head}</h4><p>{body}</p></div>'


def note(text, kind="warn"):
    st.markdown(f'<div class="note {"info" if kind == "info" else ""}">{text}</div>', unsafe_allow_html=True)


def what_it_shows(text):
    st.caption("💡 **What this shows:** " + text)
