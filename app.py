"""SkySense - Weather Intelligence & Rainfall Insights (Streamlit).
Run:  streamlit run app.py
The app only READS the files created by the notebook; it never retrains a model."""
import streamlit as st

st.set_page_config(page_title="SkySense - Rainfall Insights", page_icon="🌧️", layout="wide", initial_sidebar_state="collapsed")

from skysense import theme                                              # noqa: E402  (after set_page_config on purpose)
from skysense.views import factors, home, impacts, insights, models, predict   # noqa: E402

PAGES = {"🏠 Home": home, "🌧️ Rainfall Predictor": predict, "🌍 Impact Explorer": impacts,
         "📊 Weather Insights": insights, "🔬 Rainfall Factors": factors, "🧠 Model Insights": models}

theme.inject_css()
theme.apply_mpl()
st.session_state.setdefault("page", "🏠 Home")

st.markdown('<div class="brandbar"><span class="logo">🌦️</span><span class="name">Sky<span>Sense</span></span></div>', unsafe_allow_html=True)
st.radio("Navigate", list(PAGES), key="page", horizontal=True, label_visibility="collapsed")
PAGES[st.session_state["page"]].render()

st.markdown('<div class="footer">SkySense is an educational project built on historical WeatherAUS data (Bureau of Meteorology observations as '
            'distributed with the public dataset). It is <b>not</b> a live forecast and not an official weather warning. '
            'For real forecasts and warnings visit <b>bom.gov.au</b> or your national weather service.</div>', unsafe_allow_html=True)
