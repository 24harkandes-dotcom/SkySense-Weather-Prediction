"""Home: hero, navigation cards, data-backed discoveries, glossary and a quick tour."""
import streamlit as st

from skysense import analysis, content, data, theme


def _go(page):
    st.session_state["page"] = page


def render():
    st.markdown(f"""<div class="hero">{theme.HERO_SVG}
        <div class="eyebrow">Weather intelligence &amp; rainfall insights</div>
        <h1>Predict the rain. Understand its impact. Prepare for what comes next.</h1>
        <p>SkySense turns decades of Australian weather records into plain-English answers: what usually comes before a rainy day,
        how a machine-learning model reads the signs, and how rain shapes farms, roads, water and communities.</p></div>""", unsafe_allow_html=True)
    st.button("🌧️ Try the rainfall predictor", type="primary", on_click=_go, args=("🌧️ Rainfall Predictor",))
    theme.note("<b>Historical data, not a live forecast.</b> Predictions use patterns learned from past Australian weather. "
               "SkySense has no access to today's real weather, so you type in the conditions yourself. For actual forecasts and warnings use the Bureau of Meteorology or your local weather service.")

    theme.title("Explore SkySense")
    cards = [("🌧️", "Rainfall Predictor", "Enter a day's weather and see whether rain is more or less likely tomorrow.", "🌧️ Rainfall Predictor"),
             ("🌍", "Impact Explorer", "See how low to extreme rainfall touches farms, roads, water, nature, health and the economy.", "🌍 Impact Explorer"),
             ("📊", "Weather Insights", "Filter and explore the real data: stations, seasons, temperature, humidity, pressure and more.", "📊 Weather Insights"),
             ("🔬", "Rainfall Factors", "Which conditions go with rain, and how does the model use them?", "🔬 Rainfall Factors"),
             ("🧠", "Model Insights", "Six models compared fairly, with metrics explained in simple words.", "🧠 Model Insights")]
    for row in (cards[:3], cards[3:]):
        cols = st.columns(3)
        for col, (ic, h, b, page) in zip(cols, row):
            with col:
                st.markdown(theme.card(ic, h, b, "navcard"), unsafe_allow_html=True)
                st.button(f"Open {h}", key=f"nav_{h}", on_click=_go, args=(page,))
    # why it matters
    theme.title("Why does rainfall prediction matter?", "Knowing a day ahead changes what people do.")
    c = st.columns(3)
    c[0].markdown(theme.card("🌾", "Farmers", "Choose when to sow, spray, irrigate or harvest."), unsafe_allow_html=True)
    c[1].markdown(theme.card("🚦", "Cities & travel", "Prepare for slippery roads, delays and flash-flood risk."), unsafe_allow_html=True)
    c[2].markdown(theme.card("🏘️", "Communities", "Get ready for floods, protect water supplies and look after vulnerable people."), unsafe_allow_html=True)

    # discoveries computed from the real data
    theme.title("Discoveries from the data", "Calculated live from the dataset, not typed in by hand.")
    if data.missing(data.DATA_FILES):
        theme.note("Run the training notebook to generate the dataset file and unlock data-backed discoveries.", "info")
    else:
        for ic, head, body in analysis.facts(data.load_weather()):
            st.markdown(f'<div class="fact">{ic} <b>{head}</b><br><span>{body}</span></div>', unsafe_allow_html=True)
        st.caption("Facts describe patterns across all stations and years combined; individual places differ.")

    with st.expander("🧭 First time here? A 30-second tour"):
        st.markdown("1. **Rainfall Predictor**: set the sliders to a day's weather and press *Predict*.\n"
                    "2. **Impact Explorer**: pick a topic (e.g. farming) and an illustrative rainfall level.\n"
                    "3. **Weather Insights / Rainfall Factors**: explore the charts and tick the filters.\n"
                    "4. **Model Insights**: see how the prediction model was built and tested.")
    with st.expander("📖 Weather glossary"):
        for k, v in content.GLOSSARY.items():
            st.markdown(f"**{k}**: {v}")
    with st.expander("ℹ️ How does this work?"):
        st.markdown("A computer program (a *machine-learning model*) studied over a decade of daily weather records from Australian stations, "
                    "learning which combinations of measurements tended to come before rainy days. It was then tested on the **most recent years it had never seen**. "
                    "Given your inputs it gives a score and a yes/no answer. It can be wrong; weather is uncertain.")
