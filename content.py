"""Plain-language text: input labels/tooltips, glossary, rainfall-impact education. This is general educational content,
NOT something measured by the WeatherAUS dataset."""

# feature -> (friendly label, tooltip)
FEATURE_INFO = {
    "MinTemp": ("Minimum temperature", "Lowest temperature recorded in the 24 hours up to 9am."),
    "MaxTemp": ("Maximum temperature", "Highest temperature recorded during the day."),
    "Rainfall": ("Rain recorded today", "Rainfall measured for the day, in millimetres. 1 mm = 1 litre of water on each square metre."),
    "Evaporation": ("Evaporation", "Water that evaporates from a standard measuring pan over 24 hours (mm). Higher on hot, dry, windy days."),
    "Sunshine": ("Bright sunshine", "Hours of bright sunshine during the day. Cloudy, wet days have few hours."),
    "WindGustSpeed": ("Strongest wind gust", "Speed of the strongest gust in the 24 hours to midnight."),
    "WindSpeed9am": ("Wind speed at 9am", "Average wind speed in the 10 minutes before 9am."),
    "WindSpeed3pm": ("Wind speed at 3pm", "Average wind speed in the 10 minutes before 3pm."),
    "Humidity9am": ("Morning humidity", "Relative humidity at 9am: how close the air is to holding all the water vapour it can (100% = saturated)."),
    "Humidity3pm": ("Afternoon humidity", "Relative humidity at 3pm. One of the most useful clues in this dataset."),
    "Pressure9am": ("Morning pressure", "Atmospheric pressure at 9am, adjusted to sea level. Falling pressure often signals unsettled weather."),
    "Pressure3pm": ("Afternoon pressure", "Atmospheric pressure at 3pm, adjusted to sea level (hPa). About 1013 hPa is the world average."),
    "Cloud9am": ("Morning cloud cover", "How much of the sky is covered by cloud at 9am, in oktas (eighths): 0 = clear, 8 = fully overcast."),
    "Cloud3pm": ("Afternoon cloud cover", "How much of the sky is covered by cloud at 3pm, in oktas (eighths): 0 = clear, 8 = fully overcast."),
    "Temp9am": ("Temperature at 9am", "Air temperature at 9am."),
    "Temp3pm": ("Temperature at 3pm", "Air temperature at 3pm."),
    "WindGustDir": ("Gust direction", "Compass direction the strongest gust came FROM."),
    "WindDir9am": ("Wind direction at 9am", "Compass direction the wind blew FROM at 9am."),
    "WindDir3pm": ("Wind direction at 3pm", "Compass direction the wind blew FROM at 3pm."),
    "RainToday": ("Did it rain today?", "'Yes' if more than 1 mm of rain was recorded today."),
    "Location": ("Weather station", "One of the Australian weather stations in the dataset."),
    "Month": ("Month of the year", "Rain patterns change with the seasons."),
}
def label(f): return FEATURE_INFO.get(f, (f, ""))[0]
def tip(f): return FEATURE_INFO.get(f, ("", ""))[1]

GLOSSARY = {
    "Relative humidity": "How much water vapour the air holds compared with the most it could hold at that temperature.",
    "Atmospheric pressure (hPa)": "The weight of the air above you. Low-pressure systems usually bring cloud and rain; high pressure usually brings settled, dry weather.",
    "Okta": "A unit for cloud cover: the sky is divided into 8 parts; 0 oktas = clear sky, 8 oktas = completely cloudy.",
    "Wind gust": "A brief, sharp increase in wind speed above the average.",
    "Evaporation": "Water changing from liquid to vapour. Measured in mm of water lost from a standard pan.",
    "Class imbalance": "When one outcome is much more common than the other. Here dry days outnumber rainy days about 3 to 1.",
    "Threshold": "The score above which the model says 'rain'. It is tuned to balance missed rain against false alarms.",
    "Data leakage": "When information from the future or the answer itself sneaks into the inputs. `RISK_MM` (tomorrow's rain) was removed for this reason.",
    "Association vs causation": "Two things moving together does not prove one causes the other.",
    "Illustrative": "An example to help you think, not an official or measured value.",
}

LEVELS = ["Low", "Moderate", "Heavy", "Extreme"]
LEVEL_RANGES = {"Low": "about 1–10 mm in a day", "Moderate": "about 10–25 mm in a day", "Heavy": "about 25–50 mm in a day", "Extreme": "more than about 50 mm in a day"}
LEVEL_BANDS = [("Dry (< 1 mm)", 0, 1), ("Low (1–10 mm)", 1, 10), ("Moderate (10–25 mm)", 10, 25), ("Heavy (25–50 mm)", 25, 50), ("Extreme (50+ mm)", 50, 10_000)]
IMPACT_DISCLAIMER = ("These intensity bands are **illustrative** examples, not official thresholds. Official warning levels differ by place and are set by "
                     "your national weather service (in Australia, the Bureau of Meteorology) and local emergency services. The WeatherAUS dataset measures weather only; "
                     "the impacts below are **general educational information**, not measurements from the data.")

IMPACTS = {
    "Agriculture & food": dict(icon="🌾", tagline="Rain feeds crops, but timing and amount matter as much as the total.",
        good=["Refills soil moisture and cuts irrigation needs", "Supports seed germination, crop growth and pasture", "Tops up farm dams and tanks"],
        bad=["Long dry spells (drought) stress crops and lower yields", "Heavy rain can waterlog roots and delay sowing or harvest", "Intense downpours wash away topsoil and nutrients"],
        levels={"Low": "Light rain helps topsoil moisture and germination, but after a long dry spell it may not reach deeper roots.",
                "Moderate": "Often ideal: water soaks in instead of running off, supporting growth and reducing irrigation.",
                "Heavy": "Can exceed what soil absorbs. Runoff, waterlogging and erosion risk rise, especially on slopes or bare soil.",
                "Extreme": "Flooded paddocks, crop and livestock losses, damaged fences and farm roads; harvest may be impossible for days."},
        tips=["Plan sowing and harvest using official forecasts", "Use mulch, cover crops and contour farming to reduce erosion", "Keep drains and channels clear", "Move livestock and stored feed to higher ground before floods"],
        depends="crop type and growth stage, soil type, slope, timing of the rain and drainage", scores={"Low": (3, 1), "Moderate": (5, 1), "Heavy": (3, 3), "Extreme": (1, 5)}),
    "Transport & infrastructure": dict(icon="🚗", tagline="Rain rarely helps traffic, and the risks climb quickly with intensity.",
        good=["Washes dust from roads and can reduce fire danger along routes", "Fills the reservoirs that supply cities (see Water resources)"],
        bad=["Flooded roads and underpasses", "Poor visibility, slippery surfaces and longer stopping distances", "Delays or cancellations for trains and flights; landslips on slopes", "Drains and culverts overwhelmed"],
        levels={"Low": "Roads turn slippery, especially after a dry spell when oil and dust mix with the first rain. Allow a bigger gap.",
                "Moderate": "Reduced visibility and longer stopping distances; minor delays are common.",
                "Heavy": "Water pools on roads and drains struggle. Flights and trains may be delayed and low-lying roads can flood.",
                "Extreme": "Flash flooding can cut roads and rail. Closures and cancellations are likely. Never drive through floodwater."},
        tips=["Slow down, use headlights and keep a longer gap", "Never drive, walk or ride through floodwater", "Check official traffic and transit updates before leaving", "Avoid parking in creek beds and low spots"],
        depends="drainage capacity, road surface, terrain, how wet the ground already is and how fast the rain falls", scores={"Low": (1, 2), "Moderate": (1, 3), "Heavy": (1, 4), "Extreme": (0, 5)}),
    "Water resources": dict(icon="🚰", tagline="Rain is the source of our water supply, so both too little and too much cause problems.",
        good=["Fills reservoirs, rivers and wetlands", "Recharges groundwater", "Supports drinking water, farming and industry"],
        bad=["Droughts shrink storages and lead to restrictions", "Heavy runoff carries sediment and pollution into supplies", "Floods can damage pipes, pumps and treatment plants"],
        levels={"Low": "Small amounts mostly wet the ground; little reaches rivers or dams unless the catchment is already wet.",
                "Moderate": "Soaking rain can lift stream flows and help recharge groundwater.",
                "Heavy": "Runoff rises quickly, rivers climb and reservoirs can fill, but muddy, polluted water is harder to treat.",
                "Extreme": "Flood peaks, spilling storages, contamination and stress on water infrastructure."},
        tips=["Save water in dry spells and follow local restrictions", "Rainwater tanks with a first-flush device add resilience", "Follow official boil-water advice after floods", "Keep chemicals and waste away from drains and creeks"],
        depends="catchment condition, soil moisture, vegetation, storage capacity and how long the rain lasts", scores={"Low": (3, 1), "Moderate": (5, 1), "Heavy": (4, 3), "Extreme": (2, 5)}),
    "Environment & ecosystems": dict(icon="🌿", tagline="Ecosystems depend on rain, and many are adapted to a particular rhythm of wet and dry.",
        good=["Restores soil moisture and vegetation growth", "Maintains river flows and refills wetlands", "Triggers breeding and growth in many species"],
        bad=["Prolonged dryness stresses ecosystems and raises bushfire risk", "Heavy rain causes erosion and sediment runoff into streams", "Floods can disrupt or destroy habitats and wash pollutants into waterways"],
        levels={"Low": "Wets leaf litter and topsoil; helpful, but small amounts may evaporate quickly.",
                "Moderate": "Supports plant growth and steady river flow with little damage.",
                "Heavy": "Soil erosion and murky, sediment-rich runoff increase, particularly on bare or burnt ground.",
                "Extreme": "Flooding, landslips and habitat disruption; pollutants and debris are washed downstream."},
        tips=["Keep soil covered with plants to hold it in place", "Avoid disturbing river banks and steep bare slopes", "Never wash chemicals or waste into drains", "Stay on tracks in wet bushland to limit erosion"],
        depends="vegetation cover, soil and slope, recent fires, the season, and how adapted local species are", scores={"Low": (3, 1), "Moderate": (5, 1), "Heavy": (3, 3), "Extreme": (1, 5)}),
    "Public health & communities": dict(icon="🏥", tagline="Rain itself is not harmful, but floods and standing water can be.",
        good=["Can clear dust and air pollution and ease heat", "Reliable rain supports safe, plentiful drinking water"],
        bad=["Flood injuries, drowning and displacement", "Floodwater can carry sewage and chemicals", "Standing water can let mosquitoes breed, and contaminated water raises waterborne-illness risk", "Damp buildings grow mould"],
        levels={"Low": "Mostly everyday effects: wet, slippery paths. Some cleaner air.",
                "Moderate": "Puddles and containers that stay full for days can start to breed mosquitoes.",
                "Heavy": "Low-lying areas may flood and sewers can overflow. Stay informed and avoid floodwater.",
                "Extreme": "Evacuations, isolation, injury and drowning risk, contaminated water and disrupted health services."},
        tips=["Avoid floodwater; it can hide hazards and contamination", "Follow official boil-water and evacuation advice", "Empty standing water from containers regularly", "Keep an emergency kit with medicines, torch, radio and water", "Dry and clean damp areas quickly to prevent mould"],
        depends="flood-defence and drainage quality, sanitation, how prepared people are, and access to health care", scores={"Low": (2, 1), "Moderate": (2, 2), "Heavy": (1, 4), "Extreme": (0, 5)}),
    "Economy & energy": dict(icon="⚡", tagline="Rain shapes harvests, power supply and everyday business, for better and worse.",
        good=["Supports agricultural output and food prices", "Fills reservoirs that drive hydropower", "Reduces irrigation and fire-fighting costs"],
        bad=["Flood damage and expensive recovery", "Outdoor work, construction, events and tourism are cancelled", "Supply chains delayed by closed roads and ports", "Extreme events strain insurance and public budgets"],
        levels={"Low": "Small effects: some outdoor work slows; farms and reservoirs benefit a little.",
                "Moderate": "Good for farming and hydropower; outdoor events may need a wet-weather plan.",
                "Heavy": "Site closures, delays and some damage; transport of goods becomes slower.",
                "Extreme": "Widespread damage, power and supply disruption, and large recovery costs."},
        tips=["Businesses: keep a continuity plan and backup power", "Review insurance cover before the wet season", "Watch supplier and transport alerts", "Have a wet-weather plan for outdoor work and events"],
        depends="the type of business, location, infrastructure resilience, insurance and how quickly services recover", scores={"Low": (2, 1), "Moderate": (4, 1), "Heavy": (3, 3), "Extreme": (1, 5)}),
}

FACTORS = {
    "Humidity": dict(icon="💧", vars=["Humidity3pm", "Humidity9am"], main="Humidity3pm", unit="%",
        text="Humid air is closer to being saturated, so cloud and rain form more easily. Afternoon humidity is one of the most informative measurements for next-day rain.",
        advanced="Relative humidity depends on both the amount of water vapour and the temperature, so it must be read together with temperature."),
    "Pressure": dict(icon="🧭", vars=["Pressure3pm", "Pressure9am"], main="Pressure3pm", unit="hPa",
        text="Low pressure usually means rising air, cloud and unsettled weather; high pressure usually means sinking air and dry, settled weather.",
        advanced="Pressure differences between stations drive wind; here we only see the pressure at the station itself."),
    "Temperature": dict(icon="🌡️", vars=["MaxTemp", "MinTemp", "Temp3pm", "Temp9am"], main="MaxTemp", unit="°C",
        text="Warm air can hold more water vapour, but temperature alone is a weak signal and depends strongly on the season and location.",
        advanced="Because Australia spans tropical to temperate climates, a given temperature means different things in Darwin and Hobart."),
    "Sunshine": dict(icon="☀️", vars=["Sunshine"], main="Sunshine", unit="hours",
        text="Many hours of bright sunshine mean clear skies today. Dull days with little sun are more often followed by rain.",
        advanced="Sunshine hours are missing for many stations in this dataset, which is why imputation matters."),
    "Wind": dict(icon="🌬️", vars=["WindGustSpeed", "WindSpeed3pm", "WindSpeed9am"], main="WindGustSpeed", unit="km/h",
        text="Strong gusts can come with fronts and storms. Wind direction matters because it decides whether air arrives from the ocean (moist) or the inland (dry).",
        advanced="The meaning of a wind direction depends on the coastline near each station."),
    "Recent rainfall": dict(icon="🌧️", vars=["Rainfall"], main="Rainfall", unit="mm",
        text="Wet weather often comes in spells: if it rained today, rain tomorrow is more likely than after a dry day.",
        advanced="Rainfall is strongly right-skewed (most days near zero), so medians and rank tests describe it better than means."),
}

UNITS = {"MinTemp": "°C", "MaxTemp": "°C", "Rainfall": "mm", "Evaporation": "mm", "Sunshine": "hours", "WindGustSpeed": "km/h", "WindSpeed9am": "km/h",
         "WindSpeed3pm": "km/h", "Humidity9am": "%", "Humidity3pm": "%", "Pressure9am": "hPa", "Pressure3pm": "hPa", "Cloud9am": "oktas", "Cloud3pm": "oktas",
         "Temp9am": "°C", "Temp3pm": "°C"}
def axis_label(f): return f"{label(f)} ({UNITS.get(f, '')})"
