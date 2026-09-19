import streamlit as st
import pandas as pd
import pickle
import requests
import matplotlib.pyplot as plt

st.title("Air Pollution Predictor")

def risk(aqi):
    if aqi <= 50:
        return "Good / Green"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def diseases(r):
    if r == "Good / Green":
        return ["No major health risk", "Air quality is good"]
    elif r == "Satisfactory":
        return ["Minor breathing discomfort for sensitive people"]
    elif r == "Moderate":
        return ["Cough", "Throat irritation", "Breathing discomfort"]
    elif r == "Poor":
        return ["Asthma attacks", "Bronchitis", "Chest tightness"]
    elif r == "Very Poor":
        return ["Respiratory illness", "Reduced lung function", "Heart risk"]
    else:
        return ["Severe lung problems", "Heart diseases", "Serious respiratory illness"]

def calc_aqi(p1,p2,n,s,c):
    return 0.35*p1 + 0.25*p2 + 0.15*n + 0.10*s + 8*c

def pollutant_effect(pollutant):
    effects = {
        "PM2.5": "Deep lung penetration causes lung damage",
        "PM10": "Respiratory irritation and coughing",
        "NO2": "Lung inflammation and asthma",
        "SO2": "Breathing issues and throat irritation",
        "CO": "Reduces oxygen in blood, harmful for heart"
    }
    return effects.get(pollutant, "")

# load files
try:
    model = pickle.load(open("model.pkl","rb"))
    cities = pickle.load(open("city_mapping.pkl","rb"))
    factories = pickle.load(open("factory_mapping.pkl","rb"))
    df = pickle.load(open("training_dataset.pkl","rb"))
except:
    st.error("Train model first")
    st.stop()

city = st.selectbox("Select City", list(cities.keys()))

city_df = df[df["City"]==city]

st.write("Factories:", ", ".join(city_df["Factory_Type"].unique()))

# worst factory
row = city_df.sort_values("AQI",ascending=False).iloc[0]
st.write("Worst Factory:", row["Factory_Type"])

# inputs
pm25 = st.number_input("PM2.5", value=float(row["PM2.5"]))
pm10 = st.number_input("PM10", value=float(row["PM10"]))
no2  = st.number_input("NO2", value=float(row["NO2"]))
so2  = st.number_input("SO2", value=float(row["SO2"]))
co   = st.number_input("CO", value=float(row["CO"]))

if st.button("Analyze"):

    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/predict/",
            json={
                "city": city,
                "factory": row["Factory_Type"],
                "pm25": pm25,
                "pm10": pm10,
                "no2": no2,
                "so2": so2,
                "co": co,
            },
            timeout=5,
        )
        response.raise_for_status()
        result = response.json()
    except requests.RequestException as error:
        st.error(f"Backend unavailable: {error}")
        st.stop()

    aqi = result["aqi"]
    r = result["risk"]

    st.write("AQI:", round(aqi,2))
    st.write("Risk:", r)

    # dominant pollutant (correct logic)
    values = {
        "PM2.5": pm25,
        "PM10": pm10,
        "NO2": no2,
        "SO2": so2,
        "CO": co
    }

    limits = {
    "PM2.5": 30,
    "PM10": 50,
    "NO2": 40,
    "SO2": 40,
    "CO": 4
}

    dominant = result["dominant"]

    st.write("Dominant Pollutant:", dominant)
    st.write(f"Effect ({dominant}):", result["effect"])

    # diseases
    st.subheader("Health Effects")
    for d in result["diseases"]:
        st.write("-", d)

    # green check (fix indentation)
    if aqi <= 50:
        st.success("Green City 🌿")
    elif aqi <= 100:
        st.warning("Almost Green City / Satisfactory ⚠️")
    else:
        st.error("Not Green City ❌")

    # graph (only once)
    fig, ax = plt.subplots()
    ax.bar(["PM2.5","PM10","NO2","SO2","CO"], [pm25,pm10,no2,so2,co])
    st.pyplot(fig)