import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load models
solar_model = joblib.load("solar_model.pkl")
wind_model = joblib.load("wind_model.pkl")

st.title("Wind & Solar Power Prediction AI")

# User inputs
solar_radiation = st.slider("Solar Radiation (W/m²)", 100, 1000, 500)
wind_speed = st.slider("Wind Speed (m/s)", 1, 25, 10)
temperature = st.slider("Temperature (°C)", -10, 40, 20)
humidity = st.slider("Humidity (%)", 10, 100, 50)
pressure = st.slider("Air Pressure (hPa)", 900, 1100, 1013)

# Prediction
input_data = np.array([[solar_radiation, wind_speed, temperature, humidity, pressure]])
solar_output = solar_model.predict(input_data)[0]
wind_output = wind_model.predict(input_data)[0]

st.write(f"### 🌞 Predicted Solar Power Output: {solar_output:.2f} kW")
st.write(f"### 💨 Predicted Wind Power Output: {wind_output:.2f} kW")