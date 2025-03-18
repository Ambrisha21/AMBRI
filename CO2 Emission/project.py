# Code saved in NotePad 

import streamlit as st
import numpy as np
import joblib

# Cache model loading
@st.cache_resource
def load_model():
    return joblib.load("co2_model.pkl")

# Load the trained model
rf_model = load_model()

# Title
st.title("🚗 CO₂ Emissions Predictor")

st.markdown("### Predict the CO₂ emissions of a vehicle based on its specifications.")

# User Input Fields
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, step=1, value=4)
fuel_consumption_l = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=30.0, step=0.1, value=8.0)
fuel_consumption_mpg = st.number_input("Fuel Consumption (MPG)", min_value=5.0, max_value=100.0, step=0.5, value=30.0)

# Predict Button
if st.button("🔍 Predict CO₂ Emissions"):
    # Prepare input data
    input_data = np.array([[engine_size, cylinders, fuel_consumption_l, fuel_consumption_mpg]])
    input_data = input_data.reshape(1, -1)  # Ensure proper shape

    # Make prediction
    prediction = rf_model.predict(input_data)[0]


    # Display result
    st.success(f"🚘 Estimated CO₂ Emissions: {prediction:.2f} g/km")

