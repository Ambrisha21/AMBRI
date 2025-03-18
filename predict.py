import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
with open("co2_model.pkl", "rb") as f:
    co2_model = pickle.load(f)

# Streamlit App Title
st.title("🌍 CO₂ Emissions Predictor")
st.write("Enter vehicle details to predict CO₂ emissions.")

# User Input Fields
engine_size = st.number_input("Engine Size (L)", min_value=0.5, max_value=10.0, step=0.1, value=2.0)
cylinders = st.number_input("Number of Cylinders", min_value=2, max_value=16, step=1, value=4)
fuel_consumption_l = st.number_input("Fuel Consumption (L/100km)", min_value=1.0, max_value=30.0, step=0.1, value=8.0)
fuel_consumption_mpg = st.number_input("Fuel Consumption (MPG)", min_value=5.0, max_value=100.0, step=0.5, value=30.0)

# Predict Button
if st.button("🔍 Predict CO₂ Emissions"):
    # Prepare input data
    input_data = np.array([[engine_size, cylinders, fuel_consumption_l, fuel_consumption_mpg]])
    
    # Make prediction
    prediction = co2_model.predict(input_data)[0]
    
    # Display result
    st.success(f"🚘 Estimated CO₂ Emissions: {prediction:.2f} g/km")
