import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and scaler
model = joblib.load("best_model.pkl")

# Load the same scaler used for training
scaler = joblib.load('scaler.pkl')  

# Streamlit App Title
st.title(" Bankruptcy Risk Prediction App")

st.markdown("Enter the risk factors below to predict if a company is at risk of bankruptcy.")

# User inputs
industrial_risk = st.number_input("Industrial Risk", min_value=0.0, max_value=1.0, step=0.1)
management_risk = st.number_input("Management Risk", min_value=0.0, max_value=1.0, step=0.1)
financial_flexibility = st.number_input("Financial Flexibility", min_value=0.0, max_value=1.0, step=0.1)
credibility = st.number_input("Credibility", min_value=0.0, max_value=1.0, step=0.1)
competitiveness = st.number_input("Competitiveness", min_value=0.0, max_value=1.0, step=0.1)
operating_risk = st.number_input("Operating Risk", min_value=0.0, max_value=1.0, step=0.1)

# Prediction
if st.button("Predict Bankruptcy Risk"):
    #Prepare input data
    input_data = np.array([[industrial_risk, management_risk, financial_flexibility, credibility, competitiveness,operating_risk]])
    
    # Apply feature scaling (if used during training)
    input_data_scaled = scaler.transform(input_data)
    
    #make prediction
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data_scaled)[:, 1]  # Probability of bankruptcy risk

    # Display the result
    if prediction[0] == 1:
        st.error(f" **High Bankruptcy Risk!** (Risk Probability: {probability[0]:.2f})")
    else:
        st.success(f" **No Bankruptcy Risk** (Risk Probability: {probability[0]:.2f})")
