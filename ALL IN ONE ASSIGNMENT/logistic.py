import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load('titanic_model.pkl')

st.title("Titanic Survival Prediction App")
st.write("This app predicts whether a Titanic passenger would survive based on their details.")

# User inputs
pclass = st.selectbox("Pclass (1 = 1st, 2 = 2nd, 3 = 3rd):", [1, 2, 3],index=2)
sex = st.selectbox("Sex (0 = Female, 1 = Male):", [0, 1],index=0)
age = st.slider("Age:", 0, 100, 25)
sibsp = st.number_input("Number of Siblings/Spouses Aboard (SibSp):", min_value=0, max_value=10)
parch = st.number_input(" Number of Parents/Children Aboard (Parch):", min_value=0, max_value=10)
fare = st.number_input("Fare:", min_value=0.0, value=50.0)
embarked = st.selectbox("Embarked (0 = C, 1 = Q, 2 = S):", [0, 1, 2])

# Prepare input data for prediction
input_data = np.array([[pclass, sex, age, sibsp, parch, fare, embarked]])

# Predict survival probability
if st.button("Predict"):
    probabilities = model.predict_proba(input_data)
    survival_probability = probabilities[0][1]  # Probability of surviving (class 1)
    
    # Display result
    if survival_probability > 0.5:
        st.write(f"Prediction: Survived (Probability: {survival_probability:.2f})")
    else:
        st.write(f"Prediction: Did not survive (Probability: {1 - survival_probability:.2f})")

