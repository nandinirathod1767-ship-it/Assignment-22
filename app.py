import streamlit as st
import pandas as pd
import joblib

# Load Saved Model
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

st.title("❤️ Heart Disease Prediction System")

# Input Fields
age = st.number_input("Age", 18, 100, 40)

sex = st.selectbox("Sex", ["M", "F"])

chest = st.selectbox(
    "Chest Pain Type",
    ["ATA", "NAP", "ASY", "TA"]
)

bp = st.number_input("Resting Blood Pressure", 80, 250, 120)

chol = st.number_input("Cholesterol", 0, 700, 200)

fasting = st.selectbox(
    "Fasting Blood Sugar",
    [0, 1]
)

ecg = st.selectbox(
    "Resting ECG",
    ["Normal", "ST", "LVH"]
)

maxhr = st.number_input("Maximum Heart Rate", 60, 220, 150)

exercise = st.selectbox(
    "Exercise Angina",
    ["Y", "N"]
)

oldpeak = st.number_input("Old Peak", 0.0, 10.0, 1.0)

slope = st.selectbox(
    "ST Slope",
    ["Up", "Flat", "Down"]
)

# Predict Button
if st.button("Predict"):

    sample = pd.DataFrame({
        "Age": [age],
        "Sex": [sex],
        "ChestPainType": [chest],
        "RestingBP": [bp],
        "Cholesterol": [chol],
        "FastingBS": [fasting],
        "RestingECG": [ecg],
        "MaxHR": [maxhr],
        "ExerciseAngina": [exercise],
        "Oldpeak": [oldpeak],
        "ST_Slope": [slope]
    })

    # Encoding
    sample = pd.get_dummies(sample)

    # Match Training Columns
    sample = sample.reindex(columns=columns, fill_value=0)

    # Scaling
    sample = scaler.transform(sample)

    # Prediction
    prediction = model.predict(sample)

    if prediction[0] == 1:
        st.error("❤️ Heart Disease : YES")
    else:
        st.success("💚 Heart Disease : NO")