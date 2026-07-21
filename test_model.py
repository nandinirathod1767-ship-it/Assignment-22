import pandas as pd
import joblib

# Load Saved Files
model = joblib.load("heart_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")

# Sample Input
sample = pd.DataFrame({
    "Age": [40],
    "Sex": ["M"],
    "ChestPainType": ["ATA"],
    "RestingBP": [120],
    "Cholesterol": [230],
    "FastingBS": [0],
    "RestingECG": ["Normal"],
    "MaxHR": [150],
    "ExerciseAngina": ["N"],
    "Oldpeak": [1.0],
    "ST_Slope": ["Up"]
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
    print("Heart Disease : YES")
else:
    print("Heart Disease : NO")