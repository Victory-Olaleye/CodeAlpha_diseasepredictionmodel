import streamlit as st 
import joblib
import numpy as np

# Load model and scaler
model = joblib.load(open('disease_prediction_model.pkl', 'rb'))
scaler = joblib.load(open('scaler.pkl', 'rb'))

st.title('Diabetes Prediction Model')

with open("feature_description.txt", "r") as f:
    description = f.read()

st.markdown("Feature Descriptions")
st.text(description)

# Input fields
pregnancies = st.number_input('Number of Pregnancies', min_value=0,)
Glucose = st.number_input('Glucose Level', min_value=0,)
BloodPressure = st.number_input('Blood Pressure value', min_value=0,)
SkinThickness = st.number_input('Skin Thickness value', min_value=0,)
Insulin = st.number_input('Insulin Level', min_value=0,)
BMI = st.number_input('BMI value', min_value=0.0,)
DiabetesPedigreeFunction = st.number_input('Diabetes Pedigree Function value', min_value=0.0,)
Age = st.number_input('Age of the Person', min_value=0,)

# Applyinh scaling
input_data = (pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age)
scaled_data = scaler.transform([input_data])

if st.button('Predict'):
    prediction = model.predict(scaled_data)
    if prediction[0] == 1:
        st.error('The person is likely to have Diabetes')
    else:
        st.success('The person is unlikely to have the Diabetes.')
