import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Diabetes Predictor")

preg = st.number_input("Pregnancies")
glucose = st.number_input("Glucose")
bp = st.number_input("Blood Pressure")
skin = st.number_input("Skin Thickness")
insulin = st.number_input("Insulin")
bmi = st.number_input("BMI")
dpf = st.number_input("DPF")
age = st.number_input("Age")

if st.button("Predict"):
    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)

    result = model.predict(data_scaled)

    if result[0] == 1:
        st.error("⚠️ Diabetic")
    else:
        st.success("✅ Not Diabetic")
        