import streamlit as st
import pickle
import numpy as np

# Load model
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Diabetes Predictor", layout="centered")

st.title("🩺 Diabetes Prediction System")
st.markdown("### Enter patient details below")

# Layout (2 columns)
col1, col2 = st.columns(2)

with col1:
    preg = st.number_input("Pregnancies", 0.0)
    glucose = st.number_input("Glucose", 0.0)
    bp = st.number_input("Blood Pressure", 0.0)
    skin = st.number_input("Skin Thickness", 0.0)

with col2:
    insulin = st.number_input("Insulin", 0.0)
    bmi = st.number_input("BMI", 0.0)
    dpf = st.number_input("DPF", 0.0)
    age = st.number_input("Age", 0.0)

# Prediction
if st.button("🔍 Predict"):

    data = np.array([[preg, glucose, bp, skin, insulin, bmi, dpf, age]])
    data_scaled = scaler.transform(data)

    result = model.predict(data_scaled)
    prob = model.predict_proba(data_scaled)

    st.subheader("Result:")

    if result[0] == 1:
        st.error(f"⚠️ Diabetic (Confidence: {prob[0][1]*100:.2f}%)")
    else:
        st.success(f"✅ Not Diabetic (Confidence: {prob[0][0]*100:.2f}%)")

    st.progress(int(max(prob[0])*100))