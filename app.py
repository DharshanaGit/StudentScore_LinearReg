import streamlit as st
import numpy as np
import joblib

# ---------------- LOAD MODEL & SCALER ----------------
model = joblib.load("Students_final_score.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Student Final Score Predictor", layout="centered")
st.title("🎓 Student Final Score Prediction")
st.write("Predict a student's **Final Score** using academic and personal factors.")

# ---------------- INPUT SECTION ----------------
st.subheader("📌 Enter Student Details")

Previous_Sem_Score = st.number_input(
    "Previous Semester Score",
    min_value=0.0, max_value=100.0, value=75.0
)

Study_Hours_per_Week = st.slider(
    "Study Hours per Week",
    min_value=0, max_value=70, value=20
)

Attendance_Percentage = st.slider(
    "Attendance Percentage",
    min_value=0, max_value=100, value=85
)

Sleep_Hours = st.slider(
    "Sleep Hours per Day",
    min_value=0, max_value=12, value=7
)

Motivation_Level = st.slider(
    "Motivation Level (1-10)",
    min_value=1, max_value=10, value=6
)


# ---------------- PREDICTION ----------------
if st.button("🔮 Predict Final Score"):

    # EXACT 5 FEATURES ONLY
    input_data = np.array([[
        Previous_Sem_Score,
        Study_Hours_per_Week,
        Attendance_Percentage,
        Sleep_Hours,
        Motivation_Level
    ]])

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]

    st.success(f"🎯 Predicted Final Score: {prediction:.2f}")

    # Interpretation
    if prediction >= 70:
        st.info("🌟 Excellent performance expected!")
    elif prediction >= 40:
        st.info("✅ Average to good performance expected.")
    else:
        st.warning("⚠️ Student may need academic support.")