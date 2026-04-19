import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.set_page_config(page_title="Telecom Risk Intelligence", layout="wide")

st.title("📊 Telecom Customer Risk Intelligence Platform")

# -------------------------------
# INPUT SECTION
# -------------------------------

st.sidebar.header("Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
Partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
Dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)

PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No"])
InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No"])
OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No"])
DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No"])
TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No"])

StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No"])
StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No"])

Contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])

PaymentMethod = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

MonthlyCharges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)

# -------------------------------
# PREDICTION BUTTON
# -------------------------------

if st.button("🔍 Predict Churn Risk"):

    # TotalCharges (safe calc)
    TotalCharges = MonthlyCharges * tenure

    # Create dataframe
    input_data = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }])

    # -------------------------------
    # FEATURE ENGINEERING (MUST MATCH TRAINING)
    # -------------------------------

    input_data['avg_monthly_spend'] = input_data['TotalCharges'] / (input_data['tenure'] + 1)

    service_cols = [
        'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]

    input_data['service_count'] = input_data[service_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)

    input_data['tenure_bucket'] = pd.cut(
        input_data['tenure'],
        bins=[0,12,36,100],
        labels=['New','Medium','Loyal']
    )

    input_data['tenure_bucket'] = input_data['tenure_bucket'].astype(str)

    input_data['high_value_customer'] = (input_data['MonthlyCharges'] > 70).astype(int)

    input_data['support_risk_score'] = (
        (input_data['OnlineSecurity'] == 'No').astype(int) +
        (input_data['TechSupport'] == 'No').astype(int)
    )

    # -------------------------------
    # PREDICTION
    # -------------------------------

    prob = model.predict_proba(input_data)[0][1]

    # -------------------------------
    # DECISION INTELLIGENCE
    # -------------------------------

    if prob < 0.3:
        risk = "🟢 Low"
        action = "No Action Required"
    elif prob < 0.7:
        risk = "🟡 Medium"
        action = "Send Promotional Offer"
    else:
        risk = "🔴 High"
        action = "Call Customer + Provide Retention Discount"

    # -------------------------------
    # OUTPUT
    # -------------------------------

    st.subheader("📈 Prediction Result")

    st.metric("Churn Probability", round(prob, 2))
    st.metric("Risk Level", risk)

    st.write("### 💡 Recommended Action")
    st.success(action)

    # Extra insight
    st.write("### 🔍 Key Drivers (Business Logic)")
    if MonthlyCharges > 80:
        st.write("- High Monthly Charges detected")
    if tenure < 12:
        st.write("- Customer is new (high churn risk)")
    if TechSupport == "No":
        st.write("- No tech support increases churn risk")