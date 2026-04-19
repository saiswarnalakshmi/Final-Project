import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    confusion_matrix,
    roc_curve,
    precision_score,
    recall_score,
    f1_score
)

#Load Model

model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")

st.set_page_config(page_title="Churn Prediction App", layout="wide")

st.title("📊 Telecom Customer Churn Prediction System")

#Input

st.sidebar.header("Customer Inputs")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
tenure = st.sidebar.slider("Tenure", 0, 72, 12)

phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

monthly = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 50.0)

#Feature Engineering

def feature_engineering(df):
    df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
    df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)
    return df

##Prediction

if st.sidebar.button("Predict"):

    data = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "InternetService": internet,
        "Contract": contract,
        "MonthlyCharges": monthly
    }

    df = pd.DataFrame([data])
    df = feature_engineering(df)

    df = pd.get_dummies(df)
    df = df.reindex(columns=columns, fill_value=0)

    
    prob = model.predict_proba(df)[0][1]

    # Threshold (customizable)
    threshold = st.sidebar.slider("Threshold", 0.1, 0.9, 0.5)

    pred = (prob >= threshold).astype(int)

    # Risk label
    if prob > 0.6:
        risk = "🔴 High Risk"
    elif prob > 0.3:
        risk = "🟡 Medium Risk"
    else:
        risk = "🟢 Low Risk"

    st.subheader("🔮 Prediction Result")
    st.write(f"Churn Probability: **{round(prob, 2)}**")
    st.write(f"Risk Level: {risk}")

#Model Evaluation

st.header("📊 Model Evaluation")

if st.button("Run Evaluation"):

    df_full = pd.read_csv("customer_churn_data.csv")

    
    df_full["TotalCharges"] = pd.to_numeric(df_full["TotalCharges"], errors="coerce")
    df_full.fillna(0, inplace=True)
    df_full.drop("customerID", axis=1, inplace=True)
    df_full["Churn"] = df_full["Churn"].map({"Yes": 1, "No": 0})

    df_full = pd.get_dummies(df_full, drop_first=True)

    X = df_full.drop("Churn", axis=1)
    y = df_full["Churn"]

    X = X.reindex(columns=columns, fill_value=0)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Evaluation Metrics

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    roc = roc_auc_score(y, y_prob)

    st.write("Precision:", round(precision, 3))
    st.write("Recall:", round(recall, 3))
    st.write("F1-score:", round(f1, 3))
    st.write("ROC-AUC:", round(roc, 3))

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    st.pyplot(fig)

    # ROC Curve
    fpr, tpr, _ = roc_curve(y, y_prob)

    fig2, ax2 = plt.subplots()
    ax2.plot(fpr, tpr)
    ax2.set_title("ROC Curve")
    st.pyplot(fig2)

# SHAP Analysis

st.header("SHAP Explainability")

if st.button("Run SHAP Analysis"):

    df_shap = pd.read_csv("customer_churn_data.csv")

    df_shap["TotalCharges"] = pd.to_numeric(df_shap["TotalCharges"], errors="coerce")
    df_shap.fillna(0, inplace=True)
    df_shap.drop("customerID", axis=1, inplace=True)
    df_shap["Churn"] = df_shap["Churn"].map({"Yes": 1, "No": 0})

    df_shap = pd.get_dummies(df_shap, drop_first=True)

    X = df_shap.drop("Churn", axis=1)
    X = X.reindex(columns=columns, fill_value=0)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    st.subheader("Feature Importance")

    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], X, show=False)
    else:
        shap.summary_plot(shap_values, X, show=False)

    st.pyplot(plt.gcf())