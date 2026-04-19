from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# ✅ Load correct files
model = joblib.load("models/model.pkl")
columns = joblib.load("models/columns.pkl")   # IMPORTANT

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])

        # ✅ Feature Engineering (same as training)
        df["TotalCharges"] = df["MonthlyCharges"] * df["tenure"]
        df["AvgMonthlySpend"] = df["TotalCharges"] / (df["tenure"] + 1)

        # ✅ Encoding
        df = pd.get_dummies(df)

        # ✅ Match training columns
        df = df.reindex(columns=columns, fill_value=0)

        # ✅ Prediction
        prob = model.predict_proba(df)[0][1]

        risk = "High" if prob > 0.6 else "Medium" if prob > 0.3 else "Low"

        return {
            "churn_probability": round(prob, 2),
            "churn_risk": risk
        }

    except Exception as e:
        return {"error": str(e)}