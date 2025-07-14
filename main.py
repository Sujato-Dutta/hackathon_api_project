from fastapi import FastAPI
from pydantic import BaseModel
from enum import Enum
from datetime import date, datetime
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Load models and scalers
rf_model = joblib.load("rf_model.pkl")
lr_model = joblib.load("lr_model.pkl")
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
lstm_model = load_model("lstm_model.keras")

app = FastAPI(title="Inventory Demand & Alerts API", version="1.1")

# ENUMS for Swagger dropdowns
class WeatherEnum(str, Enum):
    Clear = "Clear"
    Rainy = "Rainy"
    Cloudy = "Cloudy"
    Snowy = "Snowy"

class ModelEnum(str, Enum):
    RandomForest = "RandomForest"
    LinearRegression = "LinearRegression"
    LSTM = "LSTM"

# Weather encoding
weather_map = {"Clear": 0, "Rainy": 1, "Cloudy": 2, "Snowy": 3}

# Request schema for prediction
class PredictInput(BaseModel):
    prev_sales: float
    price: float
    weather: WeatherEnum
    model: ModelEnum

# Request schema for alert checking
class AlertInput(BaseModel):
    stock_level: int
    expiry_date: date

# Prediction endpoint
@app.post("/predict", summary="📦 Predict Product Demand")
def predict_demand(data: PredictInput):
    try:
        weather_encoded = weather_map[data.weather.value]
        features = np.array([[data.prev_sales, data.price, weather_encoded]])
        model_type = data.model.value

        if model_type == "RandomForest":
            pred = rf_model.predict(features)[0]
        elif model_type == "LinearRegression":
            pred = lr_model.predict(features)[0]
        elif model_type == "LSTM":
            x_scaled = scaler_x.transform(features)
            pred_scaled = lstm_model.predict(x_scaled.reshape(1, 1, 3))
            pred = scaler_y.inverse_transform(pred_scaled)[0][0]
        else:
            return {"error": "Invalid model type selected"}

        return {"predicted_demand": round(float(pred), 2)}

    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# Inventory Alert endpoint
@app.post("/alerts", summary="⚠️ Check Inventory Alerts")
def check_alerts(data: AlertInput):
    try:
        alerts = []

        if data.stock_level < 10:
            alerts.append("⚠️ Low Stock")
        if data.stock_level > 150:
            alerts.append("⚠️ Overstocked")

        days_left = (data.expiry_date - datetime.today().date()).days
        if days_left <= 2:
            alerts.append("⚠️ Expiring Soon")

        if not alerts:
            return {"status": "✅ All inventory levels are normal."}
        return {"alerts": alerts}

    except Exception as e:
        return {"error": f"Alert check failed: {str(e)}"}
