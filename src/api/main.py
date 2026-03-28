import pandas as pd
from fastapi import FastAPI
import requests
from pydantic import BaseModel
import joblib
from pathlib import Path

app = FastAPI()

MODEL_PATH = Path("models/air_quality_model.pkl")
THRESHOLD_PATH = Path("models/threshold.pkl")
FEATURE_COLUMNS_PATH = Path("models/feature_columns.pkl")

model = joblib.load(MODEL_PATH)
threshold = joblib.load(THRESHOLD_PATH)
feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

class AirQuality(BaseModel):
    city: str

# Convert city name -> latitude & longitude
def get_coords(city):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    params = {
        "name": city,
        "count": 1
    }

    response = requests.get(url, params=params)
    data = response.json()

    if "results" not in data:
        raise ValueError("City not found")


    lat = data["results"][0]["latitude"]
    lon = data["results"][0]["longitude"]

    return lat, lon

# Fetching the AirQuality data
def get_airquality(city):
    lat, lon = get_coords(city)

    URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone",
        "timezone": "auto"
    }

    response = requests.get(URL, params=params)
    data = response.json()
    raw_df = pd.DataFrame(data["hourly"])

    return raw_df

# Feature building
def build_airquality_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("time").reset_index(drop=True)

    pollutants = [
        "pm10", "pm2_5", "carbon_monoxide", "nitrogen_dioxide", "ozone"
    ]

    # Lag feature (previous hour)
    for cols in pollutants:
        df[f"{cols}_lag1"] = df[cols].shift(1)
        df[f"{cols}_lag2"] = df[cols].shift(2)

    # Change features
    for cols in pollutants:
        df[f"{cols}_change_1h"] = df[cols] - df[f"{cols}_lag1"]
        df[f"{cols}_change_2h"] = df[cols] - df[f"{cols}_lag2"]

    # Rolling mean (last 3 hours)
    for cols in pollutants:
        df[f"{cols}_roll3"] = df[cols].rolling(window=3, min_periods=3).mean()

    df = df.dropna()

    return df

@app.post("/air-quality")
def predict_airquality(city: AirQuality):

    try:
        raw_df = get_airquality(city.city)
    except ValueError:
        return {"error": "City not found"}

    features = build_airquality_features(raw_df)

    latest = features.iloc[[-1]]  # use the most recent time step for prediction
    latest = latest[feature_columns]

    probability = model.predict_proba(latest)[0][1]
    prediction = int(probability >= threshold)
    label = "Worsening" if prediction == 1 else "Stable"

    return {
        "air_worsening_probability": round(float(probability), 2),
        "prediction": prediction,
        "label": label
    }

@app.get("/")
def home():
    return {
        "message": "Air Quality ML API is running",
        "status": "http://127.0.0.1:8000/docs"
            }