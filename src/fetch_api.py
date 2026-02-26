import requests
import pandas as pd
from utils import RAW_DATA_DIR

URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

params = {
    "latitude": 28.61,
    "longitude": 77.21,
    "hourly": "pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,ozone",
    "timezone": "auto"
}


def fetch_air_quality():
    print("Fetching air quality data from API...")
    response = requests.get(URL, params=params)
    data = response.json()

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])

    output_path = RAW_DATA_DIR / "data_raw_air_quality.csv"
    df.to_csv(output_path, index= False)

    print(f"Saved raw data to: {output_path}")
    return df

if __name__ == "__main__":
    fetch_air_quality()