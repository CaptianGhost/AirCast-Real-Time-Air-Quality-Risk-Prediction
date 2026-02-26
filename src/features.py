import pandas as pd
from utils import RAW_DATA_DIR, PROCESSED_DATA_DIR

RAW_FILE = RAW_DATA_DIR / "data_raw_air_quality.csv"
OUTPUT_FILE = PROCESSED_DATA_DIR / "data_features.csv"

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("time").reset_index(drop=True)

    pollutants= [
        "pm10", "pm2_5", "nitrogen_dioxide", "carbon_monoxide", "ozone"
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
        df[f"{cols}_roll3"] = df[cols].rolling(window= 3, min_periods= 3).mean()


    df["target"] = (df["pm2_5"].shift(-1) > df["pm2_5"]).astype(int)


    df = df.dropna()

    return df

def main():
    print("Building features...")

    df = pd.read_csv(RAW_FILE, parse_dates=["time"])
    df_features = build_features(df)

    df_features.to_csv(OUTPUT_FILE, index= False)
    print(f"Saved processed data to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()