import joblib
import pandas as pd

from utils import PROCESSED_DATA_DIR, MODELS_DIR

DATA_FILE = PROCESSED_DATA_DIR / "data_features.csv"
MODEL_FILE = MODELS_DIR / "air_quality_model.pkl"
THRESHOLD_FILE = MODELS_DIR / "threshold.pkl"


def main():
    print("Running predictions...")

    model = joblib.load(MODEL_FILE)
    threshold = joblib.load(THRESHOLD_FILE)

    df = pd.read_csv(DATA_FILE)

    X = df.drop(columns=["target", "time"])

    probs = model.predict_proba(X)[:, 1]
    df["worsening_probability"] = probs
    df["worsening_flag"] = (probs >= threshold).astype(int)

    output = PROCESSED_DATA_DIR / "predictions.csv"
    df.to_csv(output, index= False)

    print(f"Predictions saved to: {output}")

if __name__ == "__main__":
    main()