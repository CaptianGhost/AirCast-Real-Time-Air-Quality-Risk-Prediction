import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report

from utils import PROCESSED_DATA_DIR, MODELS_DIR


DATA_FILE = PROCESSED_DATA_DIR / "data_features.csv"
MODEL_FILE = MODELS_DIR / "air_quality_model.pkl"
THRESHOLD_FILE = MODELS_DIR / "threshold.pkl"


def main():
    print("Training model...")

    df = pd.read_csv(DATA_FILE)

    y = df["target"]
    X = df.drop(columns=["target", "time"])

    split = int(len(df) * 0.8)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    pipe = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                l1_ratio= 1.0, solver= "saga", max_iter= 5000,
                C= 0.5, random_state= 42
            ))
        ]
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:,1]
    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))



    joblib.dump(pipe, MODEL_FILE)
    joblib.dump(threshold, THRESHOLD_FILE)

    print("Model saved.")


if __name__ == "__main__":
    main()