import joblib
import pandas as pd

from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report

from utils import PROCESSED_DATA_DIR, MODELS_DIR


DATA_FILE = PROCESSED_DATA_DIR / "data_features.csv"
MODEL_FILE = MODELS_DIR / "air_quality_model.pkl"
THRESHOLD_FILE = MODELS_DIR / "threshold.pkl"
FEATURE_COLUMNS =MODELS_DIR / "feature_columns.pkl"

def main():
    print("Training model...")

    df = pd.read_csv(DATA_FILE)

    y = df["target"]

    X = df.drop(columns=["target", "time"])
    # Save feature order for inference-time consistency
    feature_columns = X.columns
    joblib.dump(feature_columns, FEATURE_COLUMNS)


    split = int(len(df) * 0.8)   # Time-aware split (avoid leakage)

    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=10,
        random_state=42
    )
    # XGBoost with early stopping on validation set
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )

    probs = model.predict_proba(X_test)[:,1]
    threshold = 0.5
    preds = (probs >= threshold).astype(int)

    print("ROC-AUC:", roc_auc_score(y_test, probs))
    print(classification_report(y_test, preds))


    # Persist model artifacts for API inference
    joblib.dump(model, MODEL_FILE)
    joblib.dump(threshold, THRESHOLD_FILE)

    print("Model saved.")


if __name__ == "__main__":
    main()