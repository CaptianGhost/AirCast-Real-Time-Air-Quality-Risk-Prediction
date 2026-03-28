# Air Quality Worsening Prediction (ML Pipeline + FastAPI API)

A structured end-to-end Machine Learning pipeline that predicts whether **PM2.5 levels will worsen in the next hour**, with a deployed **FastAPI inference API** using live data.

---

## Motivation

This project explores whether short-term pollution trends can be used to predict **near-term worsening in PM2.5 levels**, while practicing a structured end-to-end machine learning workflow — from data ingestion to a live prediction API.

---

## Project Overview

The model predicts:

> Will PM2.5 increase in the next hour?

Target definition:

`target = 1 if pm2_5(t+1) > pm2_5(t)`

`target = 0` otherwise

The pipeline includes:

- Fetching API data 
- Feature engineering (lags, momentum, rolling means)
- Time-aware train/test split
- Regularized classification models
- Serialized model, decision threshold, and features columns
- **Live FastAPI prediction endpoint**

---

## Project Structure

```markdown
project/
|
|-- data/
|   |-- processed/
|   |-- raw/
|
|-- models/
|
|-- src/
|   |-- api/
|   |    |-- main.py
|   |-- fetch_api.py
|   |-- features.py
|   |-- train.py
|   |-- predict.py
|   |-- utils.py
|
|-- requirements.txt
|-- LICENSE
|-- README.md
```

---

## Pipeline Flow

### [1] Fetch Data

Fetch hourly air quality data from Open-Meteo API:

```commandline
python src/fetch_api.py
```

Saves raw data to:

```
data/raw/data_raw_air_quality.csv
```

---

### [2] Build Features

Creates:

- Lag features (t-1, t-2)
- Change features (1-hour & 2-hour momentum)
- Rolling mean (3-hour window)
- Binary target

```markdown
data/processed/data_features.csv
```

---

### [3] Train Model

- Time-aware split (no leakage)
- Regularized classification baseline
- XGBoost (final selected model)
- ROC-AUC evaluation
- Probability-based decision threshold (default 0.5)
- Saves model artifacts

```commandline
python src/train.py
```

Outputs:

```markdown
models/air_quality_model.pkl
models/threshold.pkl
models/feature_columns.pkl
```

---

### [4] Offline Predictions

Generate probability and classification flag:

```commandline
python src/predict.py
```

Output:
```markdown
data/processed/predictions.csv
```

---

## Live Prediction API (FastAPI)

The project includes a **production-style ML inference API** that:

- Accepts a city name
- Fetches live air quality data
- Builds features dynamically
- Applies trained model
- Returns predictions + probability

### Run the API

```commandline
uvicorn src.api.main:app --reload
```

Open Swagger UI:

```http request
http://127.0.0.1:8000/docs
```

---

#### Example Request

`POST /air-quality`

```json
{
  "city": "Delhi"
}
```

---

#### Example Response

```json
{
  "air_worsening_probability": 0.91,
  "prediction": 1,
  "label": "Worsening"
}
```

---

## Model Details

- Final Model: XGBoost Classifier
- Compared against Logistic Regression and Random Forest
- Selected based on ROC-AUC performance
- Probability-based classification with explicit decision threshold

---

## Key Learning Outcomes

- End-to-end ML pipeline design
- Time-series feature engineering
- Model comparison & selection (including XGBoost)
- Threshold-based decision-making
- Model serialization & reproducibility
- FastAPI ML inference deployment
- Live external API integration
- Training vs inference consistency

---

## Requirements

```requirements
Python 3.10+

Libraries:
- pandas
- scikit-learn
- joblib
- requests
- fastapi
- uvicorn
- xgboost
```

Install with:

```commandline
pip install -r requirements.txt
```

---

## Future Improvements

- Model monitoring
- Scheduled retraining
- Docker deployment
- Cloud hosting
- Additional pollutant forecasting
- Multi-city batch prediction endpoint

---

## Author

*This project is part of my learning journey. Feedback and suggestions are always welcome.*

**- Ghost**
