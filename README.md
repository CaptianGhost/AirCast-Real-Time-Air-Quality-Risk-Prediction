# Air Quality Worsening Prediction (ML Pipeline)

A structured end-to-end Machine Learning pipeline that predicts whether **PM2.5 air quality will worsen in the next hour** using lag, momentum, and rolling statistical features.

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
- L1-regularized Logistic Regression
- Saved model + threshold
- Seprate prediction script

---

## Project Structure

```
project/
|
|-- data/
|   |-- processed/
|   |-- raw/
|
|-- models/
|
|-- src/
|   |-- fetch_api.py
|   |-- features.py
|   |-- train.py
|   |-- predict.py
|   |-- utils.py
|
|-- requirements.txt
|-- README.md
```

---

## Pipeline Flow

### [1] Fetch Data

Fetch hourly air quality data from Open-Meteo API:

```
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

- 80/20 time-based split
- StandardScaler
- L1 Logistic Regression (saga solver)
- ROC-AUC evaluation
- Saves model + threshold

```markdown
python src/train.py
```

Outputs:
```markdown
models/air_quality_model.pkl
models/threshold.pkl
```

---

### [4] Run Predictions

Generate probability and classification flag:

```markdown
python src/predict.py
```

Output:
```markdown
data/processed/predictions.csv
```

---

## Model Details

- Algorithm: Logistic Regression
- Regularization: L1 (feature selection)
- Solver: saga
- Time-aware split (no leakage)
- Evaluation metric: ROC-AUC

---

## Key Learning outcomes

- Proper ML project structuring
- Time-series safe splitting
- Feature engineering for temporal data
- Regularization-based feature selection
- Clean separation between training and inference
- Reproducible model saving & loading

---

## Requirements

Python 3.10+

Libraries:

- pandas
- scikit-learn
- joblib
- requests

Install with:

```markdown
pip install -r requirements.txt
```

---

## Future Improvements

- Cross-validation with time series splits
- Threshold optimization
- More advanced models (tree-based, boosting)
- Live prediction API deployment

---

## Author

*This project is part of my learning journey. Feedback and suggestions are always welcome.*

**- Ghost**
