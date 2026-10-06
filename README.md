# 🏠 House Price Prediction Project

A full-stack machine learning application that predicts residential house prices
using a **Gradient Boosting Regressor** (scikit-learn), exposed via a **FastAPI**
REST backend and visualised through a **Streamlit** frontend.

---

## Project Structure

```
house_price_project/
│
├── data/
│   └── house_price.csv          # Raw dataset (1 000 rows, 7 features + target)
│
├── model/                       # Auto-generated after training
│   ├── model.pkl                # Fitted sklearn Pipeline
│   └── metadata.json            # Feature meta, UI hints, evaluation metrics
│
├── backend/
│   ├── train_model.py           # Data preprocessing, training, evaluation, persistence
│   └── app.py                   # FastAPI server  (port 8000)
│
├── frontend/
│   └── frontend.py              # Streamlit dashboard (port 8501)
│
├── requirements.txt
├── run.ps1                      # One-click startup (Windows PowerShell)
└── README.md
```

---

## Features

| Feature | Type | Description |
|---|---|---|
| Square_Footage | float | Living area in sq ft |
| Num_Bedrooms | int | Number of bedrooms |
| Num_Bathrooms | int | Number of bathrooms |
| Year_Built | int | Year the property was built |
| Lot_Size | float | Lot size in acres |
| Garage_Size | int | Garage capacity (# of cars) |
| Neighborhood_Quality | int | Neighbourhood score 1–10 |
| **House_Price** | float | **Target variable (USD)** |

---

## Quick Start

### 1. Install dependencies

```powershell
cd house_price_project
pip install -r requirements.txt
```

### 2. Train the model

```powershell
python -m backend.train_model
```

Expected output:
```
[train] MAE  : $XX,XXX.XX
[train] RMSE : $XX,XXX.XX
[train] R²   : 0.XXXX
[train] Model saved → model/model.pkl
[train] Metadata saved → model/metadata.json
```

### 3. Start the FastAPI backend

```powershell
uvicorn backend.app:app --reload --port 8000
```

Swagger UI: http://localhost:8000/docs

### 4. Start the Streamlit frontend

```powershell
streamlit run frontend/frontend.py
```

App: http://localhost:8501

### One-click startup (Windows)

```powershell
.\run.ps1
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health check |
| GET | `/metadata` | Model metadata + feature UI hints |
| POST | `/predict` | Predict house price (JSON body) |
| GET | `/feature_importance` | GBR feature importance scores |

### Example prediction request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Square_Footage": 2500,
    "Num_Bedrooms": 3,
    "Num_Bathrooms": 2,
    "Year_Built": 2005,
    "Lot_Size": 3.0,
    "Garage_Size": 1,
    "Neighborhood_Quality": 7
  }'
```

### Example response

```json
{
  "predicted_price": 612345.67,
  "formatted_price": "$612,345.67",
  "confidence_range": {
    "low":  "$551,111.10",
    "high": "$673,580.24"
  }
}
```

---

## Model Details

- **Algorithm**: `GradientBoostingRegressor` (scikit-learn)
- **Pipeline**: `StandardScaler → GradientBoostingRegressor`
- **Hyperparameters**: 300 trees, lr=0.05, max_depth=5
- **Train/Test split**: 80 / 20

---

## Frontend Screens

1. **Prediction card** — large, styled predicted price with ±10 % confidence band
2. **Model metrics banner** — R², MAE, RMSE at a glance
3. **Feature Importance chart** — horizontal bar chart (Plotly)
4. **Input Profile radar** — normalised radar of your current inputs
5. **Price Sensitivity analysis** — vary any feature and watch the price curve
