"""
train_model.py
Train a RandomForestRegressor on house_price.csv, evaluate it,
and persist model.pkl + metrics.json to the same directory.
Run from inside house_price_project/:
    python train_model.py
"""

import json
import math
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "house_price.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "metrics.json")

# ── Config ───────────────────────────────────────────────────────────────────
FEATURES = [
    "Square_Footage",
    "Num_Bedrooms",
    "Num_Bathrooms",
    "Year_Built",
    "Lot_Size",
    "Garage_Size",
    "Neighborhood_Quality",
]
TARGET = "House_Price"
MODEL_NAME = "RandomForestRegressor"

# ── Load data ────────────────────────────────────────────────────────────────
print("Loading dataset ...")
df = pd.read_csv(CSV_PATH)
print(f"  Rows: {len(df):,}  |  Columns: {list(df.columns)}")

X = df[FEATURES]
y = df[TARGET]

# ── Train / test split ────────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"  Train samples: {len(X_train):,}  |  Test samples: {len(X_test):,}")

# ── Train model ───────────────────────────────────────────────────────────────
print("\nTraining RandomForestRegressor (n_estimators=100, random_state=42) ...")
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
print("  Training complete.")

# ── Evaluate ──────────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))

# ── Save model ────────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"\nModel saved  ->  {MODEL_PATH}")

# ── Save metrics ──────────────────────────────────────────────────────────────
metrics = {
    "model_name": MODEL_NAME,
    "features": FEATURES,
    "r2": round(r2, 6),
    "mae": round(mae, 4),
    "rmse": round(rmse, 4),
    "train_samples": len(X_train),
    "test_samples": len(X_test),
    "dataset_rows": len(df),
}
with open(METRICS_PATH, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved ->  {METRICS_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("  Model Training Summary")
print("=" * 50)
print(f"  Model        : {MODEL_NAME}")
print(f"  Dataset rows : {len(df):,}")
print(f"  Train samples: {len(X_train):,}")
print(f"  Test samples : {len(X_test):,}")
print(f"  R2           : {r2:.4f}")
print(f"  MAE          : {mae:,.2f}")
print(f"  RMSE         : {rmse:,.2f}")
print("=" * 50)
