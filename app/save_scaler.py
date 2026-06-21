# save as: app/save_scaler.py
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent / "data" / "MachineLearningCVE"
ARTIFACTS_DIR = SCRIPT_DIR.parent / "artifacts"

# Load the existing LightGBM to get the exact 78 feature names in training order
lgbm = joblib.load(ARTIFACTS_DIR / "tier1_lightgbm.pkl")
feature_names = lgbm.feature_name()
print(f"Feature count: {len(feature_names)}")
print(f"First 5: {feature_names[:5]}")

# Read all CSVs, take only the 80% training slice (chronological), sample down
print("\nLoading CSVs to refit scaler...")
train_dfs = []

for file in sorted(DATA_DIR.glob("*.csv")):
    try:
        df = pd.read_csv(file, encoding="utf-8", on_bad_lines="skip")
    except Exception:
        try:
            df = pd.read_csv(file, encoding="latin-1", on_bad_lines="skip")
        except Exception as e:
            print(f"  Skipping {file.name}: {e}")
            continue

    df.columns = df.columns.str.strip()

    # Drop non-numeric and label columns
    df = df.drop(columns=["Label"], errors="ignore")

    # Keep only the 78 training features, fill missing with 0
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0.0

    df = df[feature_names]

    # Remove inf/nan
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Take training slice (first 80%)
    cutoff = int(len(df) * 0.8)
    train_dfs.append(df.iloc[:cutoff])
    print(f"  {file.name}: {cutoff} rows")

full_train = pd.concat(train_dfs, ignore_index=True)
print(f"\nTotal training rows: {len(full_train):,}")

# Fit scaler on all training data (same as original training did)
print("Fitting StandardScaler...")
scaler = StandardScaler()
scaler.fit(full_train)

# Save
out_path = ARTIFACTS_DIR / "feature_scaler.pkl"
joblib.dump(scaler, out_path)
print(f"\nScaler saved to: {out_path}")
print("Done.")