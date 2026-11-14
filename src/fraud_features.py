"""Shared feature engineering + model helpers (used by training, API and Streamlit app)."""
import numpy as np, pandas as pd, joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, RobustScaler

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLUMNS = ["Time"] + V_COLS + ["Amount"]          # the 30 inputs a user must provide
NUM_SCALE = ["Time", "Log_Amount", "Transaction_Hour"]


def build_features(X):
    d = pd.DataFrame(X)[RAW_COLUMNS].copy()
    d["Log_Amount"] = np.log1p(d["Amount"].clip(lower=0))
    d["Transaction_Hour"] = ((d["Time"] // 3600) % 24).astype(int)
    d["Amount_Category"] = pd.cut(d["Amount"], [-1, 10, 100, 1000, np.inf], labels=[0, 1, 2, 3]).astype(int)
    return d[V_COLS + ["Time", "Log_Amount", "Transaction_Hour", "Amount_Category"]]


def make_pipeline(clf):
    """raw 30 columns -> engineered features -> RobustScaler (train-fitted) -> classifier"""
    return Pipeline([
        ("fe", FunctionTransformer(build_features, validate=False)),
        ("prep", ColumnTransformer([("scale", RobustScaler(), NUM_SCALE)], remainder="passthrough")),
        ("clf", clf),
    ])


def validate_frame(df):
    """Raise ValueError with a clear message for bad input."""
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    d = df[RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if d.isnull().any().any() or np.isinf(d.to_numpy(dtype=float)).any():
        bad = d.columns[d.isnull().any() | np.isinf(d.to_numpy(dtype=float)).any(axis=0)].tolist()
        raise ValueError(f"Missing, non-numeric or infinite values in: {bad}")
    if (d["Amount"] < 0).any():
        raise ValueError("Amount must be >= 0")
    if (d["Time"] < 0).any():
        raise ValueError("Time must be >= 0")
    return d


def load_bundle(path):
    return joblib.load(path)


def predict_frame(bundle, df):
    d = validate_frame(df)
    prob = bundle["pipeline"].predict_proba(d)[:, 1]
    return pd.DataFrame({"fraud_probability": prob.round(6),
                         "is_fraud": (prob >= bundle["threshold"]).astype(int)}, index=df.index)
