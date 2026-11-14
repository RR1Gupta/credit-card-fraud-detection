"""REST API (FastAPI).  Run:  uvicorn app.api:app --reload   then open http://127.0.0.1:8000/docs"""
import logging, os, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, create_model
from fraud_features import RAW_COLUMNS, load_bundle, predict_frame

log = logging.getLogger("fraud-api")
MODEL_PATH = Path(os.getenv("MODEL_PATH", ROOT / "model" / "fraud_model.joblib"))

_fields = {c: (float, Field(..., ge=0, allow_inf_nan=False) if c in ("Time", "Amount")
               else Field(..., allow_inf_nan=False)) for c in RAW_COLUMNS}
Transaction = create_model("Transaction", **_fields)


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


app = FastAPI(title="Credit Card Fraud Detection API", version="1.0")
_bundle = None


def get_bundle():
    global _bundle
    if _bundle is None:
        if not MODEL_PATH.exists():
            raise HTTPException(503, f"Model file not found at {MODEL_PATH}. Train it first.")
        _bundle = load_bundle(MODEL_PATH)
    return _bundle


def _run(rows):
    b = get_bundle()
    try:
        return b, predict_frame(b, pd.DataFrame(rows))
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception:
        log.exception("prediction failed")
        raise HTTPException(500, "Internal error while predicting")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


@app.get("/model-info")
def model_info():
    return get_bundle()["meta"]


@app.post("/predict")
def predict(t: Transaction):
    b, r = _run([t.model_dump()])
    return {"fraud_probability": float(r.fraud_probability[0]), "is_fraud": bool(r.is_fraud[0]),
            "threshold": b["threshold"]}


@app.post("/predict-batch")
def predict_batch(req: BatchRequest):
    b, r = _run([t.model_dump() for t in req.transactions])
    return {"count": len(r), "fraud_count": int(r.is_fraud.sum()), "threshold": b["threshold"],
            "predictions": r.to_dict(orient="records")}
