"""Run:  pytest -q   (needs model/fraud_model.joblib; tests are skipped if it is missing)"""
import sys
from pathlib import Path
import pandas as pd, pytest
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "src"))
if not (ROOT / "model" / "fraud_model.joblib").exists():
    pytest.skip("model not trained yet", allow_module_level=True)
from fastapi.testclient import TestClient
from app.api import app
from fraud_features import RAW_COLUMNS

client = TestClient(app)
samples = pd.read_csv(ROOT / "model" / "sample_transactions.csv")
good = samples.iloc[0][RAW_COLUMNS].to_dict()


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_model_info():
    assert "threshold" in client.get("/model-info").json()


def test_predict_valid():
    r = client.post("/predict", json=good)
    assert r.status_code == 200
    assert 0 <= r.json()["fraud_probability"] <= 1 and isinstance(r.json()["is_fraud"], bool)


def test_predict_known_fraud_rows_run():
    for _, row in samples.iterrows():
        assert client.post("/predict", json=row[RAW_COLUMNS].to_dict()).status_code == 200


def test_missing_field():
    bad = dict(good); bad.pop("V5")
    assert client.post("/predict", json=bad).status_code == 422


def test_negative_amount():
    assert client.post("/predict", json={**good, "Amount": -5}).status_code == 422


def test_non_numeric():
    assert client.post("/predict", json={**good, "V1": "abc"}).status_code == 422


def test_batch():
    r = client.post("/predict-batch", json={"transactions": [good, good]})
    assert r.status_code == 200 and r.json()["count"] == 2


def test_batch_empty_and_too_large():
    assert client.post("/predict-batch", json={"transactions": []}).status_code == 422
    assert client.post("/predict-batch", json={"transactions": [good] * 1001}).status_code == 422
