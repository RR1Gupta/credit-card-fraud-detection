"""Demo UI.  Run:  streamlit run app/streamlit_app.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, streamlit as st, sklearn
from fraud_features import RAW_COLUMNS, V_COLS, load_bundle, predict_frame

st.set_page_config(page_title="Credit Card Fraud Detection", page_icon="💳", layout="wide")
st.title("💳 Credit Card Fraud Detection")
st.caption("Educational prototype trained on the 2013 Kaggle/ULB dataset. V1-V28 are anonymized PCA features.")


@st.cache_resource
def get_bundle():
    return load_bundle(ROOT / "model" / "fraud_model.joblib")


try:
    bundle = get_bundle()
except Exception as e:
    st.error(f"Model could not be loaded: {e}")
    st.stop()
meta = bundle["meta"]
if meta.get("sklearn_version") != sklearn.__version__:
    st.warning(f"Model trained with scikit-learn {meta.get('sklearn_version')}, running {sklearn.__version__}.")

samples = pd.read_csv(ROOT / "model" / "sample_transactions.csv")
t1, t2, t3 = st.tabs(["Single transaction", "Batch CSV", "Model info"])

with t1:
    i = st.selectbox("Pick a sample transaction", range(len(samples)),
                     format_func=lambda k: f"Sample {k + 1} (Amount {samples.Amount[k]:.2f})")
    row = samples.iloc[i]
    c1, c2 = st.columns(2)
    vals = {"Time": c1.number_input("Time (seconds)", min_value=0.0, value=float(row.Time), key=f"t{i}"),
            "Amount": c2.number_input("Amount", min_value=0.0, value=float(row.Amount), key=f"a{i}")}
    with st.expander("V1-V28 (anonymized features)"):
        cols = st.columns(4)
        for j, v in enumerate(V_COLS):
            vals[v] = cols[j % 4].number_input(v, value=float(row[v]), format="%.4f", key=f"{v}{i}")
    if st.button("Predict", type="primary"):
        r = predict_frame(bundle, pd.DataFrame([vals]))
        p = float(r.fraud_probability[0])
        (st.error if r.is_fraud[0] else st.success)(
            f"{'⚠️ Likely FRAUD' if r.is_fraud[0] else '✅ Likely genuine'}  |  fraud probability = {p:.3%}")
        st.caption(f"Decision threshold: {bundle['threshold']:.3f}. Actual label of this sample: "
                   f"{'fraud' if row.Class == 1 else 'genuine'}")

with t2:
    st.write("Upload a CSV with columns: " + ", ".join(RAW_COLUMNS))
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if len(df) > 50000:
                st.error("Max 50,000 rows."); st.stop()
            out = pd.concat([df.reset_index(drop=True), predict_frame(bundle, df).reset_index(drop=True)], axis=1)
            st.success(f"{len(out)} rows scored, {int(out.is_fraud.sum())} flagged as fraud.")
            st.dataframe(out.sort_values("fraud_probability", ascending=False).head(500))
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv")
        except ValueError as e:
            st.error(f"Invalid file: {e}")
        except Exception as e:
            st.error(f"Could not process file: {e}")

with t3:
    st.json(meta)
