"""
Credit Card Fraud Detection using ML with Imbalanced Data Handling and Explainable AI
Run:  python src/fraud_detection.py --data data/creditcard.csv
Colab: upload creditcard.csv, then  !python src/fraud_detection.py --data creditcard.csv
All outputs (tables + plots) go to results/
"""
import argparse, os, warnings
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score,
                             average_precision_score, roc_auc_score, confusion_matrix,
                             precision_recall_curve)
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
SEED = 42
SCALERS = {"Standard": StandardScaler, "Robust": RobustScaler, "MinMax": MinMaxScaler}


# ---------------------------------------------------------------- Phase 1: EDA
def eda(df, out):
    print("Shape:", df.shape)
    print("Missing values:", int(df.isnull().sum().sum()))
    print("Duplicate rows:", int(df.duplicated().sum()))
    print(df["Class"].value_counts())
    print(f"Fraud %: {df['Class'].mean() * 100:.3f}")
    df.describe().T.to_csv(f"{out}/eda_summary_stats.csv")

    plt.figure(figsize=(5, 4))
    sns.countplot(x="Class", data=df); plt.yscale("log")
    plt.title("Class distribution (log scale)"); plt.tight_layout()
    plt.savefig(f"{out}/eda_class_distribution.png", dpi=150); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(df["Amount"], bins=60, ax=ax[0]); ax[0].set_title("Amount"); ax[0].set_yscale("log")
    sns.histplot(np.log1p(df["Amount"]), bins=60, ax=ax[1]); ax[1].set_title("log(1 + Amount)")
    plt.tight_layout(); plt.savefig(f"{out}/eda_amount_hist.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Class", y=np.log1p(df["Amount"]), data=df)
    plt.title("log(1+Amount) by class"); plt.tight_layout()
    plt.savefig(f"{out}/eda_amount_by_class.png", dpi=150); plt.close()

    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0)
    plt.title("Correlation heatmap"); plt.tight_layout()
    plt.savefig(f"{out}/eda_correlation_heatmap.png", dpi=150); plt.close()

    corr = df.corr()["Class"].drop("Class").sort_values(key=abs, ascending=False).head(10)
    corr.to_csv(f"{out}/eda_top_correlated_with_class.csv")


# ------------------------------------------- Phase 2-3: preprocessing + features
def add_features(df):
    d = df.copy()
    d["Log_Amount"] = np.log1p(d["Amount"])
    d["Transaction_Hour"] = ((d["Time"] // 3600) % 24).astype(int)
    d["Amount_Category"] = pd.cut(d["Amount"], [-1, 10, 100, 1000, np.inf], labels=[0, 1, 2, 3]).astype(int)
    return d


def split(df, mode, test_size=0.2):
    if mode == "chronological":     # train on earlier, test on later transactions
        df = df.sort_values("Time").reset_index(drop=True)
        cut = int(len(df) * (1 - test_size))
        return df.iloc[:cut].copy(), df.iloc[cut:].copy()
    return train_test_split(df, test_size=test_size, stratify=df["Class"], random_state=SEED)


def scale(train, test, cols, scaler_name):
    """Scaler fitted ONLY on training data (no leakage)."""
    train, test = train.copy(), test.copy()
    sc = SCALERS[scaler_name]()
    train[cols] = sc.fit_transform(train[cols])
    test[cols] = sc.transform(test[cols])
    return train, test


def get_xy(train, test, feature_set, scaler_name):
    if feature_set == "base":
        num = ["Time", "Amount"]
        feats = [f"V{i}" for i in range(1, 29)] + num
    else:
        num = ["Time", "Log_Amount", "Transaction_Hour"]
        feats = [f"V{i}" for i in range(1, 29)] + num + ["Amount_Category"]
    tr, te = scale(train, test, num, scaler_name)
    return tr[feats], tr["Class"], te[feats], te["Class"], feats


# ------------------------------------------------------- Phase 6-7: models
def make_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=SEED)
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=8, random_state=SEED)
    return RandomForestClassifier(n_estimators=100, max_depth=12, n_jobs=-1, random_state=SEED)


def fit(name, Xtr, ytr, use_smote):
    if use_smote:                    # SMOTE applied to TRAINING data only
        Xtr, ytr = SMOTE(random_state=SEED).fit_resample(Xtr, ytr)
    m = make_model(name)
    m.fit(Xtr, ytr)
    return m


# ---------------------------------------------------- Phase 8: evaluation
def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), F1=f1_score(y, pred), PR_AUC=average_precision_score(y, prob),
                ROC_AUC=roc_auc_score(y, prob), TN=tn, FP=fp, FN=fn, TP=tp)


def plot_cm(y, prob, title, path, thr=0.5):
    cm = confusion_matrix(y, (prob >= thr).astype(int), labels=[0, 1])
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(title, fontsize=9); plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()


def main(a):
    out = a.out; os.makedirs(out, exist_ok=True)
    df = pd.read_csv(a.data)
    eda(df, out)

    n0 = len(df)
    if not a.keep_duplicates:
        df = df.drop_duplicates().reset_index(drop=True)
    print(f"Duplicates removed: {n0 - len(df)}")
    if a.fast:
        df = pd.concat([df[df.Class == 1], df[df.Class == 0].sample(frac=0.1, random_state=SEED)]).reset_index(drop=True)
    df = add_features(df)

    train, test = split(df, a.split)
    print(f"Split={a.split} | train={len(train)} (fraud {int(train.Class.sum())}) | test={len(test)} (fraud {int(test.Class.sum())})")
    if test.Class.sum() == 0:
        raise SystemExit("No fraud in test set; use --split random")

    # ---- Main grid: 3 models x {Original, SMOTE}, engineered features, RobustScaler
    Xtr, ytr, Xte, yte, feats = get_xy(train, test, "engineered", "Robust")
    rows, probs, fitted = [], {}, {}
    for name in ["Logistic Regression", "Decision Tree", "Random Forest"]:
        for ver, sm in [("Original", False), ("SMOTE", True)]:
            m = fit(name, Xtr, ytr, sm)
            p = m.predict_proba(Xte)[:, 1]
            probs[(name, ver)], fitted[(name, ver)] = p, m
            rows.append(dict(Model=name, Dataset=ver, **metrics(yte, p)))
            plot_cm(yte, p, f"{name} ({ver})", f"{out}/cm_{name.replace(' ', '_')}_{ver}.png")
            print(rows[-1])
    res = pd.DataFrame(rows).round(4)
    res.to_csv(f"{out}/model_comparison.csv", index=False)
    print("\n", res[["Model", "Dataset", "Precision", "Recall", "F1", "PR_AUC"]].to_string(index=False))

    plt.figure(figsize=(7, 5))
    for (name, ver), p in probs.items():
        pr, rc, _ = precision_recall_curve(yte, p)
        plt.plot(rc, pr, label=f"{name} ({ver})")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall curves"); plt.legend(fontsize=7)
    plt.tight_layout(); plt.savefig(f"{out}/pr_curves.png", dpi=150); plt.close()

    # ---- Ablation 1: scaling comparison (Logistic Regression is scale-sensitive)
    rows = []
    for s in SCALERS:
        a_tr, a_ytr, a_te, a_yte, _ = get_xy(train, test, "engineered", s)
        p = fit("Logistic Regression", a_tr, a_ytr, False).predict_proba(a_te)[:, 1]
        rows.append(dict(Scaler=s, **metrics(a_yte, p)))
    pd.DataFrame(rows).round(4).to_csv(f"{out}/ablation_scaling.csv", index=False)

    # ---- Ablation 2: feature engineering (base vs engineered) with Random Forest
    rows = []
    for fs in ["base", "engineered"]:
        a_tr, a_ytr, a_te, a_yte, _ = get_xy(train, test, fs, "Robust")
        p = fit("Random Forest", a_tr, a_ytr, False).predict_proba(a_te)[:, 1]
        rows.append(dict(FeatureSet=fs, **metrics(a_yte, p)))
    pd.DataFrame(rows).round(4).to_csv(f"{out}/ablation_feature_engineering.csv", index=False)

    # ---- Threshold tuning (threshold picked on a validation slice of TRAIN, not on test)
    best_key = max(probs, key=lambda k: average_precision_score(yte, probs[k]))
    bname, bver = best_key
    print("\nBest model by PR-AUC:", best_key)
    tr_sorted = train.sort_values("Time") if a.split == "chronological" else train
    vcut = int(len(tr_sorted) * 0.85)
    sub, val = tr_sorted.iloc[:vcut], tr_sorted.iloc[vcut:]
    if val.Class.sum() > 0:
        s_tr, s_ytr, s_val, s_yval, _ = get_xy(sub, val, "engineered", "Robust")
        m_sub = fit(bname, s_tr, s_ytr, bver == "SMOTE")
        pr, rc, th = precision_recall_curve(s_yval, m_sub.predict_proba(s_val)[:, 1])
        f1s = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
        thr = float(th[np.argmax(f1s)])
        tuned = dict(Model=bname, Dataset=bver, Threshold=round(thr, 4), **metrics(yte, probs[best_key], thr))
        pd.DataFrame([dict(Model=bname, Dataset=bver, Threshold=0.5, **metrics(yte, probs[best_key], 0.5)), tuned]
                     ).round(4).to_csv(f"{out}/threshold_tuning.csv", index=False)
        plot_cm(yte, probs[best_key], f"{bname} ({bver}) tuned thr={thr:.3f}", f"{out}/cm_best_tuned.png", thr)

    # ---- Phase 10: Explainable AI on Random Forest (original data)
    rf = fitted[("Random Forest", "Original")]
    imp = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False).head(15)
    imp.to_csv(f"{out}/rf_feature_importance.csv")
    plt.figure(figsize=(6, 5)); imp[::-1].plot.barh(); plt.title("Random Forest feature importance (top 15)")
    plt.tight_layout(); plt.savefig(f"{out}/rf_feature_importance.png", dpi=150); plt.close()
    try:
        import shap
        fr = Xte[yte == 1]; ge = Xte[yte == 0]
        samp = pd.concat([fr.head(100), ge.sample(min(400, len(ge)), random_state=SEED)])
        sv = shap.TreeExplainer(rf).shap_values(samp)
        sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if np.ndim(sv) == 3 else sv)
        shap.summary_plot(sv, samp, show=False); plt.tight_layout()
        plt.savefig(f"{out}/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
        shap.summary_plot(sv, samp, plot_type="bar", show=False); plt.tight_layout()
        plt.savefig(f"{out}/shap_bar.png", dpi=150, bbox_inches="tight"); plt.close()
        # single-transaction explanation: top contributing features for one fraud case
        pd.Series(sv[0], index=samp.columns).sort_values(key=abs, ascending=False).head(10).to_csv(f"{out}/shap_single_fraud_case.csv")
    except Exception as e:
        print("SHAP skipped:", e)
    print(f"\nDone. See {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--split", choices=["chronological", "random"], default="chronological")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--fast", action="store_true", help="subsample genuine rows for quick test")
    main(ap.parse_args())
