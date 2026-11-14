"""
Step 2: hyperparameter tuning + final evaluation + save deployable model.
Run:  python src/train_model.py --data data/creditcard.csv
Outputs: model/fraud_model.joblib, model/model_meta.json, model/sample_transactions.csv,
         results/tuning_results.csv, results/final_model_comparison.csv, results/final_*.png
"""
import argparse, json, os, sys, warnings, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, joblib, sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (precision_score, recall_score, f1_score, accuracy_score, average_precision_score,
                             roc_auc_score, confusion_matrix, precision_recall_curve)
from fraud_features import RAW_COLUMNS, make_pipeline

warnings.filterwarnings("ignore")
SEED = 42

SPECS = {   # name: (estimator factory, param grid, n_iter for random search or None for full grid)
    "Logistic Regression": (lambda: LogisticRegression(max_iter=2000, random_state=SEED),
                            {"C": [0.01, 0.1, 1, 10], "class_weight": [None, "balanced"]}, None),
    "Decision Tree": (lambda: DecisionTreeClassifier(random_state=SEED),
                      {"max_depth": [4, 6, 8, 12], "min_samples_leaf": [1, 5, 20], "class_weight": [None, "balanced"]}, None),
    "Random Forest": (lambda: RandomForestClassifier(n_jobs=1, random_state=SEED),
                      {"n_estimators": [50, 100], "max_depth": [8, 10, 12], "min_samples_leaf": [1, 2, 5],
                       "class_weight": [None, "balanced_subsample"]}, 8),
}


def metrics(y, prob, thr=0.5):
    pred = (prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(Accuracy=accuracy_score(y, pred), Precision=precision_score(y, pred, zero_division=0),
                Recall=recall_score(y, pred), F1=f1_score(y, pred), PR_AUC=average_precision_score(y, prob),
                ROC_AUC=roc_auc_score(y, prob), TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))


def best_threshold(y, prob):
    pr, rc, th = precision_recall_curve(y, prob)
    f1 = 2 * pr[:-1] * rc[:-1] / (pr[:-1] + rc[:-1] + 1e-12)
    return float(th[np.argmax(f1)])


def main(a):
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.model_dir, exist_ok=True)
    df = pd.read_csv(a.data).dropna().drop_duplicates().reset_index(drop=True)
    df = df.sort_values("Time").reset_index(drop=True)
    cut = int(len(df) * 0.8)                       # chronological split: past -> train, future -> test
    train, test = df.iloc[:cut], df.iloc[cut:]
    Xtr, ytr, Xte, yte = train[RAW_COLUMNS], train["Class"], test[RAW_COLUMNS], test["Class"]
    print(f"train={len(train)} (fraud {int(ytr.sum())}) | test={len(test)} (fraud {int(yte.sum())})")

    # tuning set = all frauds + a fraction of genuine rows (keeps tuning fast)
    tune = pd.concat([train[train.Class == 1], train[train.Class == 0].sample(frac=a.tune_frac, random_state=SEED)])
    tune = tune.sample(frac=1, random_state=SEED)
    Xt, yt = tune[RAW_COLUMNS], tune["Class"]
    cv = StratifiedKFold(3, shuffle=True, random_state=SEED)

    tuning_rows, best_params, cv_score = [], {}, {}
    for name, (factory, grid, n_iter) in SPECS.items():
        pipe = make_pipeline(factory())
        base = cross_val_score(pipe, Xt, yt, cv=cv, scoring="average_precision", n_jobs=-1).mean()
        g = {f"clf__{k}": v for k, v in grid.items()}
        search = (RandomizedSearchCV(pipe, g, n_iter=n_iter, cv=cv, scoring="average_precision", n_jobs=-1,
                                     random_state=SEED, refit=False) if n_iter else
                  GridSearchCV(pipe, g, cv=cv, scoring="average_precision", n_jobs=-1, refit=False))
        search.fit(Xt, yt)
        bp = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        best_params[name], cv_score[name] = bp, search.best_score_
        tuning_rows.append(dict(Model=name, Default_CV_PR_AUC=round(base, 4), Tuned_CV_PR_AUC=round(search.best_score_, 4),
                                Improvement=round(search.best_score_ - base, 4), Best_Params=json.dumps(bp, default=str)))
        print(tuning_rows[-1])
    pd.DataFrame(tuning_rows).to_csv(f"{a.out}/tuning_results.csv", index=False)

    # final fit of each tuned model on the full training set, evaluated on the untouched test set
    final, probs, rows = {}, {}, []
    for name, (factory, _, _) in SPECS.items():
        pipe = make_pipeline(factory()).set_params(**{f"clf__{k}": v for k, v in best_params[name].items()})
        if name == "Random Forest":
            pipe.set_params(clf__n_jobs=-1)
        pipe.fit(Xtr, ytr)
        final[name], probs[name] = pipe, pipe.predict_proba(Xte)[:, 1]
        rows.append(dict(Model=name, **metrics(yte, probs[name])))
    res = pd.DataFrame(rows).round(4)
    res.to_csv(f"{a.out}/final_model_comparison.csv", index=False)
    print(res[["Model", "Precision", "Recall", "F1", "PR_AUC"]].to_string(index=False))

    # model selection uses cross-validation score only (test set never used to choose)
    best = max(cv_score, key=cv_score.get)
    print("Selected model:", best)

    # decision threshold chosen on a validation slice taken from the END of the training period
    vcut = int(len(train) * 0.85)
    sub, val = train.iloc[:vcut], train.iloc[vcut:]
    thr = 0.5
    if val.Class.sum() > 0:
        vp = clone(final[best]).fit(sub[RAW_COLUMNS], sub["Class"])
        thr = best_threshold(val["Class"], vp.predict_proba(val[RAW_COLUMNS])[:, 1])
    pipe = final[best]
    if "clf__n_jobs" in pipe.get_params():
        pipe.set_params(clf__n_jobs=1)
    sel = dict(Model=best, Threshold=round(thr, 4), **metrics(yte, probs[best], thr))
    pd.DataFrame([dict(Model=best, Threshold=0.5, **metrics(yte, probs[best], 0.5)), sel]).round(4).to_csv(
        f"{a.out}/final_selected_model_threshold.csv", index=False)

    plt.figure(figsize=(4, 3.5))
    cm = confusion_matrix(yte, (probs[best] >= thr).astype(int), labels=[0, 1])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Genuine", "Fraud"], yticklabels=["Genuine", "Fraud"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best} (thr={thr:.3f})", fontsize=9); plt.tight_layout()
    plt.savefig(f"{a.out}/final_confusion_matrix.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4.5))
    for n, p in probs.items():
        pr, rc, _ = precision_recall_curve(yte, p); plt.plot(rc, pr, label=n)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Tuned models: Precision-Recall"); plt.legend()
    plt.tight_layout(); plt.savefig(f"{a.out}/final_pr_curves.png", dpi=150); plt.close()

    meta = dict(model=best, threshold=thr, best_params=best_params[best], test_metrics={k: float(v) for k, v in sel.items()
                if k not in ("Model",)}, train_rows=len(train), test_rows=len(test),
                sklearn_version=sklearn.__version__, trained_on=str(datetime.date.today()),
                split="chronological 80/20", inputs=RAW_COLUMNS)
    joblib.dump(dict(pipeline=pipe, threshold=thr, meta=meta), f"{a.model_dir}/fraud_model.joblib", compress=3)
    json.dump(meta, open(f"{a.model_dir}/model_meta.json", "w"), indent=2, default=str)
    s = pd.concat([test[test.Class == 1].head(5), test[test.Class == 0].sample(5, random_state=SEED)])
    s.to_csv(f"{a.model_dir}/sample_transactions.csv", index=False)
    mb = os.path.getsize(f"{a.model_dir}/fraud_model.joblib") / 1e6
    print(f"Saved model ({mb:.1f} MB). scikit-learn version: {sklearn.__version__}")
    if mb > 24:
        print("WARNING: model > 24 MB, GitHub web upload will fail. Re-run with smaller RF grid.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model-dir", default="model")
    ap.add_argument("--tune-frac", type=float, default=0.1, help="fraction of genuine rows used while tuning")
    main(ap.parse_args())
