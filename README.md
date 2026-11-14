# Credit Card Fraud Detection Using ML with Imbalanced Data Handling and Explainable AI

Binary classification on the Kaggle ULB Credit Card Fraud dataset (284,807 transactions, 492 frauds = 0.173%).

## What it does
| Phase | Done in `src/fraud_detection.py` |
|---|---|
| EDA | class distribution, Amount histograms, correlation heatmap, summary stats |
| Preprocessing | duplicate removal, chronological (or random stratified) 80/20 split, scaler fitted on train only |
| Feature engineering | `Log_Amount`, `Transaction_Hour`, `Amount_Category` |
| Scaling | Standard vs Robust vs MinMax (ablation on Logistic Regression) |
| Imbalance | Original vs SMOTE (SMOTE on training data only) |
| Models | Logistic Regression, Decision Tree, Random Forest |
| Evaluation | Precision, Recall, F1, PR-AUC, ROC-AUC, confusion matrix, PR curves |
| Extras | threshold tuning (chosen on validation slice of train), feature-engineering ablation |
| Explainable AI | RF feature importance + SHAP summary/bar + single fraud case |

## Setup
1. Download `creditcard.csv` from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and put it in `data/`.
2. `pip install -r requirements.txt`
3. `python src/fraud_detection.py --data data/creditcard.csv`

Options: `--split random`, `--keep-duplicates`, `--fast` (quick test on subsample).

**Google Colab:** upload the repo + csv, then
```
!pip install -q imbalanced-learn shap
!python src/fraud_detection.py --data creditcard.csv
```

## Outputs (`results/`)
`model_comparison.csv` (fills the TBD table), `ablation_scaling.csv`, `ablation_feature_engineering.csv`,
`threshold_tuning.csv`, `pr_curves.png`, confusion matrices, EDA plots, `shap_*.png`, `rf_feature_importance.png`.

## Limitations
Two days of 2013 European data; V1-V28 are anonymized PCA features (no real-world meaning claimed);
educational prototype, not a production system.

## References
- Salekshahrezaee et al., *The effect of feature extraction and data sampling on credit card fraud detection*, J. Big Data, 2023. doi:10.1186/s40537-023-00684-w
- *Comparative Analysis of ML Algorithms and Data Balancing Techniques for Credit Card Fraud Detection*, Procedia CS 245, 2024. doi:10.1016/j.procs.2024.10.294
- *A systematic review of literature on credit card cyber fraud detection using ML and DL*, PeerJ CS, 2023.
- Dal Pozzolo et al., Calibrating Probability with Undersampling for Unbalanced Classification, IEEE SSCI 2015 (dataset).
