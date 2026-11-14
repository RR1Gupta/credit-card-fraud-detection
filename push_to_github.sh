#!/bin/bash
# 1) Create an EMPTY repo on github.com named credit-card-fraud-detection (no README)
# 2) Run:  bash push_to_github.sh YOUR_GITHUB_USERNAME
set -e
git init
git add .
git commit -m "Credit card fraud detection: EDA, SMOTE, LR/DT/RF, SHAP"
git branch -M main
git remote add origin https://github.com/$1/credit-card-fraud-detection.git
git push -u origin main
echo "Repo link: https://github.com/$1/credit-card-fraud-detection"
