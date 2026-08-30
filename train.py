"""
PaySim Fraud Detection — training pipeline
Project 3 in Vinay's AI portfolio series (iPhone sentiment v1/v2 -> fraud detection v3)

Data: PaySim synthetic mobile money transactions (systematic 1-in-10 sample of the
full Kaggle dataset, ~636K rows, preserves the original fraud rate).

Approach:
  - Domain-informed scoping: fraud in PaySim only ever occurs on TRANSFER and
    CASH_OUT transactions (verified in EDA), so we restrict modeling to those types.
    This is a real fraud-analytics technique (narrow to the risky transaction
    types before modeling) worth calling out in interviews.
  - Baseline: Logistic Regression with class_weight='balanced'
  - Main model: XGBoost with scale_pos_weight tuned for the imbalance
  - Evaluated on precision/recall/F1/PR-AUC (accuracy is meaningless at 0.1% fraud rate)
"""
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, precision_recall_curve, average_precision_score,
    roc_auc_score, confusion_matrix, f1_score, precision_score, recall_score
)
import xgboost as xgb
import joblib

RANDOM_STATE = 42

print("Loading data...")
df = pd.read_csv("data/paysim_sample.csv")
print(f"Raw shape: {df.shape}, fraud rate: {100*df['isFraud'].mean():.4f}%")

# --- Domain-informed scoping ---
df = df[df["type"].isin(["TRANSFER", "CASH_OUT"])].reset_index(drop=True)
print(f"After scoping to TRANSFER/CASH_OUT: {df.shape}, fraud rate: {100*df['isFraud'].mean():.4f}%")

# --- Feature engineering ---
df["isCashOut"] = (df["type"] == "CASH_OUT").astype(int)
df["destIsMerchant"] = df["nameDest"].str.startswith("M").astype(int)
df["errorBalanceOrig"] = df["newbalanceOrig"] + df["amount"] - df["oldbalanceOrg"]
df["errorBalanceDest"] = df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"]
df["hourOfDay"] = df["step"] % 24
df["origEmptiedOut"] = ((df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)).astype(int)
df["origBalanceZeroBefore"] = (df["oldbalanceOrg"] == 0).astype(int)

FEATURES = [
    "amount", "oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest",
    "isCashOut", "destIsMerchant", "errorBalanceOrig", "errorBalanceDest",
    "hourOfDay", "origEmptiedOut", "origBalanceZeroBefore",
]
TARGET = "isFraud"

X = df[FEATURES].copy()
y = df[TARGET].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print(f"Train: {X_train.shape}, fraud={y_train.sum()} | Test: {X_test.shape}, fraud={y_test.sum()}")

# --- Baseline: Logistic Regression ---
print("\n=== Baseline: Logistic Regression (class_weight=balanced) ===")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
lr.fit(X_train_s, y_train)
lr_probs = lr.predict_proba(X_test_s)[:, 1]
lr_preds = (lr_probs >= 0.5).astype(int)

lr_metrics = {
    "precision": precision_score(y_test, lr_preds),
    "recall": recall_score(y_test, lr_preds),
    "f1": f1_score(y_test, lr_preds),
    "roc_auc": roc_auc_score(y_test, lr_probs),
    "pr_auc": average_precision_score(y_test, lr_probs),
}
print(json.dumps(lr_metrics, indent=2))
print(confusion_matrix(y_test, lr_preds))

# --- Main model: XGBoost ---
print("\n=== Main model: XGBoost ===")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"scale_pos_weight = {scale_pos_weight:.1f}")

xgb_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.08,
    scale_pos_weight=scale_pos_weight,
    eval_metric="aucpr",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
xgb_preds = (xgb_probs >= 0.5).astype(int)

xgb_metrics = {
    "precision": precision_score(y_test, xgb_preds),
    "recall": recall_score(y_test, xgb_preds),
    "f1": f1_score(y_test, xgb_preds),
    "roc_auc": roc_auc_score(y_test, xgb_probs),
    "pr_auc": average_precision_score(y_test, xgb_probs),
}
print(json.dumps(xgb_metrics, indent=2))
print(confusion_matrix(y_test, xgb_preds))

# --- Feature importance ---
importances = pd.Series(xgb_model.feature_importances_, index=FEATURES).sort_values(ascending=False)
print("\nFeature importances:")
print(importances)

# --- Save everything ---
joblib.dump(xgb_model, "models/xgb_fraud_model.joblib")
joblib.dump(scaler, "models/scaler.joblib")
joblib.dump(lr, "models/logreg_baseline.joblib")

with open("models/feature_list.json", "w") as f:
    json.dump(FEATURES, f)

with open("models/metrics.json", "w") as f:
    json.dump({"logistic_regression_baseline": lr_metrics, "xgboost": xgb_metrics}, f, indent=2)

importances.to_json("models/feature_importances.json")

# save a small test sample for the app / explainability demo (with readable info)
demo_cols = FEATURES + ["isFraud"]
X_test_demo = df.loc[X_test.index, demo_cols + ["type", "nameOrig", "nameDest"]]
X_test_demo["xgb_prob"] = xgb_probs
X_test_demo["xgb_pred"] = xgb_preds
X_test_demo.to_csv("data/test_sample_for_app.csv", index=False)

print("\nSaved model, scaler, baseline, metrics, feature importances, and demo test sample.")
