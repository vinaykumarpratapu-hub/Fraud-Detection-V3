"""
Streamlit app: PaySim Fraud Detection with LLM Explainability
Project 3 in Vinay's AI portfolio (after iPhone Sentiment v1/v2).

Architecture:
  XGBoost classifier (the "brain") -> SHAP local feature attributions
  -> Claude (the "frontend") turns those attributions into a plain-English
     explanation for a fraud analyst.
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st

sys.path.append(os.path.dirname(__file__))
from explain import explain_transaction, FEATURE_LABELS

st.set_page_config(page_title="Fraud Detection + Explainability", page_icon="🕵️", layout="wide")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


@st.cache_resource
def load_artifacts():
    model = joblib.load(os.path.join(MODEL_DIR, "xgb_fraud_model.joblib"))
    with open(os.path.join(MODEL_DIR, "feature_list.json")) as f:
        features = json.load(f)
    with open(os.path.join(MODEL_DIR, "metrics.json")) as f:
        metrics = json.load(f)
    explainer = shap.TreeExplainer(model)
    return model, features, metrics, explainer


@st.cache_data
def load_demo_data():
    return pd.read_csv(os.path.join(DATA_DIR, "test_sample_for_app.csv"))


model, FEATURES, metrics, explainer = load_artifacts()
demo_df = load_demo_data()

st.title("🕵️ Fraud Detection + Explainability")
st.caption(
    "PaySim synthetic mobile-money transactions · XGBoost classifier · "
    "SHAP local attributions · Claude plain-English explanations"
)

with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
This is project 3 in a portfolio series building toward AI Product Manager
roles in banking/fintech.

**Pipeline:** PaySim transactions → domain-scoped to TRANSFER/CASH_OUT →
engineered balance-consistency features → XGBoost classifier → SHAP for
per-transaction feature attribution → Claude turns the attribution into a
plain-English explanation.

**Why not just accuracy?** Fraud is ~0.28% of scoped transactions, so this
app reports precision, recall, F1, and PR-AUC instead.
        """
    )
    st.subheader("Model performance (held-out test set)")
    xgb_m = metrics["xgboost"]
    lr_m = metrics["logistic_regression_baseline"]
    col1, col2 = st.columns(2)
    col1.metric("XGBoost F1", f"{xgb_m['f1']:.3f}")
    col2.metric("XGBoost PR-AUC", f"{xgb_m['pr_auc']:.3f}")
    st.caption(f"Baseline Logistic Regression F1: {lr_m['f1']:.3f} (precision {lr_m['precision']:.1%}, "
               f"recall {lr_m['recall']:.1%}) — high recall but very noisy without the engineered "
               f"balance-consistency features and a tree model.")

st.subheader("1. Pick a transaction")

mode = st.radio(
    "Source",
    ["Sample a fraud case", "Sample a legitimate case", "Enter transaction manually"],
    horizontal=True,
)

if mode == "Sample a fraud case":
    pool = demo_df[demo_df["isFraud"] == 1]
    if st.button("🎲 Sample another fraud case") or "row" not in st.session_state or st.session_state.get("mode") != mode:
        st.session_state.row = pool.sample(1).iloc[0]
        st.session_state.mode = mode
    row = st.session_state.row
elif mode == "Sample a legitimate case":
    pool = demo_df[demo_df["isFraud"] == 0]
    if st.button("🎲 Sample another legitimate case") or "row" not in st.session_state or st.session_state.get("mode") != mode:
        st.session_state.row = pool.sample(1).iloc[0]
        st.session_state.mode = mode
    row = st.session_state.row
else:
    st.session_state.mode = mode
    c1, c2, c3 = st.columns(3)
    with c1:
        amount = st.number_input("Amount", min_value=0.0, value=250000.0, step=1000.0)
        old_orig = st.number_input("Sender balance before", min_value=0.0, value=250000.0, step=1000.0)
        new_orig = st.number_input("Sender balance after", min_value=0.0, value=0.0, step=1000.0)
    with c2:
        old_dest = st.number_input("Recipient balance before", min_value=0.0, value=0.0, step=1000.0)
        new_dest = st.number_input("Recipient balance after", min_value=0.0, value=250000.0, step=1000.0)
        hour = st.slider("Hour of day", 0, 23, 3)
    with c3:
        is_cashout = st.selectbox("Transaction type", ["TRANSFER", "CASH_OUT"]) == "CASH_OUT"
        dest_merchant = st.checkbox("Recipient is a merchant", value=False)

    row = pd.Series({
        "amount": amount, "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
        "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
        "isCashOut": int(is_cashout), "destIsMerchant": int(dest_merchant),
        "errorBalanceOrig": new_orig + amount - old_orig,
        "errorBalanceDest": old_dest + amount - new_dest,
        "hourOfDay": hour,
        "origEmptiedOut": int(new_orig == 0 and old_orig > 0),
        "origBalanceZeroBefore": int(old_orig == 0),
        "isFraud": None,
    })

st.subheader("2. Model prediction")

X_row = row[FEATURES].to_frame().T.astype(float)
prob = model.predict_proba(X_row)[0, 1]
pred = int(prob >= 0.5)

c1, c2, c3 = st.columns(3)
c1.metric("Prediction", "🚨 FRAUD" if pred else "✅ Legitimate")
c2.metric("Fraud probability", f"{prob:.1%}")
if row.get("isFraud") is not None and not pd.isna(row.get("isFraud")):
    actual = "FRAUD" if row["isFraud"] == 1 else "Legitimate"
    c3.metric("Ground truth (sample data)", actual)

with st.expander("Raw transaction features"):
    st.dataframe(X_row.T.rename(columns={X_row.index[0]: "value"}))

st.subheader("3. Why did the model decide this?")

shap_values = explainer.shap_values(X_row)
sv = shap_values[0] if isinstance(shap_values, list) else shap_values[0]
contrib = sorted(zip(FEATURES, X_row.iloc[0].values, np.abs(sv)), key=lambda t: -t[2])

shap_df = pd.DataFrame(
    {"feature": [FEATURE_LABELS.get(f, f) for f, _, _ in contrib],
     "value": [v for _, v, _ in contrib],
     "impact on fraud score": [s for f, v, s in zip(FEATURES, X_row.iloc[0].values, sv)]}
).set_index("feature")
st.bar_chart(shap_df["impact on fraud score"])

if st.button("💬 Explain this prediction in plain English", type="primary"):
    with st.spinner("Asking Claude..."):
        explanation, source = explain_transaction(
            transaction=row.to_dict(),
            prediction=pred,
            prob=prob,
            top_features=[(f, v, s) for f, v, s in contrib[:5]],
        )
    st.info(explanation)
    if source != "claude":
        st.caption(
            "⚠️ No ANTHROPIC_API_KEY found, so this is a rule-based fallback explanation, not an "
            "LLM-generated one. Add your key to Streamlit secrets to enable Claude explanations."
        )
