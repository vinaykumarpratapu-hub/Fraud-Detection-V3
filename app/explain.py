"""
LLM explainability layer for the fraud detection model.

Mirrors the "BERT does classification, Claude explains it" architecture from
the iPhone sentiment v2 project: the XGBoost model is the "brain" that makes
the fraud call, and Claude is the "frontend" that translates the model's
signals into a plain-English explanation a fraud analyst (or a non-technical
reviewer) can actually act on.

If no API key is available, falls back to a rule-based template so the app
still works end-to-end without a key configured.
"""
import os
import streamlit as st

SYSTEM_PROMPT = """You are a fraud analyst assistant. You are given a transaction, a machine learning \
model's fraud prediction, and the top features that drove that prediction (with their values and \
relative importance). Write a short, clear explanation (3-5 sentences) for a bank fraud analyst, in \
plain English, of why the model made this call. Reference the specific numbers. Do not restate the \
raw feature names verbatim (e.g. say "the account was emptied out" instead of "origEmptiedOut=1"). \
If the transaction was NOT flagged as fraud, briefly explain what about it looked normal. \
Do not invent information that isn't in the provided data."""


def get_api_key():
    # Streamlit Cloud: st.secrets. Local/dev: environment variable.
    try:
        if "ANTHROPIC_API_KEY" in st.secrets:
            return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


FEATURE_LABELS = {
    "amount": "transaction amount",
    "oldbalanceOrg": "sender's balance before the transaction",
    "newbalanceOrig": "sender's balance after the transaction",
    "oldbalanceDest": "recipient's balance before the transaction",
    "newbalanceDest": "recipient's balance after the transaction",
    "isCashOut": "transaction is a cash-out",
    "destIsMerchant": "recipient is a registered merchant",
    "errorBalanceOrig": "mismatch between expected and actual sender balance change",
    "errorBalanceDest": "mismatch between expected and actual recipient balance change",
    "hourOfDay": "hour of day the transaction occurred",
    "origEmptiedOut": "sender's account was fully emptied",
    "origBalanceZeroBefore": "sender's account already had a zero balance beforehand",
}


def _template_explanation(transaction, prediction, prob, top_features):
    verdict = "flagged as likely fraud" if prediction == 1 else "not flagged as fraud"
    lines = [
        f"This transaction was {verdict} with a model confidence of {prob:.1%}.",
    ]
    signal_bits = []
    for feat, val, importance in top_features[:3]:
        label = FEATURE_LABELS.get(feat, feat)
        signal_bits.append(f"{label} (value: {val:,.2f})")
    if signal_bits:
        lines.append("The strongest signals driving this call were: " + "; ".join(signal_bits) + ".")
    if prediction == 1:
        lines.append(
            "In this dataset, fraudulent transfers/cash-outs typically drain the sender's account "
            "in a single transaction, which is consistent with what's seen here."
        )
    else:
        lines.append("The balance movements and amount look consistent with normal account activity.")
    return " ".join(lines)


def explain_transaction(transaction: dict, prediction: int, prob: float, top_features: list, client_factory=None):
    """
    transaction: dict of feature_name -> value for this transaction
    prediction: 0 or 1
    prob: model's fraud probability
    top_features: list of (feature_name, value, importance) sorted by importance desc
    client_factory: optional injectable factory for testing (returns an anthropic.Anthropic client)
    """
    api_key = get_api_key()
    if not api_key:
        return _template_explanation(transaction, prediction, prob, top_features), "template"

    try:
        import anthropic
        client = client_factory() if client_factory else anthropic.Anthropic(api_key=api_key)

        feature_lines = "\n".join(
            f"- {FEATURE_LABELS.get(f, f)}: {v:,.2f} (importance rank signal: {imp:.3f})"
            for f, v, imp in top_features
        )
        user_prompt = f"""Transaction details:
- Amount: {transaction.get('amount'):,.2f}
- Sender balance before -> after: {transaction.get('oldbalanceOrg'):,.2f} -> {transaction.get('newbalanceOrig'):,.2f}
- Recipient balance before -> after: {transaction.get('oldbalanceDest'):,.2f} -> {transaction.get('newbalanceDest'):,.2f}
- Transaction type: {"CASH_OUT" if transaction.get('isCashOut') else "TRANSFER"}
- Recipient is merchant: {bool(transaction.get('destIsMerchant'))}

Model prediction: {"FRAUD" if prediction == 1 else "NOT FRAUD"}
Model confidence (probability of fraud): {prob:.1%}

Top contributing signals:
{feature_lines}

Explain this prediction to a fraud analyst."""

        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text, "claude"
    except Exception as e:
        # Fail gracefully to the template rather than crashing the app
        fallback = _template_explanation(transaction, prediction, prob, top_features)
        return fallback + f"\n\n_(LLM explanation unavailable: {e})_", "template_fallback"
