import streamlit as st
import pandas as pd
from utils import call_api

st.set_page_config(page_title="Mobile Fraud Detection", layout="centered")

st.title("📱 Real-Time Mobile Payment Fraud Detector")
st.write("Enter transaction details below and get instant fraud risk scoring.")

# --- Input Form ---
with st.form("fraud_form"):
    amount = st.number_input("Transaction Amount", min_value=1.0, step=10.0)
    hour = st.slider("Transaction Hour (0-23)", 0, 23, 12)

    day_of_week = st.selectbox("Day of Week", [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"
    ])

    device_type = st.selectbox("Device Type", ["Android", "iOS", "Web"])
    transaction_type = st.selectbox(
        "Transaction Type", ["purchase", "transfer", "top-up"])

    user_txn_count = st.number_input(
        "User Transaction Count (lifetime)", 0, 10000, 20)
    user_avg_amount = st.number_input("User Avg Amount", 0.0, 500000.0, 500.0)
    user_std_amount = st.number_input("User Std Amount", 0.0, 500000.0, 50.0)

    rolling_fraud_rate_user_7d = st.slider(
        "Rolling Fraud Rate (7 days)", 0.0, 1.0, 0.1)
    user_fraud_rate = st.slider("User Fraud Rate", 0.0, 1.0, 0.05)

    amount_zscore_user = st.number_input("Amount Z-Score", -5.0, 10.0, 1.2)
    merchant_fraud_rate = st.slider("Merchant Fraud Rate", 0.0, 1.0, 0.02)
    peer_spend_ratio = st.number_input("Peer Spend Ratio", 0.1, 10.0, 1.0)
    weighted_amount = st.number_input("Weighted Amount", 0.0, 500000.0, 1500.0)

    submitted = st.form_submit_button("🚀 Predict Fraud Risk")

# --- On Submit ---
if submitted:
    payload = {
        "amount": amount,
        "hour": hour,
        "day_of_week": day_of_week,
        "device_type": device_type,
        "transaction_type": transaction_type,
        "user_txn_count": user_txn_count,
        "user_avg_amount": user_avg_amount,
        "user_std_amount": user_std_amount,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "user_fraud_rate": user_fraud_rate,
        "amount_zscore_user": amount_zscore_user,
        "merchant_fraud_rate": merchant_fraud_rate,
        "peer_spend_ratio": peer_spend_ratio,
        "weighted_amount": weighted_amount
    }

    st.write("📤 Sending request to fraud API...")
    output = call_api(payload)

    if "error" in output:
        st.error(f"❌ Error: {output['error']}")
    else:
        prob = output["fraud_probability"]
        flag = output["fraud_flag"]

        st.success(f"🔥 Fraud Probability: **{prob:.4f}**")
        st.write(f"Fraud Flag: {'🛑 YES (High Risk)' if flag else '🟢 No'}")
