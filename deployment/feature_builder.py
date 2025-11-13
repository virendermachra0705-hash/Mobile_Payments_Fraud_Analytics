# feature_builder.py (FINAL FIXED VERSION)
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import csv

HISTORY_PATH = "../data/state/transactions_history.csv"
API_LOG_PATH = "../reports/api_logs.csv"


def _safe_datetime(ts):
    try:
        return pd.to_datetime(ts)
    except:
        return pd.Timestamp.now()


def load_history():
    required_cols = [
        "transaction_id", "user_id", "merchant_id", "amount",
        "timestamp", "device_type", "transaction_type", "location",
        "is_fraud"
    ]

    try:
        df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        if not set(required_cols).issubset(df.columns):
            return pd.DataFrame(columns=required_cols)
        return df
    except:
        return pd.DataFrame(columns=required_cols)


def save_history_row(row_dict):
    os.makedirs("../data/state", exist_ok=True)

    required_cols = [
        "transaction_id", "user_id", "merchant_id", "amount",
        "timestamp", "device_type", "transaction_type", "location",
        "is_fraud"
    ]

    # Ensure missing columns are filled with None/0
    safe_row = {col: row_dict.get(col, None) for col in required_cols}

    file_exists = os.path.exists(HISTORY_PATH)

    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=required_cols)
        if not file_exists:
            writer.writeheader()
        writer.writerow(safe_row)


def append_api_log(log_dict):
    os.makedirs("../reports", exist_ok=True)
    file_exists = os.path.exists(API_LOG_PATH)

    with open(API_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_dict)


def build_features_from_raw(raw_txn, model_feature_list, training_df):

    hist = load_history()

    txn = raw_txn.copy()
    txn_ts = _safe_datetime(txn.get("timestamp"))
    txn["timestamp"] = txn_ts

    # basic fields
    for k in ["transaction_id", "user_id", "merchant_id", "device_type", "transaction_type", "location"]:
        txn.setdefault(k, None)

    try:
        txn["amount"] = float(txn["amount"])
    except:
        txn["amount"] = 0.0

    # time features
    hour = txn_ts.hour
    day_of_week = txn_ts.day_name()
    is_weekend = int(day_of_week in ["Saturday", "Sunday"])

    # user history
    user_hist = hist[hist["user_id"] == txn["user_id"]
                     ] if txn["user_id"] in hist["user_id"].values else pd.DataFrame()

    user_txn_count = len(user_hist)
    user_txn_sum = user_hist["amount"].sum() if user_txn_count > 0 else 0
    user_avg = user_hist["amount"].mean(
    ) if user_txn_count > 0 else training_df["user_avg_amount"].median()
    user_std = user_hist["amount"].std(
    ) if user_txn_count > 1 else training_df["user_std_amount"].median()

    amount_z = (txn["amount"] - user_avg) / (user_std if user_std > 0 else 1)

    if user_txn_count > 0:
        last_ts = user_hist["timestamp"].max()
        time_diff_sec = (txn_ts - last_ts).total_seconds()
    else:
        time_diff_sec = 0

    # velocity
    if user_txn_count > 0:
        one_h = txn_ts - pd.Timedelta(hours=1)
        day_h = txn_ts - pd.Timedelta(hours=24)
        txn1 = user_hist[user_hist["timestamp"] >= one_h]
        txn24 = user_hist[user_hist["timestamp"] >= day_h]
        txn_count_1h = len(txn1)
        txn_count_24h = len(txn24)
    else:
        txn_count_1h = 0
        txn_count_24h = 0

    txn_order = user_txn_count

    # rolling fraud
    if user_txn_count > 0:
        week = txn_ts - pd.Timedelta(days=7)
        recent = user_hist[user_hist["timestamp"] >= week]
        rolling_fraud = recent["is_fraud"].mean() if len(recent) > 0 else 0
        user_fraud = user_hist["is_fraud"].mean()
    else:
        rolling_fraud = 0
        user_fraud = 0

    # merchant stats
    merchant_hist = hist[hist["merchant_id"] == txn["merchant_id"]
                         ] if txn["merchant_id"] in hist["merchant_id"].values else pd.DataFrame()
    merchant_txn = len(merchant_hist)
    merchant_users = merchant_hist["user_id"].nunique(
    ) if merchant_txn > 0 else 0
    merchant_fraud = merchant_hist["is_fraud"].mean(
    ) if merchant_txn > 0 else training_df["merchant_fraud_rate"].median()

    # basic flags
    is_night = int(hour in [0, 1, 2, 3, 4, 5])
    is_new_user = int(user_txn_count == 0)
    is_high_amount = int(txn["amount"] > user_avg * 3)
    is_high_velocity = int(txn_count_1h >= 5)

    # 5-min window
    if user_txn_count > 0:
        five = txn_ts - pd.Timedelta(minutes=5)
        txns_5 = user_hist[user_hist["timestamp"] >= five]
        txns_in_last_5min = len(txns_5)
    else:
        txns_in_last_5min = 0

    weighted_amount = txn["amount"] * (1 + merchant_fraud)
    recent_weight = 1 / (1 + txn_order)

    # categorical
    one_hots = {
        f"transaction_type_{txn['transaction_type']}": 1,
        f"device_type_{txn['device_type']}": 1,
        f"day_of_week_{day_of_week}": 1
    }

    # final flat dict
    base = {
        "amount": txn["amount"],
        "hour": hour,
        "is_weekend": is_weekend,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "user_avg_amount": user_avg,
        "user_std_amount": user_std,
        "amount_zscore_user": amount_z,
        "time_diff_sec": time_diff_sec,
        "txn_order": txn_order,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "rolling_fraud_rate_user_7d": rolling_fraud,
        "user_fraud_rate": user_fraud,
        "merchant_txn_count": merchant_txn,
        "merchant_unique_users": merchant_users,
        "merchant_fraud_rate": merchant_fraud,
        "device_risk_score": 0,
        "location_risk_score": 0,
        "is_night": is_night,
        "is_new_user": is_new_user,
        "is_high_amount": is_high_amount,
        "is_high_velocity": is_high_velocity,
        "txns_in_last_5min": txns_in_last_5min,
        "weighted_amount": weighted_amount,
        "recent_txn_weight": recent_weight,
        "user_device_count": user_hist["device_type"].nunique() if user_txn_count > 0 else 0,
        "user_location_count": user_hist["location"].nunique() if user_txn_count > 0 else 0,
        "txn_velocity": txn_count_24h / 24 if txn_count_24h > 0 else 0,
        "peer_spend_ratio": txn["amount"] / (merchant_hist["amount"].median() if merchant_txn > 0 else training_df["amount"].median()),
        "merchant_user_overlap": merchant_users / (merchant_txn + 1)
    }

    # merge one-hot
    final_dict = {**base, **one_hots}

    # create aligned feature vector
    vector = {feat: final_dict.get(feat, 0) for feat in model_feature_list}

    feature_vector = pd.DataFrame([vector], columns=model_feature_list)

    # enriched record (for history)
    enriched = final_dict.copy()
    return feature_vector, enriched
