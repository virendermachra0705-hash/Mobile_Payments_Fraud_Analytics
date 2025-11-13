# feature_builder.py
import pandas as pd
import numpy as np
from datetime import datetime

HISTORY_PATH = "../data/state/transactions_history.csv"
API_LOG_PATH = "../reports/api_logs.csv"


# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def _safe_datetime(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.to_datetime(str(ts), errors="coerce")


def load_history():
    """Load transaction history. If file missing or corrupted, return empty."""
    try:
        hist = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        return hist
    except Exception:
        cols = [
            "transaction_id", "user_id", "merchant_id", "amount", "timestamp",
            "device_type", "transaction_type", "location", "is_fraud"
        ]
        return pd.DataFrame(columns=cols)


def save_history_row(enriched_txn):
    """Append a transaction row directly to history CSV."""
    import csv, os
    os.makedirs("../data/state", exist_ok=True)

    file_exists = os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=enriched_txn.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(enriched_txn)


def append_api_log(log_entry):
    """Append prediction outputs to the API log CSV."""
    import csv, os
    os.makedirs("../reports", exist_ok=True)

    file_exists = os.path.exists(API_LOG_PATH)
    with open(API_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


# -----------------------------------------------------------------------------
# MAIN FEATURE BUILDER
# -----------------------------------------------------------------------------

def build_features_from_raw(raw_txn: dict, model_feature_list: list):
    """
    Convert a raw incoming transaction into the model's required engineered features.
    History-driven features are included if available; otherwise use safe fallbacks.
    """

    # Load history safely
    hist = load_history()

    txn = raw_txn.copy()
    txn_ts = _safe_datetime(txn.get("timestamp"))
    if pd.isna(txn_ts):
        txn_ts = pd.Timestamp.utcnow()

    txn["timestamp"] = txn_ts

    # Extract raw fields with safe defaults
    transaction_id = txn.get("transaction_id")
    user_id = txn.get("user_id")
    merchant_id = txn.get("merchant_id")
    amount = float(txn.get("amount", 0))
    device_type = txn.get("device_type", "Unknown")
    transaction_type = txn.get("transaction_type", "purchase")
    location = txn.get("location", "Unknown")

    # -------------------------------------------------------------------------
    # USER-LEVEL HISTORY (SAFE)
    # -------------------------------------------------------------------------
    if "user_id" in hist.columns and user_id in hist["user_id"].values:
        user_hist = hist[hist["user_id"] == user_id].copy()
        user_hist = user_hist.sort_values("timestamp")
    else:
        user_hist = pd.DataFrame()

    user_txn_count = len(user_hist)

    if user_txn_count > 0:
        user_avg_amount = user_hist["amount"].mean()
        user_std_amount = user_hist["amount"].std() or 1.0
        user_txn_sum = user_hist["amount"].sum()
        user_fraud_rate = (
            user_hist["is_fraud"].mean()
            if "is_fraud" in user_hist.columns else 0.0
        )
    else:
        user_avg_amount = 0.0
        user_std_amount = 1.0
        user_txn_sum = 0.0
        user_fraud_rate = 0.0

    amount_zscore_user = (amount - user_avg_amount) / (user_std_amount or 1.0)

    # time difference from last user txn
    if user_txn_count > 0 and "timestamp" in user_hist.columns:
        last_ts = user_hist["timestamp"].max()
        time_diff_sec = (txn_ts - pd.Timestamp(last_ts)).total_seconds()
    else:
        time_diff_sec = 0.0

    # counts in last windows
    if user_txn_count > 0 and "timestamp" in user_hist.columns:
        one_hour_ago = txn_ts - pd.Timedelta(hours=1)
        day_ago = txn_ts - pd.Timedelta(hours=24)
        txn_count_1h = user_hist[user_hist["timestamp"] >= one_hour_ago].shape[0]
        txn_count_24h = user_hist[user_hist["timestamp"] >= day_ago].shape[0]
    else:
        txn_count_1h = 0
        txn_count_24h = 0

    # rolling 7d fraud
    if user_txn_count > 0 and "timestamp" in user_hist.columns and "is_fraud" in user_hist.columns:
        seven_days_ago = txn_ts - pd.Timedelta(days=7)
        recent = user_hist[user_hist["timestamp"] >= seven_days_ago]
        rolling_fraud_rate_user_7d = recent["is_fraud"].mean() if len(recent) > 0 else 0.0
    else:
        rolling_fraud_rate_user_7d = 0.0

    # -------------------------------------------------------------------------
    # MERCHANT-LEVEL HISTORY (SAFE)
    # -------------------------------------------------------------------------
    if "merchant_id" in hist.columns and merchant_id in hist["merchant_id"].values:
        merchant_hist = hist[hist["merchant_id"] == merchant_id]
        merchant_txn_count = len(merchant_hist)
        merchant_unique_users = merchant_hist["user_id"].nunique() if "user_id" in merchant_hist.columns else 0
        merchant_fraud_rate = merchant_hist["is_fraud"].mean() if "is_fraud" in merchant_hist.columns else 0.0
    else:
        merchant_txn_count = 0
        merchant_unique_users = 0
        merchant_fraud_rate = 0.0

    # -------------------------------------------------------------------------
    # DEVICE / LOCATION HISTORY (SAFE)
    # -------------------------------------------------------------------------
    if "device_type" in hist.columns:
        device_risk_score = (
            hist[hist["device_type"] == device_type]["is_fraud"].mean()
            if device_type in hist["device_type"].values else 0.0
        )
        device_txn_count = hist[hist["device_type"] == device_type].shape[0]
    else:
        device_risk_score = 0.0
        device_txn_count = 0

    if "location" in hist.columns:
        location_risk_score = (
            hist[hist["location"] == location]["is_fraud"].mean()
            if location in hist["location"].values else 0.0
        )
        location_txn_count = hist[hist["location"] == location].shape[0]
    else:
        location_risk_score = 0.0
        location_txn_count = 0

    # SAFE device-location interaction
    if "device_type" in hist.columns and "location" in hist.columns:
        device_location_interaction = hist[
            (hist["device_type"] == device_type) &
            (hist["location"] == location)
        ].shape[0]
    else:
        device_location_interaction = 0

    # -------------------------------------------------------------------------
    # RULE-BASED SIGNALS
    # -------------------------------------------------------------------------
    hour = txn_ts.hour
    day_of_week = txn_ts.day_name()

    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0
    is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    is_new_user = 1 if user_txn_count == 0 else 0
    is_high_amount = 1 if amount > (user_avg_amount * 3 or amount) else 0
    is_high_velocity = 1 if txn_count_1h >= 5 else 0

    weighted_amount = amount * (1 + merchant_fraud_rate)
    recent_txn_weight = 1.0 / (1 + user_txn_count)

    # safe peer spend
    peer_spend_ratio = amount / (user_avg_amount if user_avg_amount > 0 else (amount or 1.0))

    merchant_user_overlap = merchant_unique_users / (merchant_txn_count + 1)

    user_device_count = (
        user_hist["device_type"].nunique()
        if "device_type" in user_hist.columns else 0
    )

    user_location_count = (
        user_hist["location"].nunique()
        if "location" in user_hist.columns else 0
    )

    # -------------------------------------------------------------------------
    # FINAL FEATURE DICT
    # -------------------------------------------------------------------------
    features = {
        "amount": amount,
        "hour": hour,
        "is_weekend": is_weekend,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "user_avg_amount": user_avg_amount,
        "user_std_amount": user_std_amount,
        "amount_zscore_user": amount_zscore_user,
        "time_diff_sec": time_diff_sec,
        "txn_velocity": txn_count_24h / 24 if txn_count_24h > 0 else 0.0,
        "merchant_txn_count": merchant_txn_count,
        "merchant_unique_users": merchant_unique_users,
        "merchant_fraud_rate": merchant_fraud_rate,
        "device_risk_score": device_risk_score,
        "location_risk_score": location_risk_score,
        "device_location_interaction": device_location_interaction,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "is_high_amount": is_high_amount,
        "is_new_user": is_new_user,
        "is_high_velocity": is_high_velocity,
        "is_risky_device": 1 if device_risk_score > 0.05 else 0,
        "is_risky_location": 1 if location_risk_score > 0.05 else 0,
        "weighted_amount": weighted_amount,
        "user_risk_score": user_fraud_rate,
        "days_since_first_txn": user_txn_count,  # placeholder
        "txn_amount_ratio_user_avg": (amount / user_avg_amount) if user_avg_amount else 1.0,
        "is_night": is_night,
        "user_fraud_rate": user_fraud_rate,
        "merchant_user_overlap": merchant_user_overlap,
        "device_txn_count": device_txn_count,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "txns_in_last_5min": txn_count_1h,  # approx
        "avg_time_gap_prev3": time_diff_sec,  # placeholder
        "user_device_count": user_device_count,
        "user_location_count": user_location_count,
        "device_location_ratio": device_location_interaction,
        "merchant_past_fraud_rate": merchant_fraud_rate,
        "txn_order": user_txn_count,
        "recent_txn_weight": recent_txn_weight,
        "peer_spend_ratio": peer_spend_ratio,
    }

    # -------------------------------------------------------------------------
    # ONE HOT ENCODING
    # -------------------------------------------------------------------------
    onehots = {
        f"transaction_type_{transaction_type}": 1,
        f"device_type_{device_type}": 1,
        f"day_of_week_{day_of_week}": 1,
    }

    features.update(onehots)

    # -------------------------------------------------------------------------
    # ALIGN TO MODEL FEATURE LIST
    # -------------------------------------------------------------------------
    final = {}
    for feat in model_feature_list:
        final[feat] = features.get(feat, 0)

    # -------------------------------------------------------------------------
    # ENRICHED TXN FOR HISTORY
    # -------------------------------------------------------------------------
    enriched = {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "timestamp": txn_ts.isoformat(),
        "device_type": device_type,
        "transaction_type": transaction_type,
        "location": location,
        "user_txn_count": user_txn_count,
        "user_avg_amount": user_avg_amount,
        "amount_zscore_user": amount_zscore_user,
        "merchant_fraud_rate": merchant_fraud_rate,
        "txn_count_24h": txn_count_24h,
        "txn_count_1h": txn_count_1h,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "is_fraud": 0
    }

    return pd.DataFrame([final]), enriched
