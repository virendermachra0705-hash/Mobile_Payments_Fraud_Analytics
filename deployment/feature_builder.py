# feature_builder.py
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

HISTORY_PATH = "../data/state/transactions_history.csv"
API_LOG_PATH = "../reports/api_logs.csv"

# Helper: safe parse timestamp -> pandas Timestamp


def _safe_datetime(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.to_datetime(str(ts), errors="coerce")

# Load persistent history used for rolling/stateful features


def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            hist = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
            # Ensure minimal expected columns exist (avoid KeyError later)
            expected = ["transaction_id", "user_id", "merchant_id", "amount", "timestamp",
                        "device_type", "transaction_type", "location", "is_fraud",
                        "user_txn_count", "user_avg_amount", "user_std_amount",
                        "merchant_fraud_rate", "txn_count_24h", "txn_count_1h", "rolling_fraud_rate_user_7d"]
            for c in expected:
                if c not in hist.columns:
                    hist[c] = np.nan
            return hist
        except Exception:
            return pd.DataFrame(columns=["transaction_id", "user_id", "merchant_id", "amount", "timestamp",
                                         "device_type", "transaction_type", "location", "is_fraud"])
    else:
        cols = ["transaction_id", "user_id", "merchant_id", "amount", "timestamp",
                "device_type", "transaction_type", "location", "is_fraud"]
        return pd.DataFrame(columns=cols)


def save_history_row(enriched_txn):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    import csv
    file_exists = os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(enriched_txn.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(enriched_txn)


def append_api_log(log_entry):
    os.makedirs(os.path.dirname(API_LOG_PATH), exist_ok=True)
    import csv
    file_exists = os.path.exists(API_LOG_PATH)
    with open(API_LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_entry.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

# Core: build feature vector for a single raw txn


def build_features_from_raw(raw_txn: dict, model_feature_list: list, training_df: pd.DataFrame = None):
    """
    raw_txn: dict with transaction fields
    model_feature_list: ordered list of feature column names expected by model
    training_df: DataFrame used as fallback medians/defaults (can be None)
    Returns:
       feature_vector (pd.DataFrame, single row, columns == model_feature_list)
       enriched_for_history (dict)   # for saving into history CSV
    """

    # load persistent history
    hist = load_history()

    # Normalize raw txn
    txn = raw_txn.copy()
    txn_ts = _safe_datetime(txn.get("timestamp", pd.Timestamp.utcnow()))
    if pd.isna(txn_ts):
        txn_ts = pd.Timestamp.utcnow()
    txn["timestamp"] = txn_ts

    # Ensure keys
    for k in ["transaction_id", "user_id", "merchant_id", "amount", "device_type", "transaction_type", "location"]:
        if k not in txn:
            txn[k] = None

    # Force numeric amount
    try:
        txn["amount"] = float(txn["amount"])
    except Exception:
        txn["amount"] = 0.0

    # Basic temporals
    hour = int(txn["timestamp"].hour)
    day_of_week = txn["timestamp"].day_name()
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

    user_id = txn.get("user_id")
    merchant_id = txn.get("merchant_id")
    device_type = txn.get("device_type", "Unknown")
    transaction_type = txn.get("transaction_type", "purchase")
    location = txn.get("location", "Unknown")

    # Prepare training medians/fallbacks
    def median_or_default(col, default=0.0):
        try:
            if training_df is not None and col in training_df.columns:
                return float(training_df[col].median())
        except Exception:
            pass
        return default

    median_user_avg = median_or_default("user_avg_amount", 0.0)
    median_user_std = median_or_default("user_std_amount", 1.0)
    median_merchant_fraud = median_or_default("merchant_fraud_rate", 0.0)
    median_amount = median_or_default("amount", 0.0)

    # User-level history
    if (user_id is not None) and ("user_id" in hist.columns) and (user_id in hist["user_id"].values):
        user_hist = hist[hist["user_id"] == user_id].sort_values("timestamp")
    else:
        user_hist = pd.DataFrame()

    user_txn_count = int(user_hist.shape[0]) if not user_hist.empty else 0
    if user_txn_count > 0:
        user_txn_sum = float(user_hist["amount"].sum())
        user_avg_amount = float(user_hist["amount"].mean())
        user_std_amount = float(user_hist["amount"].std()) if pd.notna(
            user_hist["amount"].std()) and user_hist["amount"].std() != 0 else median_user_std
    else:
        user_txn_sum = 0.0
        user_avg_amount = median_user_avg
        user_std_amount = median_user_std

    # amount zscore relative to user's distribution (fallback to global medians)
    if user_txn_count >= 2 and user_std_amount > 0:
        amount_zscore_user = (
            txn["amount"] - user_avg_amount) / user_std_amount
    else:
        global_std = median_or_default("user_std_amount", 1.0)
        global_avg = median_or_default("user_avg_amount", median_user_avg)
        amount_zscore_user = (txn["amount"] - global_avg) / \
            (global_std if global_std != 0 else 1.0)

    # Time diff since last txn for this user (seconds)
    if not user_hist.empty:
        # ensure timestamp type
        last_ts = user_hist["timestamp"].max()
        try:
            last_ts = pd.to_datetime(last_ts)
            time_diff_sec = (txn_ts - last_ts).total_seconds()
            if pd.isna(time_diff_sec):
                time_diff_sec = 0.0
        except Exception:
            time_diff_sec = 0.0
    else:
        time_diff_sec = np.nan

    # Rolling counts for velocity
    if not user_hist.empty:
        one_h_ago = txn_ts - pd.Timedelta(hours=1)
        day_ago = txn_ts - pd.Timedelta(hours=24)
        txn_count_1h = int(
            user_hist[user_hist["timestamp"] >= one_h_ago].shape[0])
        txn_count_24h = int(
            user_hist[user_hist["timestamp"] >= day_ago].shape[0])
    else:
        txn_count_1h = 0
        txn_count_24h = 0

    txn_order = user_txn_count
    rolling_fraud_rate_user_7d = 0.0
    if not user_hist.empty and "is_fraud" in user_hist.columns:
        seven_days_ago = txn_ts - pd.Timedelta(days=7)
        recent = user_hist[user_hist["timestamp"] >= seven_days_ago]
        if recent.shape[0] > 0:
            rolling_fraud_rate_user_7d = float(recent["is_fraud"].mean())
    user_fraud_rate = float(user_hist["is_fraud"].mean()) if (
        not user_hist.empty and "is_fraud" in user_hist.columns) else 0.0

    # merchant-based stats
    if (merchant_id is not None) and ("merchant_id" in hist.columns) and (merchant_id in hist["merchant_id"].values):
        merchant_hist = hist[hist["merchant_id"] == merchant_id]
        merchant_txn_count = int(merchant_hist.shape[0])
        merchant_unique_users = int(merchant_hist["user_id"].nunique(
        )) if "user_id" in merchant_hist.columns else 0
        merchant_fraud_rate = float(merchant_hist["is_fraud"].mean(
        )) if "is_fraud" in merchant_hist.columns else median_merchant_fraud
        merchant_past_fraud_rate = merchant_fraud_rate
    else:
        merchant_txn_count = 0
        merchant_unique_users = 0
        merchant_fraud_rate = median_merchant_fraud
        merchant_past_fraud_rate = median_merchant_fraud

    # merchant_user_overlap (unique users / total txns)
    merchant_user_overlap = merchant_unique_users / (merchant_txn_count + 1)

    # device & location heuristics
    if ("device_type" in hist.columns) and (device_type in hist["device_type"].values):
        device_risk_score = float(hist[hist["device_type"] == device_type]["is_fraud"].mean(
        )) if "is_fraud" in hist.columns else 0.0
        device_txn_count = int(
            hist[hist["device_type"] == device_type].shape[0])
    else:
        device_risk_score = 0.0
        device_txn_count = 0

    if ("location" in hist.columns) and (location in hist["location"].values):
        location_risk_score = float(hist[hist["location"] == location]["is_fraud"].mean(
        )) if "is_fraud" in hist.columns else 0.0
        location_txn_count = int(hist[hist["location"] == location].shape[0])
    else:
        location_risk_score = 0.0
        location_txn_count = 0

    device_location_interaction = int(hist[(hist.get("device_type") == device_type) & (
        hist.get("location") == location)].shape[0]) if not hist.empty else 0

    # txns in last 5 min
    txns_in_last_5min = 0
    if not user_hist.empty:
        window_5m = txn_ts - pd.Timedelta(minutes=5)
        txns_in_last_5min = int(
            user_hist[user_hist["timestamp"] >= window_5m].shape[0])
    else:
        txns_in_last_5min = 0

    # velocity and related
    time_diff = time_diff_sec if not pd.isna(time_diff_sec) else 99999.0
    txn_velocity = (txn_count_24h / 24.0) if txn_count_24h > 0 else 0.0

    # boolean/rule flags
    is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    is_new_user = 1 if user_txn_count == 0 else 0
    is_high_amount = 1 if txn["amount"] > (
        user_avg_amount * 3 if user_avg_amount > 0 else median_amount) else 0
    is_high_velocity = 1 if txn_count_1h >= 5 else 0
    is_risky_device = 1 if device_risk_score > 0.0 else 0
    is_risky_location = 1 if location_risk_score > 0.0 else 0

    # weighted amount
    weighted_amount = txn["amount"] * (1 + merchant_fraud_rate)

    # recent_txn_weight (decay)
    recent_txn_weight = 1.0 / (1 + txn_order)

    # user_device_count & user_location_count
    user_device_count = int(user_hist["device_type"].nunique()) if (
        not user_hist.empty and "device_type" in user_hist.columns) else 0
    user_location_count = int(user_hist["location"].nunique()) if (
        not user_hist.empty and "location" in user_hist.columns) else 0

    device_location_ratio = user_device_count / (user_location_count + 1)

    # avg_time_gap_prev3
    if not user_hist.empty and "timestamp" in user_hist.columns:
        # create time diffs series
        diffs = user_hist.sort_values(
            "timestamp")["timestamp"].diff().dt.total_seconds().fillna(np.nan)
        avg_time_gap_prev3 = float(diffs.rolling(3).mean(
        ).iloc[-1]) if len(diffs) >= 1 else float(median_or_default("time_diff_sec", 0.0))
        if np.isnan(avg_time_gap_prev3) or avg_time_gap_prev3 is None:
            avg_time_gap_prev3 = float(median_or_default("time_diff_sec", 0.0))
    else:
        avg_time_gap_prev3 = float(median_or_default("time_diff_sec", 0.0))

    # peer_spend_ratio: amount / median merchant amount
    if (merchant_id is not None) and ("merchant_id" in hist.columns) and (merchant_id in hist["merchant_id"].values):
        peers = hist[hist["merchant_id"] == merchant_id]
        peer_median = peers["amount"].median(
        ) if not peers.empty else median_amount
    else:
        peer_median = median_amount
    peer_spend_ratio = txn["amount"] / \
        (peer_median if peer_median != 0 else 1.0)

    # user_risk_score (simple average of device+location risk)
    user_risk_score = (device_risk_score + location_risk_score) / 2.0

    # days_since_first_txn
    first_txn_time = None
    if not user_hist.empty:
        first_txn_time = user_hist["timestamp"].min()
        try:
            days_since_first_txn = int(
                (txn_ts - pd.to_datetime(first_txn_time)).days)
        except Exception:
            days_since_first_txn = 0
    else:
        days_since_first_txn = 0

    # txn_amount_ratio_user_avg
    txn_amount_ratio_user_avg = txn["amount"] / (user_avg_amount + 1)

    # Build base dict (with exact names requested)
    base = {
        "timestamp": txn_ts.isoformat(),
        "amount": txn["amount"],
        "is_fraud": 0,
        "hour": hour,
        "is_weekend": is_weekend,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "user_avg_amount": user_avg_amount,
        "user_std_amount": user_std_amount,
        "amount_zscore_user": amount_zscore_user,
        "time_diff": time_diff,
        "txn_velocity": txn_velocity,
        "merchant_txn_count": merchant_txn_count,
        "merchant_unique_users": merchant_unique_users,
        "merchant_fraud_rate": merchant_fraud_rate,
        "device_risk_score": device_risk_score,
        "location_risk_score": location_risk_score,
        "location_device_interaction": device_location_interaction,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "is_high_amount": is_high_amount,
        "is_new_user": is_new_user,
        "is_high_velocity": is_high_velocity,
        "is_risky_device": is_risky_device,
        "is_risky_location": is_risky_location,
        "weighted_amount": weighted_amount,
        "user_risk_score": user_risk_score,
        "days_since_first_txn": days_since_first_txn,
        "txn_amount_ratio_user_avg": txn_amount_ratio_user_avg,
        "is_night": is_night,
        "user_fraud_rate": user_fraud_rate,
        "merchant_user_overlap": merchant_user_overlap,
        "device_txn_count": device_txn_count,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "time_diff_sec": time_diff_sec if not pd.isna(time_diff_sec) else 0.0,
        "txns_in_last_5min": int(txns_in_last_5min),
        "avg_time_gap_prev3": avg_time_gap_prev3,
        "user_device_count": user_device_count,
        "user_location_count": user_location_count,
        "device_location_ratio": device_location_ratio,
        "merchant_past_fraud_rate": merchant_past_fraud_rate,
        "txn_order": txn_order,
        "recent_txn_weight": recent_txn_weight,
        "peer_spend_ratio": peer_spend_ratio
    }

    # Add categorical one-hot exact keys (ensure same names as model_feature_list)
    # transaction_type fields:
    base[f"transaction_type_{transaction_type}"] = 1
    # device type:
    base[f"device_type_{device_type}"] = 1
    # day_of_week:
    base[f"day_of_week_{day_of_week}"] = 1

    # Add raw categorical fields for history
    base["transaction_type"] = transaction_type
    base["device_type"] = device_type
    base["location"] = location
    base["transaction_id"] = txn.get("transaction_id")
    base["user_id"] = user_id
    base["merchant_id"] = merchant_id
    base["timestamp"] = txn_ts.isoformat()
    base["is_fraud"] = 0  # placeholder

    # Build final ordered feature dict matching model_feature_list (fallback to medians/0)
    feature_series = {}
    for feat in model_feature_list:
        if feat in base:
            feature_series[feat] = base[feat]
        else:
            # fallback: if present in training_df use median, else 0
            if training_df is not None and feat in training_df.columns:
                try:
                    val = float(training_df[feat].median(
                    )) if training_df[feat].dtype.kind in "fi" else 0
                except Exception:
                    val = 0
                feature_series[feat] = val
            else:
                # ensure booleans 0/1 for expected dummies
                feature_series[feat] = 0

    # Convert to DataFrame single-row and ensure numeric dtypes
    feature_vector = pd.DataFrame([feature_series], columns=model_feature_list)
    feature_vector = feature_vector.apply(
        pd.to_numeric, errors="coerce").fillna(0)

    # Prepare enriched_for_history dict to be saved (flat)
    enriched_for_history = base.copy()

    return feature_vector, enriched_for_history
