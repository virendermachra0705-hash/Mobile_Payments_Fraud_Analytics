# feature_builder.py
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime, timedelta

HISTORY_PATH = "../data/state/transactions_history.csv"
API_LOG_PATH = "../reports/api_logs.csv"

# Ensure directories exist (caller typically ensures, but safe here)
os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
os.makedirs(os.path.dirname(API_LOG_PATH), exist_ok=True)


def _safe_datetime(ts):
    """Return pandas.Timestamp for many inputs; fallback to now."""
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.Timestamp.utcnow()


def load_history():
    """Load history used to compute stateful features. If missing, return empty DF with expected columns."""
    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
            return df
        except Exception:
            # if file exists but is corrupted, return empty structure
            pass
    cols = ["transaction_id", "user_id", "merchant_id", "amount", "timestamp",
            "device_type", "transaction_type", "location", "is_fraud"]
    return pd.DataFrame(columns=cols)


def save_history_row(row_dict):
    """Append enriched transaction dict to history CSV (create file if not exists)."""
    file_exists = os.path.exists(HISTORY_PATH)
    # Ensure timestamp is ISO string
    if "timestamp" in row_dict and not pd.isna(row_dict["timestamp"]):
        try:
            row_dict["timestamp"] = pd.to_datetime(
                row_dict["timestamp"]).isoformat()
        except Exception:
            row_dict["timestamp"] = pd.Timestamp.utcnow().isoformat()

    with open(HISTORY_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)


def append_api_log(log_entry):
    """Append prediction and context to API logs for dashboarding."""
    file_exists = os.path.exists(API_LOG_PATH)
    # Ensure timestamp is ISO string
    if "timestamp" in log_entry:
        try:
            log_entry["timestamp"] = pd.to_datetime(
                log_entry["timestamp"]).isoformat()
        except Exception:
            log_entry["timestamp"] = pd.Timestamp.utcnow().isoformat()

    with open(API_LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def build_features_from_raw(raw_txn: dict, model_feature_list: list):
    """
    Build all features for a single raw transaction using only local history.
    Returns:
       feature_vector (pd.DataFrame single-row) aligned to model_feature_list,
       enriched_for_history (dict) which will be appended to history (includes timestamp/is_fraud)
    """

    hist = load_history()

    # Normalize raw
    txn = raw_txn.copy()
    txn_ts = _safe_datetime(txn.get("timestamp"))
    txn["timestamp"] = txn_ts if not pd.isna(txn_ts) else pd.Timestamp.utcnow()

    # ensure keys
    for k in ["transaction_id", "user_id", "merchant_id", "amount", "device_type", "transaction_type", "location"]:
        txn.setdefault(k, None)

    # coerce amount
    try:
        txn["amount"] = float(txn.get("amount", 0.0) or 0.0)
    except Exception:
        txn["amount"] = 0.0

    # Derive time fields
    hour = int(pd.to_datetime(txn["timestamp"]).hour)
    day_of_week = pd.to_datetime(txn["timestamp"]).day_name()
    is_weekend = int(day_of_week in ("Saturday", "Sunday"))

    # User historical slice
    user_id = txn.get("user_id")
    if user_id is not None and user_id in hist["user_id"].values:
        user_hist = hist[hist["user_id"] == user_id].sort_values("timestamp")
    else:
        user_hist = pd.DataFrame()

    # User aggregates (exclude current txn)
    user_txn_count = int(user_hist.shape[0])
    user_txn_sum = float(user_hist["amount"].sum()
                         ) if user_txn_count > 0 else 0.0
    user_avg_amount = float(
        user_hist["amount"].mean()) if user_txn_count > 0 else 0.0
    user_std_amount = float(
        user_hist["amount"].std()) if user_txn_count > 1 else 0.0

    # Use reasonable non-breaking defaults
    if pd.isna(user_std_amount) or user_std_amount == 0:
        user_std_amount = 0.0

    # amount zscore vs user's distribution
    if user_txn_count >= 2 and user_std_amount > 0:
        amount_zscore_user = (
            txn["amount"] - user_avg_amount) / user_std_amount
    else:
        # if insufficient history, keep 0 (pure history-driven, no global fallback)
        amount_zscore_user = 0.0

    # time_diff seconds since last txn for user
    time_diff_sec = 0.0
    if not user_hist.empty:
        try:
            last_ts = pd.to_datetime(user_hist["timestamp"].max())
            time_diff_sec = (pd.to_datetime(
                txn["timestamp"]) - last_ts).total_seconds()
            if pd.isna(time_diff_sec):
                time_diff_sec = 0.0
        except Exception:
            time_diff_sec = 0.0

    # txns in last 1h / 24h
    txn_count_1h = 0
    txn_count_24h = 0
    if not user_hist.empty:
        t_cut_1h = pd.to_datetime(txn["timestamp"]) - pd.Timedelta(hours=1)
        t_cut_24h = pd.to_datetime(txn["timestamp"]) - pd.Timedelta(hours=24)
        txn_count_1h = int(
            user_hist[user_hist["timestamp"] >= t_cut_1h].shape[0])
        txn_count_24h = int(
            user_hist[user_hist["timestamp"] >= t_cut_24h].shape[0])

    # txn_order (0-based index of next txn)
    txn_order = user_txn_count

    # rolling fraud rate (7 days)
    rolling_fraud_rate_user_7d = 0.0
    if not user_hist.empty:
        cut7 = pd.to_datetime(txn["timestamp"]) - pd.Timedelta(days=7)
        rec = user_hist[user_hist["timestamp"] >= cut7]
        if not rec.empty and "is_fraud" in rec.columns:
            rolling_fraud_rate_user_7d = float(rec["is_fraud"].mean())

    # user historical fraud rate
    user_fraud_rate = 0.0
    if not user_hist.empty and "is_fraud" in user_hist.columns:
        user_fraud_rate = float(user_hist["is_fraud"].mean())

    # Merchant stats
    merchant_id = txn.get("merchant_id")
    merchant_hist = hist[hist["merchant_id"] ==
                         merchant_id] if merchant_id is not None and merchant_id in hist["merchant_id"].values else pd.DataFrame()
    merchant_txn_count = int(
        merchant_hist.shape[0]) if not merchant_hist.empty else 0
    merchant_unique_users = int(
        merchant_hist["user_id"].nunique()) if not merchant_hist.empty else 0
    merchant_fraud_rate = float(merchant_hist["is_fraud"].mean()) if (
        not merchant_hist.empty and "is_fraud" in merchant_hist.columns) else 0.0

    # Device & location risk proxies
    device_type = txn.get("device_type") or "Unknown"
    if not hist.empty and device_type in hist["device_type"].values:
        device_risk_score = float(hist[hist["device_type"] == device_type]["is_fraud"].mean(
        )) if "is_fraud" in hist.columns else 0.0
        device_txn_count = int(
            hist[hist["device_type"] == device_type].shape[0])
    else:
        device_risk_score = 0.0
        device_txn_count = 0

    location = txn.get("location") or "Unknown"
    if not hist.empty and location in hist["location"].values:
        location_risk_score = float(hist[hist["location"] == location]["is_fraud"].mean(
        )) if "is_fraud" in hist.columns else 0.0
        location_txn_count = int(hist[hist["location"] == location].shape[0])
    else:
        location_risk_score = 0.0
        location_txn_count = 0

    # Flags
    is_night = int(hour in (0, 1, 2, 3, 4, 5))
    is_new_user = int(user_txn_count == 0)
    is_high_amount = int(txn["amount"] > (
        user_avg_amount * 3) if user_avg_amount > 0 else 0)
    is_high_velocity = int(txn_count_1h >= 5)

    # txns in last 5 min
    txns_in_last_5min = 0
    if not user_hist.empty:
        cut5 = pd.to_datetime(txn["timestamp"]) - pd.Timedelta(minutes=5)
        txns_in_last_5min = int(
            user_hist[user_hist["timestamp"] >= cut5].shape[0])

    # weighted amount & recent weight
    weighted_amount = txn["amount"] * (1 + merchant_fraud_rate)
    recent_txn_weight = 1.0 / (1 + txn_order)

    # device_location_interaction
    device_location_interaction = 0
    if not hist.empty:
        device_location_interaction = int(hist[(hist["device_type"] == device_type) & (
            hist["location"] == location)].shape[0])

    user_device_count = int(
        user_hist["device_type"].nunique()) if not user_hist.empty else 0
    user_location_count = int(
        user_hist["location"].nunique()) if not user_hist.empty else 0

    txn_velocity = float(txn_count_24h) / 24.0 if txn_count_24h > 0 else 0.0

    # peer_spend_ratio relative to merchant peers (history)
    if not merchant_hist.empty and not merchant_hist["amount"].empty:
        peer_median = float(merchant_hist["amount"].median())
        peer_spend_ratio = txn["amount"] / \
            peer_median if peer_median > 0 else 0.0
    else:
        peer_spend_ratio = 0.0

    merchant_user_overlap = merchant_unique_users / (merchant_txn_count + 1)

    # Build base dict of features (note: do NOT include timestamp/is_fraud)
    base = {
        "amount": txn["amount"],
        "hour": hour,
        "is_weekend": is_weekend,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "user_avg_amount": user_avg_amount,
        "user_std_amount": user_std_amount,
        "amount_zscore_user": amount_zscore_user,
        # keep 'time_diff' name if model expects; else ensure name matches schema
        "time_diff": time_diff_sec,
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
        "is_risky_device": 1 if device_risk_score > 0.5 else 0,
        "is_risky_location": 1 if location_risk_score > 0.5 else 0,
        "weighted_amount": weighted_amount,
        "user_risk_score": user_fraud_rate,  # user-level risk proxy
        "days_since_first_txn": user_txn_count and int((pd.to_datetime(txn["timestamp"]) - pd.to_datetime(user_hist["timestamp"].min())).days) or 0,
        "txn_amount_ratio_user_avg": (txn["amount"] / user_avg_amount) if user_avg_amount > 0 else 0,
        "is_night": is_night,
        "user_fraud_rate": user_fraud_rate,
        "merchant_user_overlap": merchant_user_overlap,
        "device_txn_count": device_txn_count,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "time_diff_sec": time_diff_sec,
        "txns_in_last_5min": txns_in_last_5min,
        # advanced; left 0 for now (can compute from last 3 txns)
        "avg_time_gap_prev3": 0.0,
        "user_device_count": user_device_count,
        "user_location_count": user_location_count,
        "device_location_ratio": device_location_interaction / (user_location_count+1) if user_location_count >= 0 else 0,
        "merchant_past_fraud_rate": merchant_fraud_rate,
        "txn_order": txn_order,
        "recent_txn_weight": recent_txn_weight,
        "peer_spend_ratio": peer_spend_ratio
    }

    # One-hot indicators for categorical fields according to naming convention expected by model
    one_hots = {
        f"transaction_type_{txn.get('transaction_type') or 'purchase'}": 1,
        f"device_type_{txn.get('device_type') or 'Unknown'}": 1,
        f"day_of_week_{day_of_week}": 1
    }

    # Combine
    final = {**base, **one_hots}

    # Align strictly to model_feature_list (any missing -> 0)
    vector = {feat: final.get(feat, 0) for feat in model_feature_list}
    feature_vector = pd.DataFrame([vector], columns=model_feature_list)

    # Prepare enriched record (for history); include raw categorical fields and will add timestamp/is_fraud in caller
    enriched = {
        "transaction_id": txn.get("transaction_id"),
        "user_id": txn.get("user_id"),
        "merchant_id": txn.get("merchant_id"),
        "amount": txn.get("amount"),
        "timestamp": pd.to_datetime(txn["timestamp"]).isoformat(),
        "device_type": device_type,
        "transaction_type": txn.get("transaction_type"),
        "location": location,
        # include a subset of computed fields for traceability
        **{k: final.get(k, 0) for k in [
            "user_txn_count", "user_avg_amount", "amount_zscore_user", "merchant_fraud_rate",
            "txn_count_24h", "txn_count_1h", "rolling_fraud_rate_user_7d"
        ]},
        # placeholder for is_fraud (caller updates)
        "is_fraud": 0
    }

    return feature_vector, enriched
