# feature_builder.py
import pandas as pd
import numpy as np
import os
import csv
from datetime import datetime, timedelta

HISTORY_PATH = "../data/state/transactions_history.csv"
API_LOG_PATH = "../reports/api_logs.csv"

os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
os.makedirs(os.path.dirname(API_LOG_PATH), exist_ok=True)

def _safe_dt(x):
    try:
        return pd.to_datetime(x)
    except Exception:
        return pd.Timestamp.utcnow()

def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            df = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
            return df
        except Exception:
            # corrupted file: move aside and return empty df
            bak = HISTORY_PATH + ".bak"
            os.rename(HISTORY_PATH, bak)
    cols = ["transaction_id","user_id","merchant_id","amount","timestamp","device_type","transaction_type","location","is_fraud"]
    return pd.DataFrame(columns=cols)

def save_history_row(row_dict):
    file_exists = os.path.exists(HISTORY_PATH)
    # Ensure timestamp iso
    if "timestamp" in row_dict:
        try:
            row_dict["timestamp"] = pd.to_datetime(row_dict["timestamp"]).isoformat()
        except Exception:
            row_dict["timestamp"] = pd.Timestamp.utcnow().isoformat()
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_dict)

def append_api_log(log_entry):
    file_exists = os.path.exists(API_LOG_PATH)
    if "timestamp" in log_entry:
        try:
            log_entry["timestamp"] = pd.to_datetime(log_entry["timestamp"]).isoformat()
        except:
            log_entry["timestamp"] = pd.Timestamp.utcnow().isoformat()
    with open(API_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)

def _get_recent_gaps(series, n=3):
    # series is timestamps sorted asc
    if len(series) < 2:
        return []
    diffs = series.diff().dropna().dt.total_seconds()
    return diffs.tail(n).tolist()

def build_features_from_raw(raw_txn: dict, model_feature_list: list):
    """
    Input: raw_txn dict, expected model_feature_list
    Returns: feature_vector (1-row DataFrame aligned to model_feature_list),
             enriched_for_history (dict for saving to history)
    """
    hist = load_history()

    txn = raw_txn.copy()
    txn_ts = _safe_dt(txn.get("timestamp", pd.Timestamp.utcnow()))
    txn["timestamp"] = txn_ts

    # ensure keys
    for k in ["transaction_id","user_id","merchant_id","amount","device_type","transaction_type","location"]:
        txn.setdefault(k, None)

    try:
        amount = float(txn.get("amount") or 0.0)
    except:
        amount = 0.0
    txn["amount"] = amount

    # time fields
    ts = pd.to_datetime(txn["timestamp"])
    hour = int(ts.hour)
    day_of_week = ts.day_name()
    is_weekend = int(day_of_week in ("Saturday","Sunday"))

    # user-history
    user_id = txn.get("user_id")
    user_hist = hist[hist["user_id"] == user_id].sort_values("timestamp") if (user_id is not None and "user_id" in hist.columns and user_id in hist["user_id"].values) else pd.DataFrame()
    user_txn_count = int(user_hist.shape[0])
    user_txn_sum = float(user_hist["amount"].sum()) if user_txn_count>0 else 0.0
    user_avg_amount = float(user_hist["amount"].mean()) if user_txn_count>0 else 0.0
    user_std_amount = float(user_hist["amount"].std()) if user_txn_count>1 else 0.0
    if pd.isna(user_std_amount) or user_std_amount == 0:
        user_std_amount = 0.0

    # amount zscore relative user
    amount_zscore_user = (amount - user_avg_amount) / user_std_amount if (user_txn_count>=2 and user_std_amount>0) else 0.0

    # time diff to last txn
    time_diff_sec = 0.0
    if not user_hist.empty:
        try:
            last_ts = pd.to_datetime(user_hist["timestamp"].max())
            time_diff_sec = (ts - last_ts).total_seconds()
            if pd.isna(time_diff_sec): time_diff_sec = 0.0
        except:
            time_diff_sec = 0.0

    # txns in last windows
    txn_count_1h = 0
    txn_count_24h = 0
    if not user_hist.empty:
        cut1 = ts - pd.Timedelta(hours=1)
        cut24 = ts - pd.Timedelta(hours=24)
        txn_count_1h = int(user_hist[user_hist["timestamp"] >= cut1].shape[0])
        txn_count_24h = int(user_hist[user_hist["timestamp"] >= cut24].shape[0])

    txn_order = user_txn_count

    # rolling fraud (7d)
    rolling_fraud_rate_user_7d = 0.0
    if not user_hist.empty:
        cut7 = ts - pd.Timedelta(days=7)
        rec = user_hist[user_hist["timestamp"] >= cut7]
        if not rec.empty and "is_fraud" in rec.columns:
            rolling_fraud_rate_user_7d = float(rec["is_fraud"].mean())

    user_fraud_rate = float(user_hist["is_fraud"].mean()) if (not user_hist.empty and "is_fraud" in user_hist.columns) else 0.0

    # merchant stats
    merchant_id = txn.get("merchant_id")
    merchant_hist = hist[hist["merchant_id"] == merchant_id] if (merchant_id is not None and "merchant_id" in hist.columns and merchant_id in hist["merchant_id"].values) else pd.DataFrame()
    merchant_txn_count = int(merchant_hist.shape[0]) if not merchant_hist.empty else 0
    merchant_unique_users = int(merchant_hist["user_id"].nunique()) if not merchant_hist.empty else 0
    merchant_fraud_rate = float(merchant_hist["is_fraud"].mean()) if (not merchant_hist.empty and "is_fraud" in merchant_hist.columns) else 0.0

    # device/location risk
    device_type = txn.get("device_type") or "Unknown"
    if (not hist.empty) and ("device_type" in hist.columns) and (device_type in hist["device_type"].values):
        device_risk_score = float(hist[hist["device_type"] == device_type]["is_fraud"].mean()) if "is_fraud" in hist.columns else 0.0
        device_txn_count = int(hist[hist["device_type"] == device_type].shape[0])
    else:
        device_risk_score = 0.0
        device_txn_count = 0

    location = txn.get("location") or "Unknown"
    if (not hist.empty) and ("location" in hist.columns) and (location in hist["location"].values):
        location_risk_score = float(hist[hist["location"] == location]["is_fraud"].mean()) if "is_fraud" in hist.columns else 0.0
        location_txn_count = int(hist[hist["location"] == location].shape[0])
    else:
        location_risk_score = 0.0
        location_txn_count = 0

    # flags
    is_night = int(hour in (0,1,2,3,4,5))
    is_new_user = int(user_txn_count == 0)
    is_high_amount = int(amount > (user_avg_amount * 3) if user_avg_amount>0 else 0)
    is_high_velocity = int(txn_count_1h >= 5)

    # txns in last 5 min
    txns_in_last_5min = 0
    if not user_hist.empty:
        cut5 = ts - pd.Timedelta(minutes=5)
        txns_in_last_5min = int(user_hist[user_hist["timestamp"] >= cut5].shape[0])

    # weighted & recent weight
    weighted_amount = amount * (1 + merchant_fraud_rate)
    recent_txn_weight = 1.0 / (1 + txn_order)

    # device-location interaction
    device_location_interaction = 0
    if not hist.empty:
        device_location_interaction = int(hist[(hist["device_type"] == device_type) & (hist["location"] == location)].shape[0])

    user_device_count = int(user_hist["device_type"].nunique()) if not user_hist.empty and "device_type" in user_hist.columns else 0
    user_location_count = int(user_hist["location"].nunique()) if not user_hist.empty and "location" in user_hist.columns else 0

    txn_velocity = float(txn_count_24h) / 24.0 if txn_count_24h > 0 else 0.0

    # peer spend ratio relative to merchant peers (history-driven)
    if (not merchant_hist.empty) and ("amount" in merchant_hist.columns) and (not merchant_hist["amount"].dropna().empty):
        peer_median = float(merchant_hist["amount"].median())
        peer_spend_ratio = amount / peer_median if peer_median>0 else 0.0
    else:
        peer_spend_ratio = 0.0

    merchant_user_overlap = merchant_unique_users / (merchant_txn_count + 1)

    # avg_time_gap_prev3 (from last 3 gaps)
    avg_time_gap_prev3 = 0.0
    if not user_hist.empty and "timestamp" in user_hist.columns and user_hist.shape[0] >= 2:
        ts_sorted = pd.to_datetime(user_hist["timestamp"]).sort_values()
        gaps = _get_recent_gaps(ts_sorted, n=3)
        avg_time_gap_prev3 = float(np.mean(gaps)) if len(gaps)>0 else 0.0

    # Build base features dict (ensure names are consistent with training schema)
    base = {
        "amount": amount,
        "hour": hour,
        "is_weekend": is_weekend,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "user_avg_amount": user_avg_amount,
        "user_std_amount": user_std_amount,
        "amount_zscore_user": amount_zscore_user,
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
        "user_risk_score": user_fraud_rate,
        "days_since_first_txn": int((ts - pd.to_datetime(user_hist["timestamp"].min())).days) if (not user_hist.empty and "timestamp" in user_hist.columns) else 0,
        "txn_amount_ratio_user_avg": (amount / user_avg_amount) if user_avg_amount>0 else 0.0,
        "is_night": is_night,
        "user_fraud_rate": user_fraud_rate,
        "merchant_user_overlap": merchant_user_overlap,
        "device_txn_count": device_txn_count,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "time_diff_sec": time_diff_sec,
        "txns_in_last_5min": txns_in_last_5min,
        "avg_time_gap_prev3": avg_time_gap_prev3,
        "user_device_count": user_device_count,
        "user_location_count": user_location_count,
        "device_location_ratio": (device_location_interaction / (user_location_count+1)) if user_location_count>=0 else 0.0,
        "merchant_past_fraud_rate": merchant_fraud_rate,
        "txn_order": txn_order,
        "recent_txn_weight": recent_txn_weight,
        "peer_spend_ratio": peer_spend_ratio
    }

    # Compose one-hot map but only create keys that exist in model_feature_list to avoid unknown columns
    one_hots = {}
    candidate_onehots = [
        f"transaction_type_{txn.get('transaction_type') or 'purchase'}",
        f"device_type_{txn.get('device_type') or 'Unknown'}",
        f"day_of_week_{day_of_week}"
    ]
    for k in candidate_onehots:
        if k in model_feature_list:
            one_hots[k] = 1

    final = {**base, **one_hots}

    # Always include keys in model_feature_list in the same order, missing -> 0
    vector = {feat: final.get(feat, 0) for feat in model_feature_list}
    feature_vector = pd.DataFrame([vector], columns=model_feature_list)

    # Convert bools -> int
    for c in feature_vector.columns:
        if feature_vector[c].dtype == bool:
            feature_vector[c] = feature_vector[c].astype(int)

    # Now apply transforms that model expects (log1p) only if those names present
    for col in ["amount", "user_txn_sum", "weighted_amount"]:
        if col in feature_vector.columns:
            # keep original values in enriched history, but transform for model input
            feature_vector[col] = np.log1p(feature_vector[col].astype(float))

    enriched = {
        "transaction_id": txn.get("transaction_id"),
        "user_id": txn.get("user_id"),
        "merchant_id": txn.get("merchant_id"),
        "amount": amount,
        "timestamp": ts.isoformat(),
        "device_type": device_type,
        "transaction_type": txn.get("transaction_type"),
        "location": location,
        # some trace columns
        "user_txn_count": user_txn_count,
        "user_avg_amount": user_avg_amount,
        "amount_zscore_user": amount_zscore_user,
        "merchant_fraud_rate": merchant_fraud_rate,
        "txn_count_24h": txn_count_24h,
        "txn_count_1h": txn_count_1h,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "is_fraud": 0  # caller will update to predicted flag
    }

    return feature_vector, enriched
