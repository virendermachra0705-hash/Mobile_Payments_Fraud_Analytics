# feature_builder.py
"""
Production feature builder that mirrors notebook feature engineering.
- Uses history at ../data/state/transactions_history.csv (append-only)
- Uses training summary from ../data/processed/features_enriched_v2.csv
- Returns (feature_vector_df, enriched_record_dict)
- Writes last feature vector to ../reports/last_feature_vector.csv for debugging
"""
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

HISTORY_PATH = "../data/state/transactions_history.csv"
TRAINING_PATH = "../data/processed/features_enriched_v2.csv"
API_LOG_PATH = "../reports/api_logs.csv"
LAST_FEATURE_PATH = "../reports/last_feature_vector.csv"


def _safe_datetime(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.to_datetime(str(ts), errors="coerce")


def load_history():
    if os.path.exists(HISTORY_PATH):
        try:
            return pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        except Exception:
            # fallback if file corrupt
            return pd.DataFrame()
    else:
        return pd.DataFrame()


def load_training_df():
    if os.path.exists(TRAINING_PATH):
        try:
            return pd.read_csv(TRAINING_PATH)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def save_history_row(enriched_txn):
    os.makedirs(os.path.dirname(HISTORY_PATH), exist_ok=True)
    write_header = not os.path.exists(HISTORY_PATH)
    # ensure timestamp is string
    row = enriched_txn.copy()
    if "timestamp" in row and not isinstance(row["timestamp"], str):
        row["timestamp"] = pd.to_datetime(row["timestamp"]).isoformat()
    import csv
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def append_api_log(log_entry):
    os.makedirs(os.path.dirname(API_LOG_PATH), exist_ok=True)
    write_header = not os.path.exists(API_LOG_PATH)
    import csv
    with open(API_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(log_entry)


def _write_last_feature_vector(df):
    os.makedirs(os.path.dirname(LAST_FEATURE_PATH), exist_ok=True)
    df.to_csv(LAST_FEATURE_PATH, index=False)


def build_features_from_raw(raw_txn: dict, model_feature_list: list):
    """
    Build features according to notebook logic and align with model_feature_list.
    Returns (feature_vector_df, enriched_record_dict)
    """
    # load history and training
    hist = load_history()
    training_df = load_training_df()

    # normalize input
    txn = raw_txn.copy()
    txn_ts = _safe_datetime(txn.get("timestamp", pd.Timestamp.utcnow()))
    if pd.isna(txn_ts):
        txn_ts = pd.Timestamp.utcnow()
    txn["timestamp"] = txn_ts

    # ensure keys
    transaction_id = txn.get("transaction_id")
    user_id = txn.get("user_id")
    merchant_id = txn.get("merchant_id")
    try:
        amount = float(txn.get("amount", 0.0))
    except Exception:
        amount = 0.0
    device_type = txn.get("device_type", "Unknown")
    transaction_type = txn.get("transaction_type", "purchase")
    location = txn.get("location", "Unknown")

    # training-derived thresholds / medians
    def _median(col, default=0.0):
        if col in training_df.columns and not training_df[col].isna().all():
            try:
                return float(training_df[col].median())
            except Exception:
                return default
        return default

    median_amount = _median("amount", 1.0)
    median_txn_velocity = _median("txn_velocity", 0.0)
    median_device_risk = _median("device_risk_score", 0.0)
    median_location_risk = _median("location_risk_score", 0.0)
    median_user_avg_amount = _median("user_avg_amount", median_amount)

    # historical slices (exclude current)
    if not hist.empty and "user_id" in hist.columns and pd.notna(user_id):
        user_hist = hist[hist["user_id"] == user_id].sort_values("timestamp")
    else:
        user_hist = pd.DataFrame()

    # user aggregates
    user_txn_count = int(user_hist.shape[0]) if not user_hist.empty else 0
    user_txn_sum = float(user_hist["amount"].sum()) if (
        not user_hist.empty and "amount" in user_hist.columns) else 0.0
    user_avg_amount = float(user_hist["amount"].mean()) if (
        not user_hist.empty and user_hist["amount"].notna().any()) else median_user_avg_amount
    # STD fallback: if std=0 or NaN, use training median user_std_amount or 1
    if not user_hist.empty and user_hist["amount"].std() == user_hist["amount"].std():
        user_std_amount = float(user_hist["amount"].std())
        if user_std_amount == 0:
            user_std_amount = _median("user_std_amount", 1.0)
    else:
        user_std_amount = _median("user_std_amount", 1.0)
    if user_std_amount == 0:
        user_std_amount = 1.0

    # amount zscore
    amount_zscore_user = (amount - user_avg_amount) / \
        (user_std_amount if user_std_amount else 1.0)

    # time diff since last user txn (seconds)
    time_diff_sec = 0.0
    if not user_hist.empty and "timestamp" in user_hist.columns:
        try:
            last_ts = pd.to_datetime(user_hist["timestamp"]).max()
            time_diff_sec = (pd.to_datetime(txn_ts) -
                             pd.to_datetime(last_ts)).total_seconds()
            if pd.isna(time_diff_sec):
                time_diff_sec = 0.0
        except Exception:
            time_diff_sec = 0.0
    else:
        # fallback large value (user has no history)
        time_diff_sec = 999999.0

    # time_diff in minutes
    time_diff = time_diff_sec / 60.0

    # txn velocity = 1/(time_diff+1)
    txn_velocity = 1.0 / (time_diff_sec + 1.0)

    # merchant-level aggregates
    if not hist.empty and "merchant_id" in hist.columns and pd.notna(merchant_id) and merchant_id in hist["merchant_id"].values:
        merchant_hist = hist[hist["merchant_id"] == merchant_id]
        merchant_txn_count = int(merchant_hist.shape[0])
        merchant_unique_users = int(merchant_hist["user_id"].nunique(
        )) if "user_id" in merchant_hist.columns else 0
        merchant_fraud_rate = float(merchant_hist["is_fraud"].mean(
        )) if "is_fraud" in merchant_hist.columns else _median("merchant_fraud_rate", 0.0)
    else:
        merchant_txn_count = 0
        merchant_unique_users = 0
        merchant_fraud_rate = _median("merchant_fraud_rate", 0.0)

    # merchant_user_overlap
    if merchant_txn_count > 0:
        merchant_user_overlap = merchant_unique_users / merchant_txn_count
    else:
        merchant_user_overlap = 0.0

    # device and location risk from history (device_type/location may be missing in hist)
    if not hist.empty and "device_type" in hist.columns:
        device_txn_count = int(
            hist[hist["device_type"] == device_type].shape[0])
        device_risk_score = float(hist[hist["device_type"] == device_type]["is_fraud"].mean(
        )) if device_txn_count > 0 and "is_fraud" in hist.columns else _median("device_risk_score", 0.0)
    else:
        device_txn_count = 0
        device_risk_score = _median("device_risk_score", 0.0)

    if not hist.empty and "location" in hist.columns:
        location_txn_count = int(hist[hist["location"] == location].shape[0])
        location_risk_score = float(hist[hist["location"] == location]["is_fraud"].mean(
        )) if location_txn_count > 0 and "is_fraud" in hist.columns else _median("location_risk_score", 0.0)
    else:
        location_txn_count = 0
        location_risk_score = _median("location_risk_score", 0.0)

    # location_device_interaction
    location_device_interaction = device_risk_score * location_risk_score

    # rolling counts (use history timestamps)
    if not user_hist.empty and "timestamp" in user_hist.columns:
        # create series of timestamps (exclude current)
        hist_ts = pd.to_datetime(user_hist["timestamp"])
        # count txns in last 1H and 24H relative to txn_ts
        one_h_ago = pd.to_datetime(txn_ts) - pd.Timedelta(hours=1)
        day_ago = pd.to_datetime(txn_ts) - pd.Timedelta(hours=24)
        txn_count_1h = int(
            user_hist[user_hist["timestamp"] >= one_h_ago].shape[0])
        txn_count_24h = int(
            user_hist[user_hist["timestamp"] >= day_ago].shape[0])
    else:
        txn_count_1h = 0
        txn_count_24h = 0

    # is_high_amount, is_new_user, is_high_velocity thresholds from training
    is_high_amount = 1 if amount > (
        3 * user_avg_amount if user_avg_amount > 0 else 3 * median_amount) else 0
    is_new_user = 1 if user_txn_count < 5 else 0

    # threshold for high velocity: use training 95th percentile of txn_velocity
    try:
        velocity_thr = float(training_df["txn_velocity"].quantile(
            0.95)) if "txn_velocity" in training_df.columns else (median_txn_velocity + 1e-6)
    except Exception:
        velocity_thr = median_txn_velocity + 1e-6
    is_high_velocity = 1 if txn_velocity > velocity_thr else 0

    # is_risky_device / is_risky_location thresholds from training 90th quantile
    try:
        device_thr = float(training_df["device_risk_score"].quantile(
            0.9)) if "device_risk_score" in training_df.columns else median_device_risk
    except Exception:
        device_thr = median_device_risk
    try:
        location_thr = float(training_df["location_risk_score"].quantile(
            0.9)) if "location_risk_score" in training_df.columns else median_location_risk
    except Exception:
        location_thr = median_location_risk
    is_risky_device = 1 if device_risk_score > device_thr else 0
    is_risky_location = 1 if location_risk_score > location_thr else 0

    # weighted amount and user risk score
    weighted_amount = amount * device_risk_score * location_risk_score
    user_risk_score = float((device_risk_score + location_risk_score) / 2.0)

    # rolling fraud rate user 7d (exclude current)
    if not user_hist.empty and "timestamp" in user_hist.columns and "is_fraud" in user_hist.columns:
        recent = user_hist[pd.to_datetime(user_hist["timestamp"]) >= (
            pd.to_datetime(txn_ts) - pd.Timedelta(days=7))]
        rolling_fraud_rate_user_7d = float(
            recent["is_fraud"].mean()) if not recent.empty else 0.0
    else:
        rolling_fraud_rate_user_7d = 0.0

    # user historical fraud rate (exclude current)
    user_fraud_rate = float(user_hist["is_fraud"].mean()) if (
        not user_hist.empty and "is_fraud" in user_hist.columns) else 0.0

    # txns_in_last_5min (history)
    if not user_hist.empty and "timestamp" in user_hist.columns:
        window_5m = pd.to_datetime(txn_ts) - pd.Timedelta(minutes=5)
        txns_in_last_5min = int(user_hist[pd.to_datetime(
            user_hist["timestamp"]) >= window_5m].shape[0])
    else:
        txns_in_last_5min = 0

    # avg_time_gap_prev3 from history (seconds)
    avg_time_gap_prev3 = 0.0
    if not user_hist.empty and "timestamp" in user_hist.columns:
        ts = pd.to_datetime(user_hist["timestamp"]).sort_values()
        if len(ts) >= 2:
            gaps = ts.diff().dt.total_seconds().dropna()
            avg_time_gap_prev3 = float(gaps.tail(3).mean()) if not gaps.tail(
                3).empty else float(gaps.mean())
        else:
            avg_time_gap_prev3 = float(_median("time_diff_sec", 0.0))

    # peer_spend_ratio: amount / user median (use training fallback)
    try:
        user_median_amount = float(user_hist["amount"].median()) if (
            not user_hist.empty and "amount" in user_hist.columns) else _median("amount", median_amount)
    except Exception:
        user_median_amount = median_amount
    peer_spend_ratio = amount / \
        (user_median_amount if user_median_amount != 0 else 1.0)

    # recent_txn_weight and txn_order
    txn_order = int(user_txn_count)
    recent_txn_weight = float(np.exp(-0.01 * txn_order))

    # device_location_ratio
    device_location_ratio = device_txn_count / \
        (location_txn_count + 1) if (location_txn_count + 1) > 0 else 0.0

    # user_device_count and user_location_count
    user_device_count = int(user_hist["device_type"].nunique()) if (
        not user_hist.empty and "device_type" in user_hist.columns) else 0
    user_location_count = int(user_hist["location"].nunique()) if (
        not user_hist.empty and "location" in user_hist.columns) else 0

    # Build feature dict (names must match training)
    features = {
        "timestamp": pd.to_datetime(txn_ts).isoformat(),
        "amount": amount,
        "hour": int(pd.to_datetime(txn_ts).hour),
        "is_weekend": 1 if pd.to_datetime(txn_ts).day_name() in ["Saturday", "Sunday"] else 0,
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
        "location_device_interaction": location_device_interaction,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "is_high_amount": is_high_amount,
        "is_new_user": is_new_user,
        "is_high_velocity": is_high_velocity,
        "is_risky_device": is_risky_device,
        "is_risky_location": is_risky_location,
        "weighted_amount": weighted_amount,
        "user_risk_score": user_risk_score,
        "days_since_first_txn": 0,  # will compute below if history present
        "txn_amount_ratio_user_avg": amount / (user_avg_amount + 1e-9),
        "is_night": 1 if int(pd.to_datetime(txn_ts).hour) in [0, 1, 2, 3, 4, 5] else 0,
        "user_fraud_rate": user_fraud_rate,
        "merchant_user_overlap": merchant_user_overlap,
        "device_txn_count": device_txn_count,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "time_diff_sec": time_diff_sec,
        "txns_in_last_5min": txns_in_last_5min,
        "avg_time_gap_prev3": avg_time_gap_prev3,
        "user_device_count": user_device_count,
        "user_location_count": user_location_count,
        "device_location_ratio": device_location_ratio,
        "merchant_past_fraud_rate": merchant_fraud_rate,
        "txn_order": txn_order,
        "recent_txn_weight": recent_txn_weight,
        "peer_spend_ratio": peer_spend_ratio,
        # categorical one-hot placeholders (explicit names from training)
        "transaction_type_purchase": 1 if transaction_type == "purchase" else 0,
        "transaction_type_top-up": 1 if transaction_type == "top-up" else 0,
        "transaction_type_transfer": 1 if transaction_type == "transfer" else 0,
        "device_type_Web": 1 if device_type == "Web" else 0,
        "device_type_iOS": 1 if device_type == "iOS" else 0,
        "day_of_week_Monday": 1 if pd.to_datetime(txn_ts).day_name() == "Monday" else 0,
        "day_of_week_Saturday": 1 if pd.to_datetime(txn_ts).day_name() == "Saturday" else 0,
        "day_of_week_Sunday": 1 if pd.to_datetime(txn_ts).day_name() == "Sunday" else 0,
        "day_of_week_Thursday": 1 if pd.to_datetime(txn_ts).day_name() == "Thursday" else 0,
        "day_of_week_Tuesday": 1 if pd.to_datetime(txn_ts).day_name() == "Tuesday" else 0,
        "day_of_week_Wednesday": 1 if pd.to_datetime(txn_ts).day_name() == "Wednesday" else 0,
    }

    # compute days_since_first_txn
    if not user_hist.empty and "timestamp" in user_hist.columns:
        first_ts = pd.to_datetime(user_hist["timestamp"]).min()
        features["days_since_first_txn"] = int(
            (pd.to_datetime(txn_ts) - pd.to_datetime(first_ts)).days)
    else:
        features["days_since_first_txn"] = 0

    # Ensure all model features present; fallback to training median or 0
    final = {}
    for feat in model_feature_list:
        if feat in features:
            final[feat] = features[feat]
        else:
            # fallback to training median or 0
            if feat in training_df.columns:
                try:
                    final[feat] = float(training_df[feat].median())
                except Exception:
                    final[feat] = 0.0
            else:
                final[feat] = 0.0

    # convert bools -> ints and ensure numeric dtype
    for k, v in final.items():
        if isinstance(v, (bool, np.bool_)):
            final[k] = int(v)
        elif v is None:
            final[k] = 0.0

    feature_vector = pd.DataFrame([final], columns=model_feature_list)

    # write last feature vector for debugging
    try:
        _write_last_feature_vector(feature_vector.copy())
    except Exception:
        pass

    enriched = {
        "transaction_id": transaction_id,
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": amount,
        "timestamp": pd.to_datetime(txn_ts).isoformat(),
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

    return feature_vector, enriched
