
# feature_builder.py
import pandas as pd
import numpy as np
from datetime import datetime

HISTORY_PATH = "../data/state/transactions_history.csv"
TRAINING_DATA_PATH = "../data/processed/features_enriched_v2.csv"
API_LOG_PATH = "../reports/api_logs.csv"


def _safe_datetime(ts):
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.to_datetime(str(ts), errors="coerce")


def load_history():
    try:
        hist = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        return hist
    except Exception:
        cols = [
            "transaction_id", "user_id", "merchant_id", "amount", "timestamp",
            "device_type", "transaction_type", "location", "is_fraud"
        ]
        return pd.DataFrame(columns=cols)


def load_training_df():
    try:
        return pd.read_csv(TRAINING_DATA_PATH)
    except Exception:
        return pd.DataFrame()


def save_history_row(enriched_txn):
    import os
    import csv
    os.makedirs("../data/state", exist_ok=True)
    file_exists = os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=enriched_txn.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(enriched_txn)


def append_api_log(log_entry):
    import os
    import csv
    os.makedirs("../reports", exist_ok=True)
    file_exists = os.path.exists(API_LOG_PATH)
    with open(API_LOG_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def build_features_from_raw(raw_txn: dict, model_feature_list: list):
    """
    Produces feature vector aligned to model_feature_list (names must match training).
    Uses persistent history (HISTORY_PATH) and TRAINING CSV for fallback medians.
    Returns: (pd.DataFrame single-row), enriched_dict_for_history
    """
    hist = load_history()
    training_df = load_training_df()

    txn = raw_txn.copy()
    txn_ts = _safe_datetime(txn.get("timestamp", pd.Timestamp.utcnow()))
    if pd.isna(txn_ts):
        txn_ts = pd.Timestamp.utcnow()
    txn["timestamp"] = txn_ts

    # Raw fields
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

    # TRAINING MEDIANS for fallbacks / thresholds
    def col_median(col, default=0.0):
        if col in training_df.columns and not training_df[col].isna().all():
            return float(training_df[col].median())
        return default

    median_user_txn_count = col_median("user_txn_count", 0)
    median_amount = col_median("amount", 1.0)
    median_device_risk = col_median("device_risk_score", 0.0)
    median_location_risk = col_median("location_risk_score", 0.0)
    median_txn_velocity = col_median("txn_velocity", 0.0)
    median_merchant_fraud = col_median("merchant_fraud_rate", 0.0)
    median_user_avg_amount = col_median("user_avg_amount", median_amount)

    # USER history
    if "user_id" in hist.columns and user_id in hist["user_id"].values:
        user_hist = hist[hist["user_id"] == user_id].sort_values("timestamp")
    else:
        user_hist = pd.DataFrame()

    user_txn_count = int(user_hist.shape[0]) if not user_hist.empty else 0
    user_txn_sum = float(user_hist["amount"].sum()
                         ) if not user_hist.empty else 0.0
    user_avg_amount = float(user_hist["amount"].mean()) if (
        not user_hist.empty and user_hist["amount"].notna().any()) else median_user_avg_amount
    user_std_amount = float(user_hist["amount"].std()) if (not user_hist.empty and user_hist["amount"].std() == user_hist["amount"].std(
    )) else float(training_df["user_std_amount"].median()) if "user_std_amount" in training_df.columns else 0.0

    # amount zscore relative to user
    user_std = user_std_amount if user_std_amount and user_std_amount > 0 else (
        training_df["user_std_amount"].median() if "user_std_amount" in training_df.columns else 1.0)
    amount_zscore_user = (amount - user_avg_amount) / (user_std or 1.0)

    # time diffs
    if not user_hist.empty and "timestamp" in user_hist.columns:
        last_ts = pd.to_datetime(user_hist["timestamp"]).max()
        time_diff_sec = (txn_ts - pd.to_datetime(last_ts)).total_seconds()
    else:
        time_diff_sec = 0.0

    # time_diff (friendly)
    time_diff = time_diff_sec / 60.0  # minutes

    # txn counts for velocity
    if not user_hist.empty and "timestamp" in user_hist.columns:
        one_h_ago = txn_ts - pd.Timedelta(hours=1)
        day_ago = txn_ts - pd.Timedelta(hours=24)
        txn_count_1h = int(
            user_hist[user_hist["timestamp"] >= one_h_ago].shape[0])
        txn_count_24h = int(
            user_hist[user_hist["timestamp"] >= day_ago].shape[0])
    else:
        txn_count_1h = 0
        txn_count_24h = 0

    txn_velocity = txn_count_24h / 24.0 if txn_count_24h > 0 else 0.0

    # txn_order
    txn_order = user_txn_count

    # txns in last 5 min
    if not user_hist.empty and "timestamp" in user_hist.columns:
        window_5m = txn_ts - pd.Timedelta(minutes=5)
        txns_in_last_5min = int(
            user_hist[user_hist["timestamp"] >= window_5m].shape[0])
    else:
        txns_in_last_5min = 0

    # rolling fraud (7d)
    if not user_hist.empty and "timestamp" in user_hist.columns and "is_fraud" in user_hist.columns:
        recent = user_hist[user_hist["timestamp"]
                           >= (txn_ts - pd.Timedelta(days=7))]
        rolling_fraud_rate_user_7d = float(
            recent["is_fraud"].mean()) if not recent.empty else 0.0
    else:
        rolling_fraud_rate_user_7d = 0.0

    user_fraud_rate = float(user_hist["is_fraud"].mean()) if (
        not user_hist.empty and "is_fraud" in user_hist.columns) else 0.0

    # merchant stats
    if "merchant_id" in hist.columns and merchant_id in hist["merchant_id"].values:
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

    # device and location stats
    if "device_type" in hist.columns:
        device_txn_count = int(
            hist[hist["device_type"] == device_type].shape[0])
        device_risk_score = float(hist[hist["device_type"] == device_type]["is_fraud"].mean(
        )) if device_txn_count > 0 and "is_fraud" in hist.columns else 0.0
    else:
        device_txn_count = 0
        device_risk_score = 0.0

    if "location" in hist.columns:
        location_txn_count = int(hist[hist["location"] == location].shape[0])
        location_risk_score = float(hist[hist["location"] == location]["is_fraud"].mean(
        )) if location_txn_count > 0 and "is_fraud" in hist.columns else 0.0
    else:
        location_txn_count = 0
        location_risk_score = 0.0

    # device-location interaction and ratio
    if "device_type" in hist.columns and "location" in hist.columns:
        location_device_interaction = int(hist[(hist["device_type"] == device_type) & (
            hist["location"] == location)].shape[0])
    else:
        location_device_interaction = 0

    device_location_ratio = (
        device_txn_count / (location_txn_count + 1)) if (location_txn_count + 1) > 0 else 0.0

    # days since first txn (approx using earliest timestamp in user history)
    if not user_hist.empty and "timestamp" in user_hist.columns:
        first_ts = pd.to_datetime(user_hist["timestamp"]).min()
        days_since_first_txn = (txn_ts - first_ts).days
    else:
        days_since_first_txn = 0

    # peer spend ratio (merchant peer median)
    if "merchant_id" in hist.columns and merchant_id in hist["merchant_id"].values:
        merchant_peers = hist[hist["merchant_id"] == merchant_id]
        peer_median = merchant_peers["amount"].median(
        ) if not merchant_peers.empty else median_amount
    else:
        peer_median = median_amount
    peer_spend_ratio = amount / (peer_median if peer_median > 0 else 1.0)

    # weighted amount
    weighted_amount = amount * (1.0 + merchant_past_fraud_rate)

    # recent txn weight
    recent_txn_weight = 1.0 / (1 + txn_order)

    # txn_amount_ratio_user_avg
    txn_amount_ratio_user_avg = amount / \
        (user_avg_amount if user_avg_amount > 0 else median_user_avg_amount)

    # avg_time_gap_prev3: estimate from last 3 txns in user_hist (in seconds)
    avg_time_gap_prev3 = 0.0
    if not user_hist.empty and "timestamp" in user_hist.columns:
        last3 = user_hist.sort_values("timestamp").tail(
            4)  # include current -> compute gaps
        ts = pd.to_datetime(last3["timestamp"])
        if len(ts) >= 2:
            gaps = ts.diff().dt.total_seconds().dropna()
            avg_time_gap_prev3 = float(gaps.tail(3).mean()) if not gaps.tail(
                3).empty else float(gaps.mean())

    # user_device_count & user_location_count
    user_device_count = int(user_hist["device_type"].nunique()) if (
        "device_type" in user_hist.columns and not user_hist.empty) else 0
    user_location_count = int(user_hist["location"].nunique()) if (
        "location" in user_hist.columns and not user_hist.empty) else 0

    # user_risk_score: combine historical fraud + normalized velocity/frequency signals
    # (reconstructed logic: weighted combination, scale to 0-1)
    u_fraud = user_fraud_rate
    u_velocity = min(1.0, txn_count_24h / (median_txn_velocity +
                     1e-6)) if median_txn_velocity >= 0 else 0.0
    user_risk_score = float(0.7 * u_fraud + 0.3 * u_velocity)

    # is_risky_device / is_risky_location thresholds: use training medians of device/location risk if available
    is_risky_device = 1 if device_risk_score > median_device_risk else 0
    is_risky_location = 1 if location_risk_score > median_location_risk else 0

    # is_high_amount: amount >> user average (3x) or global median
    is_high_amount = 1 if amount > max(
        3 * user_avg_amount, 3 * median_amount) else 0

    # is_high_velocity set earlier
    is_high_velocity = 1 if txn_count_1h >= 5 else 0

    # is_new_user
    is_new_user = 1 if user_txn_count == 0 else 0

    # is_weekend/is_night/hour/day_of_week
    hour = int(txn_ts.hour)
    dow = txn_ts.day_name()
    is_weekend = 1 if dow in ["Saturday", "Sunday"] else 0
    is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0

    # txn_amount_ratio_user_avg already computed

    # Build final features dict exactly matching training column names
    features = {
        "amount": amount,
        "amount_zscore_user": amount_zscore_user,
        "avg_time_gap_prev3": avg_time_gap_prev3,
        "day_of_week_Monday": 1 if dow == "Monday" else 0,
        "day_of_week_Saturday": 1 if dow == "Saturday" else 0,
        "day_of_week_Sunday": 1 if dow == "Sunday" else 0,
        "day_of_week_Thursday": 1 if dow == "Thursday" else 0,
        "day_of_week_Tuesday": 1 if dow == "Tuesday" else 0,
        "day_of_week_Wednesday": 1 if dow == "Wednesday" else 0,
        "days_since_first_txn": days_since_first_txn,
        "device_location_ratio": device_location_ratio,
        "device_risk_score": device_risk_score,
        "device_txn_count": device_txn_count,
        "device_type_Web": 1 if device_type == "Web" else 0,
        "device_type_iOS": 1 if device_type == "iOS" else 0,
        "hour": hour,
        # "is_fraud" is label; not supplied to model but may be in training file
        "is_high_amount": is_high_amount,
        "is_high_velocity": is_high_velocity,
        "is_new_user": is_new_user,
        "is_night": is_night,
        "is_risky_device": is_risky_device,
        "is_risky_location": is_risky_location,
        "is_weekend": is_weekend,
        "location_device_interaction": location_device_interaction,
        "location_risk_score": location_risk_score,
        "merchant_fraud_rate": merchant_fraud_rate,
        "merchant_past_fraud_rate": merchant_past_fraud_rate,
        "merchant_txn_count": merchant_txn_count,
        "merchant_unique_users": merchant_unique_users,
        "merchant_user_overlap": merchant_user_overlap,
        "peer_spend_ratio": peer_spend_ratio,
        "recent_txn_weight": recent_txn_weight,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "time_diff": time_diff,
        "time_diff_sec": time_diff_sec,
        "timestamp": txn_ts.isoformat(),
        "transaction_type_purchase": 1 if transaction_type == "purchase" else 0,
        "transaction_type_top-up": 1 if transaction_type == "top-up" else 0,
        "transaction_type_transfer": 1 if transaction_type == "transfer" else 0,
        "txn_amount_ratio_user_avg": txn_amount_ratio_user_avg,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "txn_order": txn_order,
        "txn_velocity": txn_velocity,
        "txns_in_last_5min": txns_in_last_5min,
        "user_avg_amount": user_avg_amount,
        "user_device_count": user_device_count,
        "user_fraud_rate": user_fraud_rate,
        "user_location_count": user_location_count,
        "user_risk_score": user_risk_score,
        "user_std_amount": user_std_amount,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "weighted_amount": weighted_amount
    }

    # Ensure all expected model features exist; fill with training medians or 0
    final = {}
    for feat in model_feature_list:
        if feat in features:
            final[feat] = features[feat]
        else:
            # fallback: if training has this col, use median else 0
            if feat in training_df.columns:
                try:
                    final[feat] = float(training_df[feat].median())
                except Exception:
                    final[feat] = 0.0
            else:
                final[feat] = 0.0

    # Convert booleans to int and ensure numeric types
    for k, v in final.items():
        if isinstance(v, (bool, np.bool_)):
            final[k] = int(v)
        elif v is None:
            final[k] = 0.0

    feature_vector = pd.DataFrame([final], columns=model_feature_list)

    # Enriched record to append to history (raw + chosen fields)
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

    return feature_vector, enriched
