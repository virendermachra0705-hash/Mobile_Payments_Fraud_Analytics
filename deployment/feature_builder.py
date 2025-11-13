# feature_builder.py
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# persistent history used to compute rolling features
HISTORY_PATH = "../data/state/transactions_history.csv"
API_LOG_PATH = "../reports/api_logs.csv"

# Ensure directories exist in caller (fraud_api will create them). This module focuses on transforms.


def _safe_datetime(ts):
    # Accept many common formats; return pandas Timestamp
    try:
        return pd.to_datetime(ts)
    except Exception:
        return pd.to_datetime(str(ts), errors='coerce')


def load_history():
    """Load transaction history used to compute stateful features."""
    try:
        hist = pd.read_csv(HISTORY_PATH, parse_dates=["timestamp"])
        return hist
    except Exception:
        # Return empty df with expected minimal columns
        cols = ["transaction_id", "user_id", "merchant_id", "amount", "timestamp", "device_type",
                "transaction_type", "location", "is_fraud"]
        return pd.DataFrame(columns=cols)


def save_history_row(enriched_txn):
    """Append enriched transaction dict to history CSV (create file if not exists)."""
    import os
    import csv
    os.makedirs("../data/state", exist_ok=True)
    file_exists = os.path.exists(HISTORY_PATH)
    with open(HISTORY_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=enriched_txn.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(enriched_txn)


def append_api_log(log_entry):
    """Append prediction and context to API logs for dashboarding."""
    import os
    import csv
    os.makedirs("../reports", exist_ok=True)
    file_exists = os.path.exists(API_LOG_PATH)
    with open(API_LOG_PATH, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log_entry.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(log_entry)


def build_features_from_raw(raw_txn: dict, model_feature_list: list, training_df: pd.DataFrame):
    """
    raw_txn : dict with keys:
      transaction_id, user_id, merchant_id, amount, timestamp, transaction_type, device_type, location
    model_feature_list: list of expected model features (columns)
    training_df: the training data (used to compute user/merchant baselines when history absent)
    Returns:
      feature_vector: pd.DataFrame (single-row) with columns matching model_feature_list
      enriched_txn_for_history: dict of enriched fields that will be appended to history and api_logs
    """

    # Load history
    hist = load_history()

    # Normalize input
    txn = raw_txn.copy()
    txn["timestamp"] = _safe_datetime(txn.get("timestamp", pd.Timestamp.now()))
    if pd.isna(txn["timestamp"]):
        txn["timestamp"] = pd.Timestamp.now()

    # ensure basic keys
    for k in ["transaction_id", "user_id", "merchant_id", "amount", "device_type", "transaction_type", "location"]:
        if k not in txn:
            txn[k] = None

    # Convert types
    try:
        txn["amount"] = float(txn["amount"])
    except Exception:
        txn["amount"] = 0.0

    # Add derived temporal fields
    txn_ts = pd.Timestamp(txn["timestamp"])
    hour = txn_ts.hour
    day_of_week = txn_ts.day_name()
    is_weekend = 1 if day_of_week in ["Saturday", "Sunday"] else 0

    # Compute user-level aggregates from history (including current txn for "what-if" behavior)
    user_id = txn["user_id"]
    if user_id is not None and user_id in hist["user_id"].values:
        user_hist = hist[hist["user_id"] == user_id].sort_values("timestamp")
    else:
        user_hist = pd.DataFrame()

    # lifetime counts and stats (excluding current txn)
    user_txn_count = int(user_hist.shape[0])
    if user_txn_count > 0:
        user_txn_sum = user_hist["amount"].sum()
        user_avg_amount = float(user_hist["amount"].mean())
        user_std_amount = float(user_hist["amount"].std()) if user_hist["amount"].std(
        ) == user_hist["amount"].std() else 0.0
    else:
        # fallback to training population stats for new users
        user_txn_sum = 0.0
        user_avg_amount = float(training_df["user_avg_amount"].median(
        )) if "user_avg_amount" in training_df.columns else 0.0
        user_std_amount = float(training_df["user_std_amount"].median(
        )) if "user_std_amount" in training_df.columns else 0.0

    # amount z-score relative to user's history (if few txns, fallback to global)
    if user_txn_count >= 2 and user_std_amount > 0:
        amount_zscore_user = (
            txn["amount"] - user_avg_amount) / user_std_amount
    else:
        global_avg = training_df["user_avg_amount"].median(
        ) if "user_avg_amount" in training_df.columns else 0.0
        global_std = training_df["user_std_amount"].median(
        ) if "user_std_amount" in training_df.columns else 1.0
        amount_zscore_user = (txn["amount"] - global_avg) / \
            (global_std if global_std != 0 else 1.0)

    # time-based features
    # compute time diff since last txn for same user (seconds)
    time_diff_sec = None
    if not user_hist.empty:
        last_ts = user_hist["timestamp"].max()
        time_diff_sec = (txn_ts - pd.Timestamp(last_ts)).total_seconds()
        if pd.isna(time_diff_sec):
            time_diff_sec = 0.0
    else:
        time_diff_sec = np.nan

    # txns in last 1h and 24h for velocity
    if not user_hist.empty:
        one_h_ago = txn_ts - pd.Timedelta(hours=1)
        day_ago = txn_ts - pd.Timedelta(hours=24)
        txn_count_1h = user_hist[user_hist["timestamp"] >= one_h_ago].shape[0]
        txn_count_24h = user_hist[user_hist["timestamp"] >= day_ago].shape[0]
    else:
        txn_count_1h = 0
        txn_count_24h = 0

    # transaction order index per user (0-based)
    txn_order = user_txn_count

    # rolling fraud rate for user in last 7 days (exclude current txn)
    rolling_fraud_rate_user_7d = 0.0
    if not user_hist.empty:
        seven_days_ago = txn_ts - pd.Timedelta(days=7)
        recent = user_hist[user_hist["timestamp"] >= seven_days_ago]
        if recent.shape[0] > 0 and "is_fraud" in recent.columns:
            rolling_fraud_rate_user_7d = float(recent["is_fraud"].mean())
    # user historical fraud rate
    user_fraud_rate = float(user_hist["is_fraud"].mean()) if (
        not user_hist.empty and "is_fraud" in user_hist.columns) else 0.0

    # merchant-level stats
    merchant_id = txn["merchant_id"]
    if merchant_id is not None and merchant_id in hist["merchant_id"].values:
        merchant_hist = hist[hist["merchant_id"] == merchant_id]
        merchant_txn_count = int(merchant_hist.shape[0])
        merchant_unique_users = int(merchant_hist["user_id"].nunique())
        merchant_fraud_rate = float(merchant_hist["is_fraud"].mean(
        )) if "is_fraud" in merchant_hist.columns else 0.0
    else:
        merchant_txn_count = 0
        merchant_unique_users = 0
        merchant_fraud_rate = float(training_df["merchant_fraud_rate"].median(
        )) if "merchant_fraud_rate" in training_df.columns else 0.0

    # device and location risk proxies (simple heuristics from history)
    device_type = txn.get("device_type", "Unknown")
    if device_type in hist["device_type"].values:
        device_risk_score = float(hist[hist["device_type"] == device_type]["is_fraud"].mean(
        )) if "is_fraud" in hist.columns else 0.0
        device_txn_count = int(
            hist[hist["device_type"] == device_type].shape[0])
    else:
        device_risk_score = 0.0
        device_txn_count = 0

    location = txn.get("location", "Unknown")
    if location in hist["location"].values:
        location_risk_score = float(hist[hist["location"] == location]["is_fraud"].mean(
        )) if "is_fraud" in hist.columns else 0.0
        location_txn_count = int(hist[hist["location"] == location].shape[0])
    else:
        location_risk_score = 0.0
        location_txn_count = 0

    # some boolean/rule flags
    is_night = 1 if hour in [0, 1, 2, 3, 4, 5] else 0
    is_weekend = is_weekend
    is_new_user = 1 if user_txn_count == 0 else 0
    is_high_amount = 1 if txn["amount"] > (
        user_avg_amount * 3 if user_avg_amount > 0 else training_df["amount"].median()) else 0
    is_high_velocity = 1 if txn_count_1h >= 5 else 0  # example threshold
    txns_in_last_5min = 0
    if not user_hist.empty:
        window_5m = txn_ts - pd.Timedelta(minutes=5)
        txns_in_last_5min = int(
            user_hist[user_hist["timestamp"] >= window_5m].shape[0])
    else:
        txns_in_last_5min = 0

    # simple weighted amount (example combining amount and merchant risk)
    weighted_amount = txn["amount"] * (1 + merchant_fraud_rate)

    # recent_txn_weight (example exponential decay using txn_order)
    recent_txn_weight = 1.0 / (1 + txn_order)

    # device_location_interaction: how many txns with same device+location
    if not hist.empty:
        device_location_interaction = int(hist[(hist["device_type"] == device_type) & (
            hist["location"] == location)].shape[0])
    else:
        device_location_interaction = 0

    # user_device_count & user_location_count
    user_device_count = int(
        user_hist["device_type"].nunique()) if not user_hist.empty else 0
    user_location_count = int(
        user_hist["location"].nunique()) if not user_hist.empty else 0

    # txn_velocity (txns per hour in last 24h)
    txn_velocity = 0.0
    if txn_count_24h > 0:
        txn_velocity = txn_count_24h / 24.0

    # peer_spend_ratio: current amount divided by median of users in same city / merchant (approx using merchant users)
    if merchant_id is not None and merchant_id in hist["merchant_id"].values:
        merchant_peers = hist[hist["merchant_id"] == merchant_id]
        peer_median = merchant_peers["amount"].median(
        ) if not merchant_peers.empty else 0.0
        peer_spend_ratio = txn["amount"] / (peer_median if peer_median > 0 else (
            training_df["amount"].median() if "amount" in training_df.columns else 1.0))
    else:
        peer_spend_ratio = txn["amount"] / (
            training_df["amount"].median() if "amount" in training_df.columns else 1.0)

    # merchant_user_overlap (unique users / total txns for merchant) - avoid div by zero
    merchant_user_overlap = merchant_unique_users / (merchant_txn_count + 1)

    # Build the feature dict (only features we engineered in training; best-effort)
    base = {
        "timestamp": txn_ts.isoformat(),
        "transaction_id": txn.get("transaction_id"),
        "user_id": user_id,
        "merchant_id": merchant_id,
        "amount": txn["amount"],
        "hour": hour,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "user_txn_count": user_txn_count,
        "user_txn_sum": user_txn_sum,
        "user_avg_amount": user_avg_amount,
        "user_std_amount": user_std_amount,
        "amount_zscore_user": amount_zscore_user,
        "time_diff_sec": time_diff_sec if not pd.isna(time_diff_sec) else 0.0,
        "txn_order": txn_order,
        "txn_count_1h": txn_count_1h,
        "txn_count_24h": txn_count_24h,
        "rolling_fraud_rate_user_7d": rolling_fraud_rate_user_7d,
        "user_fraud_rate": user_fraud_rate,
        "merchant_txn_count": merchant_txn_count,
        "merchant_unique_users": merchant_unique_users,
        "merchant_fraud_rate": merchant_fraud_rate,
        "device_risk_score": device_risk_score,
        "device_txn_count": device_txn_count,
        "location_risk_score": location_risk_score,
        "location_txn_count": location_txn_count,
        "device_location_interaction": device_location_interaction,
        "is_night": is_night,
        "is_new_user": is_new_user,
        "is_high_amount": is_high_amount,
        "is_high_velocity": is_high_velocity,
        "txns_in_last_5min": txns_in_last_5min,
        "weighted_amount": weighted_amount,
        "recent_txn_weight": recent_txn_weight,
        "user_device_count": user_device_count,
        "user_location_count": user_location_count,
        "txn_velocity": txn_velocity,
        "peer_spend_ratio": peer_spend_ratio,
        "merchant_user_overlap": merchant_user_overlap
    }

    # Add one-hot columns for device_type, transaction_type, day_of_week using expected naming convention
    # We'll create a temporary df and then reindex to model_feature_list later.
    temp_df = pd.DataFrame([base])

    # Add original categorical raw values (for historical storage)
    base["device_type"] = device_type
    base["transaction_type"] = txn.get("transaction_type", "purchase")
    base["location"] = location

    # Add dummies for common categories (these will be zero if model expected columns present)
    # We'll create the final features DataFrame by mapping against model_feature_list
    # Build a dictionary of categorical indicators
    cat_map = {}
    # transaction_type_*
    cat_map[f"transaction_type_{base['transaction_type']}"] = 1
    # device_type_*
    cat_map[f"device_type_{device_type}"] = 1
    # day_of_week_*
    cat_map[f"day_of_week_{day_of_week}"] = 1

    # Combine base and cat_map into final flat dict to be later reindexed
    final_dict = base.copy()
    for k, v in cat_map.items():
        final_dict[k] = v

    # Now produce a DataFrame aligned with model_feature_list
    feature_series = {}
    for feat in model_feature_list:
        if feat in final_dict:
            feature_series[feat] = final_dict[feat]
        else:
            # if training df had some features (like user_risk_score) but we didn't compute explicitly, try to fill from training medians
            if feat in training_df.columns:
                # use column median as fallback
                feature_series[feat] = float(training_df[feat].median(
                )) if training_df[feat].dtype.kind in "fi" else 0
            else:
                feature_series[feat] = 0

    feature_vector = pd.DataFrame([feature_series], columns=model_feature_list)

    # Convert boolean-like to int
    for c in feature_vector.columns:
        if feature_vector[c].dtype == bool:
            feature_vector[c] = feature_vector[c].astype(int)

    # Prepare enriched transaction dict that we will save in history and logs
    enriched_for_history = final_dict.copy()
    # Add prediction placeholders and timestamp will be filled later by caller
    enriched_for_history.setdefault("is_fraud", 0)

    return feature_vector, enriched_for_history
