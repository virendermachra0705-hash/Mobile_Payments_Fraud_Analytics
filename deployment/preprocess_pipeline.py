import pandas as pd
import numpy as np


def preprocess_transaction(txn: dict, feature_list: list):
    """
    Convert incoming transaction JSON into a feature-ready DataFrame
    matching model training schema.
    """
    df = pd.DataFrame([txn])

    # Fill missing values with 0
    df = df.fillna(0)

    # Log-transform numeric columns if present
    for col in ["amount", "user_txn_sum", "weighted_amount"]:
        if col in df.columns:
            df[col] = np.log1p(df[col])

    # Ensure all required columns exist (fill missing with 0)
    for feat in feature_list:
        if feat not in df.columns:
            df[feat] = 0

    # Reorder columns to match model
    df = df[feature_list]
    return df
