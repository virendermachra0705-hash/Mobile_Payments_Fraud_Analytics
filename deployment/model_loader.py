# model_loader.py
import joblib
import os
import json
import pandas as pd

MODEL_PATH = "../models/xgboost_tuned.pkl"
ALT_FEATURES_PATH = "../models/feature_names.txt"
TRAINING_DATA_PATH = "../data/processed/features_enriched_v2.csv"


def load_model_and_schema():
    # ---- LOAD MODEL ----
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    # ---- LOAD FEATURE LIST ----
    model_feature_list = None

    # 1) Try booster names
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            model_feature_list = list(booster.feature_names)
    except Exception:
        model_feature_list = None

    # 2) Fallback: TXT file
    if model_feature_list is None and os.path.exists(ALT_FEATURES_PATH):
        with open(ALT_FEATURES_PATH, "r") as f:
            model_feature_list = [line.strip() for line in f if line.strip()]

    # 3) Fallback: model_schema.json
    if model_feature_list is None:
        alt_json = os.path.splitext(MODEL_PATH)[0] + "_schema.json"
        if os.path.exists(alt_json):
            with open(alt_json, "r") as fj:
                schema = json.load(fj)
                model_feature_list = schema.get("feature_list")

    if model_feature_list is None:
        raise RuntimeError(
            "Unable to determine model feature list. "
            "Provide booster.feature_names or feature_names.txt or schema.json."
        )

    # Remove unwanted columns
    model_feature_list = [f for f in model_feature_list
                          if f not in ("timestamp", "is_fraud")]

    # ---- LOAD TRAINING DF ----
    if not os.path.exists(TRAINING_DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found at {TRAINING_DATA_PATH}")

    training_df = pd.read_csv(TRAINING_DATA_PATH)

    return model, model_feature_list, training_df
