# model_loader.py
import joblib
import os
import json

MODEL_PATH = "../models/xgboost_tuned.pkl"
ALT_FEATURES_PATH = "../models/feature_names.txt"  # optional: one-per-line


def load_model_and_schema():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    model_feature_list = None
    # Try booster feature names (XGBoost)
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            model_feature_list = list(booster.feature_names)
    except Exception:
        model_feature_list = None

    # Fallback to text file
    if model_feature_list is None and os.path.exists(ALT_FEATURES_PATH):
        with open(ALT_FEATURES_PATH, "r") as f:
            model_feature_list = [line.strip() for line in f if line.strip()]

    # Fallback to JSON schema
    if model_feature_list is None:
        alt_json = os.path.splitext(MODEL_PATH)[0] + "_schema.json"
        if os.path.exists(alt_json):
            with open(alt_json, "r") as fj:
                schema = json.load(fj)
                model_feature_list = schema.get("feature_list")

    if model_feature_list is None:
        raise RuntimeError(
            "Unable to determine model feature list. Provide booster.feature_names, ../models/feature_names.txt, or model_schema.json."
        )

    # Remove unsafe target/time columns if present
    model_feature_list = [
        f for f in model_feature_list if f not in ("timestamp", "is_fraud")]

    return model, model_feature_list

# model_loader.py


MODEL_PATH = "../models/xgboost_tuned.pkl"
ALT_FEATURES_PATH = "../models/feature_names.txt"  # optional: one-per-line


def load_model_and_schema():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    model = joblib.load(MODEL_PATH)

    model_feature_list = None
    # Try booster feature names (XGBoost)
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            model_feature_list = list(booster.feature_names)
    except Exception:
        model_feature_list = None

    # Fallback to text file
    if model_feature_list is None and os.path.exists(ALT_FEATURES_PATH):
        with open(ALT_FEATURES_PATH, "r") as f:
            model_feature_list = [line.strip() for line in f if line.strip()]

    # Fallback to JSON schema
    if model_feature_list is None:
        alt_json = os.path.splitext(MODEL_PATH)[0] + "_schema.json"
        if os.path.exists(alt_json):
            with open(alt_json, "r") as fj:
                schema = json.load(fj)
                model_feature_list = schema.get("feature_list")

    if model_feature_list is None:
        raise RuntimeError(
            "Unable to determine model feature list. Provide booster.feature_names, ../models/feature_names.txt, or model_schema.json."
        )

    # Remove unsafe target/time columns if present
    model_feature_list = [
        f for f in model_feature_list if f not in ("timestamp", "is_fraud")]

    return model, model_feature_list
