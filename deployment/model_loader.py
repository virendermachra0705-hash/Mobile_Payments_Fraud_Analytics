# model_loader.py
import joblib
import os
import json

MODEL_PATH = "../models/xgboost_tuned.pkl"
ALT_FEATURES_PATH = "../models/feature_names.txt"  # optional: one feature per line

def load_model_and_schema():
    """
    Load serialized model and extract its expected feature list.
    Returns: model, feature_list
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please place your trained model there.")

    model = joblib.load(MODEL_PATH)

    # Try to get feature names from booster (XGBoost)
    model_feature_list = None
    try:
        booster = model.get_booster()
        if getattr(booster, "feature_names", None):
            model_feature_list = list(booster.feature_names)
    except Exception:
        # some wrappers may not have get_booster; continue to fallback
        model_feature_list = None

    # Fallback: try to read feature names file
    if model_feature_list is None:
        if os.path.exists(ALT_FEATURES_PATH):
            with open(ALT_FEATURES_PATH, "r") as f:
                model_feature_list = [line.strip() for line in f if line.strip()]
        else:
            # very last resort: try to read schema stored as json next to model
            alt_json = os.path.splitext(MODEL_PATH)[0] + "_schema.json"
            if os.path.exists(alt_json):
                with open(alt_json, "r") as fj:
                    model_feature_list = json.load(fj).get("feature_list")

    if model_feature_list is None:
        raise RuntimeError(
            "Unable to determine model feature list. "
            "Ensure the XGBoost model was saved with booster.feature_names set or "
            "create ../models/feature_names.txt (one feature per line) or save schema JSON."
        )

    # remove any accidental target/timestamp if present
    cleaned = [f for f in model_feature_list if f not in ("timestamp", "is_fraud")]
    return model, cleaned
