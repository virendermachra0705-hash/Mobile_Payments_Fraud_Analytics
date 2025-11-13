# model_loader.py
import joblib
import pandas as pd

TRAIN_FEATURES_PATH = "../data/processed/features_enriched_v2.csv"
MODEL_PATH = "../models/xgboost_tuned.pkl"


def load_model_and_schema():
    """
    Loads trained XGBoost model and the training schema (feature columns and basic stats).
    Returns:
      model: loaded sklearn XGBClassifier
      feature_columns: list of columns expected by the model (prefers booster.feature_names)
      training_df: DataFrame loaded from features_enriched_v2.csv (used to compute baseline stats)
    """
    print("Loading training schema from:", TRAIN_FEATURES_PATH)
    training_df = pd.read_csv(TRAIN_FEATURES_PATH)

    # keep a copy of training columns (exclude target if present)
    if "is_fraud" in training_df.columns:
        feature_columns = [c for c in training_df.columns if c != "is_fraud"]
    else:
        feature_columns = training_df.columns.tolist()

    print(
        f"Loaded training schema with {len(feature_columns)} feature columns.")

    print("Loading model from:", MODEL_PATH)
    model = joblib.load(MODEL_PATH)

    # If the loaded model exposes booster feature names, prefer them as the canonical schema.
    try:
        booster = model.get_booster()
        if booster.feature_names is not None:
            # Make sure we return a python list of strings
            feature_columns = list(booster.feature_names)
            print(
                f"Using booster.feature_names ({len(feature_columns)} features) as canonical schema.")
        else:
            # assign training schema to booster so booster can refer to them later
            try:
                booster.feature_names = feature_columns
                print("Assigned training schema to booster.feature_names (best-effort).")
            except Exception:
                pass
    except Exception:
        # some models may not expose get_booster
        print("Model does not support get_booster(); using training CSV schema.")

    return model, feature_columns, training_df
