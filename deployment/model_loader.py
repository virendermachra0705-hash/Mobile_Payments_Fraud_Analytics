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
      feature_columns: list of columns expected by the model
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

    # align booster feature names if possible
    try:
        booster = model.get_booster()
        # if model has feature names already, keep them; else assign training schema (best-effort)
        booster.feature_names = booster.feature_names if booster.feature_names is not None else feature_columns
    except Exception:
        # some models may not expose get_booster
        pass

    return model, feature_columns, training_df
