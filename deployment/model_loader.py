import joblib
from xgboost import XGBClassifier


def load_model(model_path: str):
    print("Loading model from:", model_path)
    model = joblib.load(model_path)
    booster = model.get_booster()
    booster.feature_names = model.get_booster().feature_names
    print("Model loaded successfully with",
          len(booster.feature_names), "features.")
    return model, booster
