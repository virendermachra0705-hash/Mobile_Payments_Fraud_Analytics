# model_loader.py
import pandas as pd
import pickle
import os

MODEL_PATH = "../models/xgb_fraud_model.pkl"
SCHEMA_PATH = "../models/feature_list.pkl"
TRAINING_DATA_PATH = "../data/processed/features_enriched_v2.csv"


def load_model_and_schema():
    # Load model
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Load feature list
    with open(SCHEMA_PATH, "rb") as f:
        feature_list = pickle.load(f)

    # Load training DF (used for fallback medians)
    if os.path.exists(TRAINING_DATA_PATH):
        training_df = pd.read_csv(TRAINING_DATA_PATH)
    else:
        training_df = pd.DataFrame()  # Safe fallback

    return model, feature_list, training_df
