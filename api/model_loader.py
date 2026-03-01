# api/model_loader.py
from pathlib import Path
import joblib
import json

# Dossier api/
BASE_DIR = Path(__file__).resolve().parent

# On remonte d’un niveau vers la racine du repo
ROOT_DIR = BASE_DIR.parent

MODEL_PATH = ROOT_DIR / "model" / "final_model.pkl"
FEATURES_PATH = ROOT_DIR / "model" / "features_list.txt"
METADATA_PATH = ROOT_DIR / "model" / "metadata.json"


def load_artifacts():
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH) as f:
        features = [line.strip() for line in f if line.strip()]

    with open(METADATA_PATH) as f:
        metadata = json.load(f)

    return model, features, metadata["optimal_threshold"]


print("MODEL_PATH:", MODEL_PATH)
