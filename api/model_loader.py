# api/model_loader.py
import joblib
import json
import os
from pathlib import Path

# Chemin relatif au dossier racine
BASE_DIR = Path(__file__).parent.parent

MODEL_PATH = BASE_DIR / "model" / "final_model.pkl"
FEATURES_PATH = BASE_DIR / "model" / "features_list.txt"
METADATA_PATH = BASE_DIR / "model" / "metadata.json"


def load_artifacts():
    """Charge le modèle, la liste des features et le seuil métier."""
    model = joblib.load(MODEL_PATH)

    with open(FEATURES_PATH, "r") as f:
        features = [line.strip() for line in f if line.strip()]

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    threshold = metadata["optimal_threshold"]
    return model, features, threshold


# Mode CI : on ne charge pas le modèle
if os.getenv("CI") == "true":
    model = None
    features = []
    threshold = 0.5
else:
    model, features, threshold = load_artifacts()
