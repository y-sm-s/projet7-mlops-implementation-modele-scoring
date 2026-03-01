import pandas as pd
from .model_loader import load_artifacts

model = None
features = None
threshold = None


def get_artifacts():
    global model, features, threshold
    if model is None:
        model, features, threshold = load_artifacts()
    return model, features, threshold


def predict_client(data: dict):
    model, features, threshold = get_artifacts()

    missing = set(features) - set(data.keys())
    if missing:
        raise ValueError(f"Features manquantes : {sorted(missing)}")

    df = pd.DataFrame([{f: data[f] for f in features}])
    proba = model.predict_proba(df)[0, 1]
    decision = int(proba >= threshold)

    return {
        "decision": decision,
        "probability": float(proba),
        "threshold": float(threshold),
    }


