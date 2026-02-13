import pandas as pd
from .model_loader import load_artifacts


# Charger les artifacts au démarrage (une seule fois)


model, features, threshold = load_artifacts()


def predict_client(data: dict):
    """
    Prédit le risque de défaut pour un client

    Args:
        data: Dictionnaire avec les features du client

    Returns:
        Dict avec decision, probability, threshold

    Raises:
        ValueError: Si des features sont manquantes
    """
    # Vérifier les features manquantes
    missing = set(features) - set(data.keys())
    if missing:
        raise ValueError(f"Features manquantes : {sorted(missing)}")

    # Créer DataFrame avec le bon ordre des features
    df = pd.DataFrame([{f: data[f] for f in features}])

    # Prédiction
    proba = model.predict_proba(df)[0, 1]
    decision = int(proba >= threshold)

    # Vérifier features
    missing = set(features) - set(data.keys())
    if missing:
        raise ValueError(f"Features manquantes : {missing}")
    # Créer DataFrame avec bon ordre
    df = pd.DataFrame([{f: data[f] for f in features}])
    proba = model.predict_proba(df)[0, 1]
    decision = int(proba >= threshold)

    return {
        "decision": decision,
        "probability": float(proba),
        "threshold": float(threshold),
    }
