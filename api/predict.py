import pandas as pd
from .model_loader import load_artifacts

# Lazy loading : le modèle n'est chargé qu'au premier appel
model = None
features = None
threshold = None


def get_artifacts():
    """Charge les artifacts une seule fois, au premier appel."""
    global model, features, threshold

    if model is None:
        model, features, threshold = load_artifacts()

    return model, features, threshold


def predict_client(data: dict):
    """
    Prédit le risque de défaut pour un client.

    Args:
        data: Dictionnaire avec les features du client

    Returns:
        Dict avec decision, probability, threshold

    Raises:
        ValueError: Si des features sont manquantes
    """

    model, features, threshold = get_artifacts()

    # Vérifier les features manquantes
    missing = set(features) - set(data.keys())
    if missing:
        raise ValueError(f"Features manquantes : {sorted(missing)}")

    # Créer DataFrame avec le bon ordre des features
    df = pd.DataFrame([{f: data[f] for f in features}])

    # Prédiction
    proba = model.predict_proba(df)[0, 1]
    decision = int(proba >= threshold)

    return {
        "decision": decision,
        "probability": float(proba),
        "threshold": float(threshold),
    }
