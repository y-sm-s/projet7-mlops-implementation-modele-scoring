import pandas as pd
from .model_loader import load_artifacts

<<<<<<< HEAD
# Charger les artifacts au démarrage (une seule fois)
=======
>>>>>>> eb03a00 (Ajout des dossiers api, model, data et streamlit_app)
model, features, threshold = load_artifacts()


def predict_client(data: dict):
<<<<<<< HEAD
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

=======
    # Vérifier features
    missing = set(features) - set(data.keys())
    if missing:
        raise ValueError(f"Features manquantes : {missing}")
    # Créer DataFrame avec bon ordre
    df = pd.DataFrame([{f: data[f] for f in features}])
    proba = model.predict_proba(df)[0, 1]
    decision = int(proba >= threshold)
>>>>>>> eb03a00 (Ajout des dossiers api, model, data et streamlit_app)
    return {
        "decision": decision,
        "probability": float(proba),
        "threshold": float(threshold),
    }
