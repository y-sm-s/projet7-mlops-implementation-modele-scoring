import sys
from pathlib import Path

# Ajoute le répertoire racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import api.predict
from api.app import app

client = TestClient(app)


@pytest.fixture
def mock_model_artifacts():
    """
    Remplace get_artifacts par un mock et réinitialise le cache.
    """
    with patch("api.predict.get_artifacts") as mock_get_artifacts:
        # Crée un faux modèle avec une méthode predict_proba contrôlée
        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

        # Valeur de retour du mock
        mock_get_artifacts.return_value = (mock_model, ["feature1", "feature2"], 0.6)

        # Réinitialise les variables globales pour éviter que le vrai modèle ne soit chargé
        api.predict.model = None
        api.predict.features = None
        api.predict.threshold = None

        yield  # Le test s'exécute ici


def test_health():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"


def test_predict_valid(mock_model_artifacts):
    payload = {"data": {"feature1": 10, "feature2": 20}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == 1
    assert data["probability"] == 0.7
    assert data["threshold"] == 0.6


def test_predict_missing_feature(mock_model_artifacts):
    payload = {"data": {"feature1": 10}}  # feature2 manquante
    response = client.post("/predict", json=payload)
    assert response.status_code == 400
    assert "Features manquantes" in response.json()["detail"]


def test_predict_wrong_structure():
    payload = {"wrong_key": {"feature1": 10, "feature2": 20}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation Pydantic


def test_predict_empty_data(mock_model_artifacts):
    payload = {"data": {}}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400


def test_predict_internal_error():
    with patch("api.app.predict_client", side_effect=Exception("Modèle cassé")):
        payload = {"data": {"feature1": 10, "feature2": 20}}
        response = client.post("/predict", json=payload)
        assert response.status_code == 500
        assert "Erreur interne" in response.json()["detail"]


def test_predict_value_error(mock_model_artifacts):
    # Ce cas est déjà couvert par test_predict_missing_feature
    pass
