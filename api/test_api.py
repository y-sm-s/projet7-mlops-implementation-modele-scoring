"""
Tests unitaires pour l'API Credit Risk - Projet 7
Exécution : python -m unittest test_api.py -v
"""

import sys
import os
import unittest
import json

# Ajouter le dossier parent au path
sys.path.insert(0, os.path.dirname(__file__))

from api.predict import predict_client


class TestCreditAPI(unittest.TestCase):
    """Tests de l'API de prédiction de risque crédit"""

    @classmethod
    def setUpClass(cls):
        """Charge un exemple de client pour les tests"""
        try:
            import pandas as pd

            # Essayer de charger depuis sample_clients.csv
            df = pd.read_csv("sample_clients.csv")
            cls.sample_client = df.iloc[0].to_dict()
            print(f"✅ Client chargé depuis sample_clients.csv")
        except:
            # Fallback : données réalistes minimales
            cls.sample_client = {
                "AMT_CREDIT": 450000.0,
                "DAYS_BIRTH": -12000,
                "EXT_SOURCE_2": 0.62,
                "AMT_INCOME_TOTAL": 135000.0,
                "DAYS_EMPLOYED": -1500,
                "CNT_FAM_MEMBERS": 2.0,
                "AMT_ANNUITY": 25000.0,
                "DAYS_ID_PUBLISH": -3000,
            }
            print(f"⚠️  Données fallback utilisées pour les tests")

    def test_prediction_structure(self):
        """Test 1 : La prédiction retourne la bonne structure"""
        result = predict_client(self.sample_client)

        self.assertIn("decision", result, "Le champ 'decision' doit être présent")
        self.assertIn("probability", result, "Le champ 'probability' doit être présent")
        self.assertIn("threshold", result, "Le champ 'threshold' doit être présent")
        print("✅ Test structure : OK")

    def test_probability_range(self):
        """Test 2 : La probabilité est entre 0 et 1"""
        result = predict_client(self.sample_client)

        self.assertGreaterEqual(
            result["probability"], 0.0, "Probabilité doit être >= 0"
        )
        self.assertLessEqual(result["probability"], 1.0, "Probabilité doit être <= 1")
        print(f"✅ Test probabilité : {result['probability']:.4f} ∈ [0, 1]")

    def test_decision_binary(self):
        """Test 3 : La décision est 0 ou 1"""
        result = predict_client(self.sample_client)

        self.assertIn(result["decision"], [0, 1], "Decision doit être 0 ou 1")
        print(f"✅ Test décision binaire : {result['decision']} ∈ {{0, 1}}")

    def test_decision_logic(self):
        """Test 4 : La décision respecte le seuil métier"""
        result = predict_client(self.sample_client)

        if result["probability"] >= result["threshold"]:
            self.assertEqual(
                result["decision"],
                1,
                "Si proba >= seuil, decision doit être 1 (refusé)",
            )
        else:
            self.assertEqual(
                result["decision"],
                0,
                "Si proba < seuil, decision doit être 0 (accordé)",
            )

        print(
            f"✅ Test logique : proba={result['probability']:.4f}, "
            f"seuil={result['threshold']:.4f}, decision={result['decision']}"
        )

    def test_missing_features(self):
        """Test 5 : Gestion des features manquantes"""
        incomplete_data = {"AMT_CREDIT": 100000.0}

        with self.assertRaises(ValueError) as context:
            predict_client(incomplete_data)

        error_msg = str(context.exception).lower()
        self.assertIn(
            "manquantes", error_msg, "L'erreur doit mentionner les features manquantes"
        )
        print(f"✅ Test features manquantes : ValueError correctement levée")

    def test_threshold_consistency(self):
        """Test 6 : Le seuil est cohérent"""
        result = predict_client(self.sample_client)
        threshold = result["threshold"]

        self.assertGreater(threshold, 0.0, "Seuil doit être > 0")
        self.assertLess(threshold, 1.0, "Seuil doit être < 1")
        print(f"✅ Test seuil : {threshold:.4f} ∈ (0, 1)")

    def test_data_types(self):
        """Test 7 : Les types de données retournés sont corrects"""
        result = predict_client(self.sample_client)

        self.assertIsInstance(result["decision"], int, "decision doit être int")
        self.assertIsInstance(
            result["probability"], float, "probability doit être float"
        )
        self.assertIsInstance(result["threshold"], float, "threshold doit être float")
        print("✅ Test types de données : OK")


if __name__ == "__main__":
    # Lancer les tests avec verbosité
    unittest.main(verbosity=2)
