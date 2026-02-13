from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from .predict import predict_client

<<<<<<< HEAD
app = FastAPI(
    title="Credit Risk API - Projet 7",
    description="API de scoring crédit avec fonction de coût métier",
    version="1.0.0",
)


class ClientInput(BaseModel):
    data: Dict[str, Any]  # ← CORRECTION : Ajouter "data:"

    class Config:
        json_schema_extra = {
            "example": {
                "data": {
                    "AMT_CREDIT": 450000.0,
                    "DAYS_BIRTH": -12000,
                    "EXT_SOURCE_2": 0.62,
                    "AMT_INCOME_TOTAL": 135000.0,
                    "DAYS_EMPLOYED": -1500,
                    "CNT_FAM_MEMBERS": 2.0,
                    "AMT_ANNUITY": 25000.0,
                    "DAYS_ID_PUBLISH": -3000,
                }
            }
        }
=======
app = FastAPI(title="Credit Risk API - Projet 7")


class ClientInput(BaseModel):
    Dict[str, Any]
>>>>>>> eb03a00 (Ajout des dossiers api, model, data et streamlit_app)


@app.get("/")
def health():
<<<<<<< HEAD
    """Health check endpoint"""
    return {"status": "OK", "message": "Credit Risk API is running", "version": "1.0.0"}
=======
    return {"status": "OK"}
>>>>>>> eb03a00 (Ajout des dossiers api, model, data et streamlit_app)


@app.post("/predict")
def predict(payload: ClientInput):
<<<<<<< HEAD
    """
    Prédit le risque de défaut de paiement pour un client

    Returns:
        - decision: 0 (crédit accordé) ou 1 (crédit refusé)
        - probability: Probabilité de défaut (0-1)
        - threshold: Seuil métier optimal
    """
    try:
        return predict_client(payload.data)
    except ValueError as e:
        # Erreur de validation (ex: features manquantes)
        raise HTTPException(status_code=400, detail=f"Erreur de validation : {str(e)}")
    except Exception as e:
        # Erreur interne
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
=======
    try:
        return predict_client(payload.data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
>>>>>>> eb03a00 (Ajout des dossiers api, model, data et streamlit_app)
