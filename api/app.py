from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from .predict import predict_client

app = FastAPI(
    title="Credit Risk API - Projet 7",
    description="API de scoring crédit avec fonction de coût métier",
    version="1.0.0",
)


class ClientInput(BaseModel):
    data: Dict[str, Any]


@app.get("/")
def health():
    """Health check endpoint"""
    return {"status": "OK", "message": "Credit Risk API is running", "version": "1.0.0"}


@app.post("/predict")
def predict(payload: ClientInput):
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
        raise HTTPException(status_code=400, detail=f"Erreur de validation : {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur interne : {str(e)}")
