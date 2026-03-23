"""
Module : main.py (API)
Auteur : William BLENON & Véronèse Nikina ZINSOU
Date   : 2026
Projet : Maintenance Prédictive Industrielle – M2 Data Engineering EFREI

Description :
    API REST développée avec FastAPI exposant le modèle de maintenance
    prédictive comme un service d'inférence autonome.

    Dans un environnement professionnel réel, le modèle n'est jamais
    appelé directement depuis un notebook ou un dashboard. Il est exposé
    via une API REST qui joue le rôle de couche d'abstraction entre :
        - Les clients (dashboard, ERP, application mobile, SCADA),
        - Le modèle entraîné sérialisé sur disque.

    Architecture :
        Client → POST /predict → Pipeline preprocessing → Modèle → Réponse JSON

    Endpoints disponibles :
        GET  /health       : Vérification de l'état du service.
        GET  /model-info   : Informations sur le modèle chargé.
        POST /predict      : Prédiction à partir des données capteurs.
        POST /predict/batch: Prédiction sur plusieurs observations.

Usage :
    $ uvicorn api.main:app --reload --port 8000

    Documentation interactive (Swagger) :
        http://localhost:8000/docs

    Documentation alternative (ReDoc) :
        http://localhost:8000/redoc
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Ajout du répertoire racine au path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.data_preprocessor import DataPreprocessor
from src.data.data_loader import DataLoader

# ─────────────────────────────────────────────────────────────────
# Constantes de configuration
# ─────────────────────────────────────────────────────────────────

API_TITLE = "Maintenance Prédictive – API d'Inférence"
API_VERSION = "1.0.0"
API_DESCRIPTION = """
## Système Intelligent de Maintenance Prédictive Industrielle

Cette API expose un modèle de classification binaire entraîné pour détecter
les risques de panne industrielle dans les **24 prochaines heures**.

### Fonctionnalités
- **Prédiction unitaire** : `/predict` – une observation à la fois
- **Prédiction batch** : `/predict/batch` – plusieurs observations simultanées
- **Supervision** : `/health` et `/model-info`

### Variables d'entrée attendues
Les données doivent correspondre aux capteurs industriels :
`vibration_rms`, `temperature_motor`, `rpm`, `pressure_level`, `operating_mode`

### Format de réponse
Chaque prédiction retourne :
- `prediction` (int) : 0 = pas de panne, 1 = panne probable
- `probability` (float) : probabilité de panne entre 0 et 1
- `risk_level` (str) : LOW / MEDIUM / HIGH
- `recommendation` (str) : action recommandée
"""

# Chemin vers le modèle sérialisé
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "industrial_machine_maintenance.csv")

# Seuil de décision par défaut
DEFAULT_THRESHOLD = 0.5

# ─────────────────────────────────────────────────────────────────
# État global de l'application (chargé au démarrage)
# ─────────────────────────────────────────────────────────────────

app_state: dict[str, Any] = {
    "model": None,
    "preprocessor": None,
    "model_name": None,
    "model_path": None,
    "loaded_at": None,
    "prediction_count": 0,
}


# ─────────────────────────────────────────────────────────────────
# Lifecycle : chargement au démarrage
# ─────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionnaire de cycle de vie FastAPI.

    Exécuté au démarrage : charge le modèle et le pipeline de preprocessing.
    Exécuté à l'arrêt : libère les ressources.

    Le modèle est chargé une seule fois en mémoire au démarrage du serveur,
    évitant un rechargement coûteux à chaque requête.
    """
    print("[API] Démarrage du service d'inférence...")

    # Recherche du meilleur modèle disponible dans models/
    model_files = []
    if os.path.exists(MODEL_DIR):
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pkl")]

    if not model_files:
        print(
            "[API] ⚠️  Aucun modèle trouvé dans models/. "
            "Exécutez main.py pour entraîner et sauvegarder un modèle."
        )
    else:
        # Charge le premier modèle disponible (le plus récent)
        model_files.sort(key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)),
                         reverse=True)
        model_path = os.path.join(MODEL_DIR, model_files[0])

        try:
            app_state["model"] = joblib.load(model_path)
            app_state["model_name"] = model_files[0].replace(".pkl", "").replace("_", " ").title()
            app_state["model_path"] = model_path
            app_state["loaded_at"] = datetime.now().isoformat()
            print(f"[API] ✅ Modèle chargé : {model_files[0]}")
        except Exception as e:
            print(f"[API] ❌ Erreur de chargement du modèle : {e}")

    # Chargement et ajustement du pipeline de preprocessing
    try:
        loader = DataLoader(DATA_PATH)
        df = loader.load()
        preprocessor = DataPreprocessor(target="failure_within_24h")
        preprocessor.fit_transform(df)
        app_state["preprocessor"] = preprocessor
        print("[API] ✅ Pipeline de preprocessing prêt.")
    except Exception as e:
        print(f"[API] ❌ Erreur de chargement du preprocessing : {e}")

    yield  # L'application est prête à recevoir des requêtes

    # Nettoyage à l'arrêt
    print("[API] Arrêt du service. Libération des ressources.")
    app_state.clear()


# ─────────────────────────────────────────────────────────────────
# Instanciation de l'application FastAPI
# ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Middleware CORS : autorise les requêtes depuis le dashboard Streamlit
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # En production : restreindre aux domaines autorisés
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────
# Schémas Pydantic – Validation des entrées/sorties
# ─────────────────────────────────────────────────────────────────

class SensorData(BaseModel):
    """
    Schéma de validation pour les données capteurs d'une machine industrielle.

    Correspond aux 14 features réelles du dataset après preprocessing :
    7 numériques + 2 catégorielles (machine_type, operating_mode).

    Attributes:
        vibration_rms (float)            : Amplitude vibration RMS (g).
        temperature_motor (float)        : Température moteur (°C).
        current_phase_avg (float)        : Courant de phase moyen (A).
        pressure_level (float)           : Niveau de pression (bar).
        rpm (float)                      : Vitesse de rotation (tr/min).
        hours_since_maintenance (float)  : Heures depuis dernière maintenance.
        ambient_temp (float)             : Température ambiante (°C).
        machine_type (str)               : Type de machine (CNC, Pump, Compressor...).
        operating_mode (str)             : Mode de fonctionnement.
        threshold (float)                : Seuil de décision. Par défaut : 0.5.
    """
    # ── Capteurs numériques ───────────────────────────────────
    vibration_rms: float = Field(
        ..., ge=0.0, le=50.0,
        description="Amplitude de vibration RMS (g).",
        example=2.5,
    )
    temperature_motor: float = Field(
        ..., ge=0.0, le=300.0,
        description="Température du moteur (°C).",
        example=75.0,
    )
    current_phase_avg: float = Field(
        ..., ge=0.0, le=100.0,
        description="Courant de phase moyen (A).",
        example=6.5,
    )
    pressure_level: float = Field(
        ..., ge=0.0, le=300.0,
        description="Niveau de pression (bar).",
        example=45.0,
    )
    rpm: float = Field(
        ..., ge=0.0, le=10000.0,
        description="Vitesse de rotation (tr/min).",
        example=1200.0,
    )
    hours_since_maintenance: float = Field(
        ..., ge=0.0, le=10000.0,
        description="Heures écoulées depuis la dernière maintenance.",
        example=120.0,
    )
    ambient_temp: float = Field(
        ..., ge=-20.0, le=60.0,
        description="Température ambiante (°C).",
        example=22.0,
    )
    # ── Variables catégorielles ───────────────────────────────
    machine_type: str = Field(
        ...,
        description="Type de machine : CNC, Pump, Compressor, etc.",
        example="CNC",
    )
    operating_mode: str = Field(
        ...,
        description="Mode de fonctionnement : normal, idle, peak, degraded, etc.",
        example="normal",
    )
    # ── Paramètre de décision ─────────────────────────────────
    threshold: float = Field(
        default=DEFAULT_THRESHOLD,
        ge=0.0, le=1.0,
        description="Seuil de décision. Valeur < 0.5 augmente le Recall.",
        example=0.5,
    )

    @field_validator("operating_mode", "machine_type")
    @classmethod
    def normalize_strings(cls, v: str) -> str:
        """Normalise les chaînes en minuscules sans espaces superflus."""
        return v.strip().lower()


class BatchSensorData(BaseModel):
    """
    Schéma pour la prédiction par lot (plusieurs machines simultanément).

    Attributes:
        observations (list[SensorData]): Liste des observations à prédire.
                                          Maximum 100 observations par requête.
        threshold (float): Seuil de décision global pour toutes les observations.
    """
    observations: list[SensorData] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="Liste des observations capteurs. Maximum 100 par requête.",
    )


class PredictionResponse(BaseModel):
    """
    Schéma de réponse pour une prédiction unitaire.

    Attributes:
        prediction (int): Classe prédite : 0 = pas de panne, 1 = panne probable.
        probability (float): Probabilité de panne (classe 1), entre 0 et 1.
        risk_level (str): Niveau de risque : LOW / MEDIUM / HIGH.
        recommendation (str): Action recommandée selon le niveau de risque.
        model_name (str): Nom du modèle ayant effectué la prédiction.
        threshold_used (float): Seuil de décision utilisé.
        timestamp (str): Horodatage ISO 8601 de la prédiction.
    """
    prediction: int
    probability: float
    risk_level: str
    recommendation: str
    model_name: str
    threshold_used: float
    timestamp: str


class BatchPredictionResponse(BaseModel):
    """
    Schéma de réponse pour une prédiction par lot.

    Attributes:
        predictions (list[PredictionResponse]): Prédictions pour chaque observation.
        summary (dict): Synthèse : nombre total, machines à risque, taux de panne.
        processing_time_ms (float): Temps de traitement total en millisecondes.
    """
    predictions: list[PredictionResponse]
    summary: dict[str, Any]
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Schéma de réponse pour le endpoint /health."""
    status: str
    model_loaded: bool
    preprocessor_loaded: bool
    model_name: str | None
    loaded_at: str | None
    prediction_count: int
    uptime_info: str


class ModelInfoResponse(BaseModel):
    """Schéma de réponse pour le endpoint /model-info."""
    model_name: str | None
    model_path: str | None
    model_type: str | None
    loaded_at: str | None
    target_variable: str
    threshold_default: float
    input_features: list[str]
    api_version: str


# ─────────────────────────────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────────────────────────────

def _get_risk_level(probability: float) -> str:
    """
    Détermine le niveau de risque à partir de la probabilité de panne.

    Args:
        probability (float): Probabilité de panne entre 0 et 1.

    Returns:
        str: "LOW" si < 30%, "MEDIUM" si 30–60%, "HIGH" si > 60%.
    """
    if probability < 0.30:
        return "LOW"
    elif probability < 0.60:
        return "MEDIUM"
    else:
        return "HIGH"


def _get_recommendation(risk_level: str) -> str:
    """
    Retourne une recommandation d'action selon le niveau de risque.

    Args:
        risk_level (str): "LOW", "MEDIUM" ou "HIGH".

    Returns:
        str: Action recommandée au responsable maintenance.
    """
    recommendations = {
        "LOW": (
            "Machine opérationnelle. Aucune intervention immédiate requise. "
            "Continuer la surveillance selon le calendrier standard."
        ),
        "MEDIUM": (
            "Risque modéré détecté. Planifier une inspection préventive "
            "dans les 48 prochaines heures. Surveiller l'évolution des capteurs."
        ),
        "HIGH": (
            "RISQUE ÉLEVÉ DE PANNE. Intervention immédiate recommandée. "
            "Vérifier en priorité : vibration, température moteur, pression. "
            "Envisager un arrêt préventif si la situation se dégrade."
        ),
    }
    return recommendations.get(risk_level, "Niveau de risque inconnu.")


def _check_service_ready() -> None:
    """
    Vérifie que le modèle et le preprocessor sont bien chargés.

    Raises:
        HTTPException 503: Si le service n'est pas prêt à traiter les requêtes.
    """
    if app_state["model"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Modèle non disponible.",
                "solution": "Exécutez main.py pour entraîner et sauvegarder un modèle, "
                            "puis redémarrez l'API.",
            },
        )
    if app_state["preprocessor"] is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "error": "Pipeline de preprocessing non disponible.",
                "solution": "Vérifiez que le fichier CSV est présent dans data/.",
            },
        )


def _run_inference(sensor_data: SensorData) -> PredictionResponse:
    """
    Exécute le pipeline complet d'inférence pour une observation.

    Étapes :
        1. Construction du DataFrame depuis les données capteurs,
        2. Transformation via le pipeline ajusté,
        3. Prédiction de probabilité,
        4. Application du seuil de décision,
        5. Construction de la réponse.

    Args:
        sensor_data (SensorData): Données capteurs validées.

    Returns:
        PredictionResponse: Réponse complète avec prédiction et métadonnées.

    Raises:
        HTTPException 500: En cas d'erreur interne lors de l'inférence.
    """
    try:
        # Construction du DataFrame d'entrée avec tous les champs du dataset
        input_dict = {
            "vibration_rms":           [sensor_data.vibration_rms],
            "temperature_motor":       [sensor_data.temperature_motor],
            "current_phase_avg":       [sensor_data.current_phase_avg],
            "pressure_level":          [sensor_data.pressure_level],
            "rpm":                     [sensor_data.rpm],
            "hours_since_maintenance": [sensor_data.hours_since_maintenance],
            "ambient_temp":            [sensor_data.ambient_temp],
            "machine_type":            [sensor_data.machine_type],
            "operating_mode":          [sensor_data.operating_mode],
        }
        input_df = pd.DataFrame(input_dict)

        # Transformation via le pipeline ajusté sur le dataset complet
        preprocessor: DataPreprocessor = app_state["preprocessor"]
        X_input = preprocessor.transform(input_df)

        # Prédiction de probabilité
        model = app_state["model"]
        proba = float(model.predict_proba(X_input)[0][1])

        # Application du seuil
        prediction = int(proba >= sensor_data.threshold)
        risk_level = _get_risk_level(proba)
        recommendation = _get_recommendation(risk_level)

        # Compteur de prédictions
        app_state["prediction_count"] += 1

        return PredictionResponse(
            prediction=prediction,
            probability=round(proba, 4),
            risk_level=risk_level,
            recommendation=recommendation,
            model_name=app_state["model_name"] or "unknown",
            threshold_used=sensor_data.threshold,
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "Erreur interne lors de l'inférence.",
                "details": str(e),
            },
        )


# ─────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Vérification de l'état du service",
    tags=["Supervision"],
)
async def health_check() -> HealthResponse:
    """
    Vérifie que l'API est opérationnelle et que le modèle est bien chargé.

    Utilisé par les systèmes de monitoring (Kubernetes liveness probe,
    load balancer, dashboard Streamlit) pour vérifier la disponibilité.

    Returns:
        HealthResponse: État du service avec informations de chargement.

    Example:
        ```bash
        curl http://localhost:8000/health
        ```
    """
    model_loaded = app_state["model"] is not None
    preprocessor_loaded = app_state["preprocessor"] is not None
    service_status = "healthy" if (model_loaded and preprocessor_loaded) else "degraded"

    return HealthResponse(
        status=service_status,
        model_loaded=model_loaded,
        preprocessor_loaded=preprocessor_loaded,
        model_name=app_state["model_name"],
        loaded_at=app_state["loaded_at"],
        prediction_count=app_state["prediction_count"],
        uptime_info=f"Service démarré · {app_state['prediction_count']} prédictions effectuées",
    )


@app.get(
    "/model-info",
    response_model=ModelInfoResponse,
    summary="Informations sur le modèle chargé",
    tags=["Supervision"],
)
async def model_info() -> ModelInfoResponse:
    """
    Retourne les métadonnées du modèle actuellement chargé en mémoire.

    Utile pour vérifier quelle version du modèle est en production,
    quels features sont attendus en entrée, et quel seuil est appliqué.

    Returns:
        ModelInfoResponse: Informations complètes sur le modèle actif.

    Example:
        ```bash
        curl http://localhost:8000/model-info
        ```
    """
    model_type = None
    if app_state["model"] is not None:
        model_type = type(app_state["model"]).__name__

    feature_names = []
    if app_state["preprocessor"] is not None:
        feature_names = app_state["preprocessor"].feature_names or []

    return ModelInfoResponse(
        model_name=app_state["model_name"],
        model_path=app_state["model_path"],
        model_type=model_type,
        loaded_at=app_state["loaded_at"],
        target_variable="failure_within_24h",
        threshold_default=DEFAULT_THRESHOLD,
        input_features=feature_names,
        api_version=API_VERSION,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédiction de panne pour une machine",
    tags=["Inférence"],
)
async def predict(sensor_data: SensorData) -> PredictionResponse:
    """
    Prédit la probabilité de panne dans les 24 prochaines heures
    à partir des données capteurs d'une machine industrielle.

    Le pipeline complet est appliqué :
    1. Validation des données (Pydantic),
    2. Preprocessing (StandardScaler + OneHotEncoder),
    3. Inférence via le modèle chargé,
    4. Calcul du niveau de risque et de la recommandation.

    Args:
        sensor_data (SensorData): Données capteurs validées.

    Returns:
        PredictionResponse: Prédiction avec probabilité, niveau de risque
                            et recommandation d'action.

    Raises:
        HTTPException 422: Si les données d'entrée sont invalides.
        HTTPException 503: Si le service n'est pas prêt.
        HTTPException 500: En cas d'erreur interne.

    Example:
        ```bash
        curl -X POST http://localhost:8000/predict \\
          -H "Content-Type: application/json" \\
          -d '{
            "vibration_rms": 8.5,
            "temperature_motor": 145.0,
            "rpm": 4200.0,
            "pressure_level": 12.5,
            "operating_mode": "degraded",
            "threshold": 0.4
          }'
        ```
    """
    _check_service_ready()
    return _run_inference(sensor_data)


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    summary="Prédiction de panne pour plusieurs machines simultanément",
    tags=["Inférence"],
)
async def predict_batch(batch_data: BatchSensorData) -> BatchPredictionResponse:
    """
    Prédit les probabilités de panne pour un lot de machines simultanément.

    Optimisé pour le monitoring en temps réel d'un parc de machines :
    une seule requête permet d'évaluer jusqu'à 100 machines.

    Args:
        batch_data (BatchSensorData): Lot d'observations capteurs.

    Returns:
        BatchPredictionResponse: Prédictions individuelles + synthèse du parc.

    Example:
        ```bash
        curl -X POST http://localhost:8000/predict/batch \\
          -H "Content-Type: application/json" \\
          -d '{
            "observations": [
              {"vibration_rms": 2.1, "temperature_motor": 65.0,
               "rpm": 2800.0, "pressure_level": 5.5, "operating_mode": "normal"},
              {"vibration_rms": 9.8, "temperature_motor": 160.0,
               "rpm": 5500.0, "pressure_level": 18.0, "operating_mode": "degraded"}
            ]
          }'
        ```
    """
    _check_service_ready()

    start_time = time.time()
    predictions = []

    for obs in batch_data.observations:
        pred = _run_inference(obs)
        predictions.append(pred)

    processing_time = round((time.time() - start_time) * 1000, 2)

    # Synthèse du parc
    n_total = len(predictions)
    n_at_risk = sum(1 for p in predictions if p.prediction == 1)
    n_high = sum(1 for p in predictions if p.risk_level == "HIGH")
    n_medium = sum(1 for p in predictions if p.risk_level == "MEDIUM")

    summary = {
        "total_machines": n_total,
        "machines_at_risk": n_at_risk,
        "failure_rate_percent": round(n_at_risk / n_total * 100, 1),
        "high_risk": n_high,
        "medium_risk": n_medium,
        "low_risk": n_total - n_high - n_medium,
        "avg_probability": round(
            sum(p.probability for p in predictions) / n_total, 4
        ),
    }

    return BatchPredictionResponse(
        predictions=predictions,
        summary=summary,
        processing_time_ms=processing_time,
    )


# ─────────────────────────────────────────────────────────────────
# Gestionnaire d'erreurs global
# ─────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Gestionnaire d'erreurs global : capture toute exception non gérée.

    Retourne une réponse JSON structurée plutôt qu'une stacktrace Python brute,
    ce qui est essentiel en production pour ne pas exposer des informations
    sensibles sur l'architecture interne.

    Args:
        request (Request): La requête HTTP ayant déclenché l'erreur.
        exc (Exception): L'exception levée.

    Returns:
        JSONResponse: Réponse d'erreur structurée avec code HTTP 500.
    """
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Erreur interne du serveur.",
            "details": str(exc),
            "path": str(request.url),
            "timestamp": datetime.now().isoformat(),
        },
    )


# ─────────────────────────────────────────────────────────────────
# Point d'entrée
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,       # Rechargement automatique en développement
        log_level="info",
    )