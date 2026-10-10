"""API d'inférence : endpoints /training et /predict."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, status
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from src.config import ensure_dirs, setup_logging
from src.data.db import init_db, ping
from src.models import predict as predict_module
from src.models.predict import ModelNotFoundError
from src.models.training import train
from src.monitoring.drift import compute_drift

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    ensure_dirs()
    if ping():
        init_db()
    else:
        logger.warning("PostgreSQL injoignable au démarrage")
    yield


app = FastAPI(
    title="MLOps Phase 1 — Inference API",
    description="Entraînement et inférence du modèle baseline.",
    version="0.1.0",
    lifespan=lifespan,
)

Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)


# --------------------------------------------------------------------------- schémas
class TrainingRequest(BaseModel):
    test_size: float | None = Field(default=None, gt=0, lt=1)
    random_state: int | None = None
    limit: int | None = Field(default=None, gt=0, description="Limiter à N lignes (test rapide)")
    continuous: bool = Field(
        default=False,
        description="Mode versioning continu : traite la prochaine tranche depuis pipeline_cursor",
    )


class TrainingResponse(BaseModel):
    status: str
    model_name: str
    version: str
    trained_at: str
    n_train: int
    n_test: int
    n_features: int
    metrics: dict[str, Any]
    mlflow_run_id: str | None = None
    mlflow_model_version: str | None = None
    comparison: dict[str, Any] | None = None


class PredictRequest(BaseModel):
    records: list[dict[str, Any]] = Field(min_length=1)

    model_config = {
        "json_schema_extra": {
            "example": {
                "records": [
                    {
                        "designation": "Console de jeu portable retro",
                        "description": "400 jeux inclus, écran couleur",
                        "productid": 123456,
                        "imageid": 987654,
                        "image_path": None,
                    }
                ]
            }
        }
    }

class PredictionItem(BaseModel):
    prediction: int
    probability: float
    model_version: str


class PredictResponse(BaseModel):
    count: int
    results: list[PredictionItem]


# --------------------------------------------------------------------------- endpoints
@app.get("/health", tags=["monitoring"])
def health() -> dict[str, Any]:
    """État de l'API, de la base et du modèle."""
    try:
        predict_module.load_artifact()  # registry @champion first, local file as fallback
        model_available = True
    except Exception:  # noqa: BLE001
        model_available = False
    return {
        "status": "ok",
        "database": "up" if ping() else "down",
        "model_available": model_available,
    }


@app.post("/training", response_model=TrainingResponse, tags=["model"])
async def run_training(request: TrainingRequest) -> TrainingResponse:
    """Réentraîne le modèle baseline à partir des données en base."""
    try:
        summary = await run_in_threadpool(
            train,
            test_size=request.test_size,
            random_state=request.random_state,
            limit=request.limit,
            continuous=request.continuous,
        )
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Échec de l'entraînement")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    predict_module.load_artifact(force_reload=True)

    return TrainingResponse(status="trained", **summary)


@app.post("/predict", response_model=PredictResponse, tags=["model"])
async def run_predict(request: PredictRequest) -> PredictResponse:
    """Prédit la classe pour une ou plusieurs observations."""
    try:
        results = await run_in_threadpool(predict_module.predict, request.records)
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Échec de l'inférence")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return PredictResponse(count=len(results), results=results)


@app.get("/model", tags=["model"])
def model_metadata() -> dict[str, Any]:
    """Métadonnées et métriques du modèle actuellement servi."""
    try:
        return predict_module.model_info()
    except ModelNotFoundError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@app.post("/drift/compute", tags=["monitoring"])
async def run_drift_computation() -> dict[str, Any]:
    """Calcule le data drift entre la baseline et le dernier batch traité,
    et persiste les métriques en base pour Grafana."""
    from src.data.db import get_session
    from src.models.training import _get_cursor

    with get_session() as session:
        current_max_id = _get_cursor(session)

    if current_max_id == 0:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            detail="Aucun entraînement continu n'a encore tourné (curseur à 0).",
        )

    try:
        result = await run_in_threadpool(compute_drift, current_max_id)
    except RuntimeError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Échec du calcul de drift")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    return result
