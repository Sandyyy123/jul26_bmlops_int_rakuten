"""Inférence à partir du modèle entraîné.

Usage :
    python -m src.models.predict --sample
    python -m src.models.predict --input payload.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.tracking import MlflowClient
from src.config import settings, setup_logging
from src.data.db import Prediction, get_session
from src.features.build_features import load_dataframe, prepare_frame

logger = logging.getLogger(__name__)

_ARTIFACT: dict[str, Any] | None = None

FLAG_THRESHOLD = 0.6  # en dessous de ce score de confiance, la prédiction est flaggée


class ModelNotFoundError(RuntimeError):
    """Aucun modèle entraîné n'est disponible."""


CHAMPION_ALIAS = "champion"  # must match src/models/training.py
# Behind nginx there are several API copies; only the copy that ran /training reloads at once.
# Every copy re-checks the @champion alias at most this often (seconds) and reloads if it moved.
CHAMPION_REFRESH_SECONDS = float(os.getenv("CHAMPION_REFRESH_SECONDS", "30"))
_LAST_CHECK = 0.0


def _load_from_registry() -> dict[str, Any]:
    """Load the model that the @champion alias points to in the MLflow registry."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    name = settings.mlflow_registered_model_name
    model_version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    pipeline = mlflow.sklearn.load_model(f"models:/{name}@{CHAMPION_ALIAS}")
    run = client.get_run(model_version.run_id)
    return {
        "pipeline": pipeline,
        "feature_names": list(pipeline.feature_names_in_),
        "model_name": settings.model_name,
        "version": f"registry-v{model_version.version}",
        "trained_at": datetime.fromtimestamp(run.info.start_time / 1000, tz=UTC).isoformat(),
        "metrics": dict(run.data.metrics),
        "mlflow_model_version": str(model_version.version),
        "source": "registry",
    }


def _champion_changed() -> bool:
    """True if @champion now points to another version than the model in memory."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    current = MlflowClient().get_model_version_by_alias(
        settings.mlflow_registered_model_name, CHAMPION_ALIAS
    )
    return str(current.version) != _ARTIFACT.get("mlflow_model_version")


def load_artifact(force_reload: bool = False) -> dict[str, Any]:
    """Load the served model: the registry @champion first, the local file as fallback."""
    global _ARTIFACT, _LAST_CHECK

    if _ARTIFACT is not None and not force_reload:
        if time.monotonic() - _LAST_CHECK < CHAMPION_REFRESH_SECONDS:
            return _ARTIFACT
        _LAST_CHECK = time.monotonic()
        try:
            if not _champion_changed():
                return _ARTIFACT
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Could not check @%s (%s); keeping the model in memory", CHAMPION_ALIAS, exc
            )
            return _ARTIFACT
        logger.info("@%s moved: reloading the model", CHAMPION_ALIAS)

    _LAST_CHECK = time.monotonic()
    try:
        _ARTIFACT = _load_from_registry()
        logger.info("Model loaded from the registry (%s)", _ARTIFACT["version"])
        return _ARTIFACT
    except Exception as exc:  # noqa: BLE001
        logger.warning("Registry @%s unavailable (%s); using the local file", CHAMPION_ALIAS, exc)

    if not settings.model_path.exists():
        raise ModelNotFoundError(
            f"Modèle introuvable ({settings.model_path}). "
            "Lancez d'abord : python -m src.models.training"
        )

    _ARTIFACT = joblib.load(settings.model_path)
    _ARTIFACT.setdefault("source", "local")
    logger.info("Modèle chargé (version %s)", _ARTIFACT["version"])
    return _ARTIFACT


def model_info() -> dict[str, Any]:
    """Métadonnées du modèle courant, sans les objets sklearn."""
    artifact = load_artifact()
    return {
        "model_name": artifact["model_name"],
        "version": artifact["version"],
        "trained_at": artifact["trained_at"],
        "n_features": len(artifact["feature_names"]),
        "metrics": artifact["metrics"],
        "source": artifact.get("source", "local"),
        "mlflow_model_version": artifact.get("mlflow_model_version"),
    }


def _log_batch_to_mlflow(results: list[dict[str, Any]]) -> None:
    """Journalise un résumé du batch de prédictions dans MLflow (monitoring)."""
    n_total = len(results)
    n_flagged = sum(1 for r in results if r["flagged"])
    flag_rate = (n_flagged / n_total) if n_total else 0.0

    try:
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name="prediction_batch"):
            mlflow.set_tags(
                {
                    "run_type": "prediction_batch",
                    "has_flagged_predictions": "true" if n_flagged > 0 else "false",
                    "model_version": results[0]["model_version"] if results else "unknown",
                }
            )
            mlflow.log_metrics(
                {
                    "n_predictions": n_total,
                    "n_flagged": n_flagged,
                    "flag_rate": flag_rate,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Journalisation MLflow du batch impossible : %s", exc)


def predict(records: list[dict], log_to_db: bool = True, log_to_mlflow: bool = True) -> list[dict[str, Any]]:
    """Prédit pour une liste d'observations et journalise le résultat."""
    if not records:
        raise ValueError("Aucune observation fournie")

    artifact = load_artifact()
    pipeline = artifact["pipeline"]
    feature_names: list[str] = artifact["feature_names"]

    X = prepare_frame(records, feature_names)
    labels = pipeline.predict(X)
    probabilities_matrix = pipeline.predict_proba(X)

    results = []
    for label, proba_row in zip(labels, probabilities_matrix, strict=True):
        probability = round(float(max(proba_row)), 6)
        flagged = probability < FLAG_THRESHOLD
        results.append(
            {
                "prediction": int(label),
                "probability": probability,
                "model_version": artifact["version"],
                "flagged": flagged,
                "flag_reason": "low_confidence" if flagged else None,
            }
        )

    n_flagged = sum(1 for r in results if r["flagged"])
    if n_flagged:
        logger.info("%d/%d prédiction(s) flaggée(s) (confiance < %.2f)", n_flagged, len(results), FLAG_THRESHOLD)

    if log_to_db:
        try:
            with get_session() as session:
                session.add_all(
                    [
                        Prediction(
                            model_version=artifact["version"],
                            features=record,
                            prediction=result["prediction"],
                            probability=result["probability"],
                            flagged=result["flagged"],
                            flag_reason=result["flag_reason"],
                        )
                        for record, result in zip(records, results, strict=True)
                    ]
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Journalisation des prédictions impossible : %s", exc)

    if log_to_mlflow:
        _log_batch_to_mlflow(results)

    return results

def sample_record() -> dict[str, Any]:
    """Construit une observation d'exemple à partir de la première ligne en base."""
    artifact = load_artifact()
    df = load_dataframe(limit=1)
    row = df[artifact["feature_names"]].iloc[0]
    # Pas de cast global en float : designation/description/image_path sont du texte.
    return row.where(pd.notna(row), None).to_dict()

def main() -> int:
    parser = argparse.ArgumentParser(description="Inférence du modèle baseline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=Path, help="Fichier JSON (objet ou liste d'objets)")
    group.add_argument("--sample", action="store_true", help="Utilise une ligne de la base")
    parser.add_argument("--no-log", action="store_true", help="Ne journalise pas en base")
    parser.add_argument("--no-mlflow", action="store_true", help="Ne journalise pas dans MLflow")
    args = parser.parse_args()

    setup_logging()

    if args.sample:
        records = [sample_record()]
    else:
        payload = json.loads(args.input.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else [payload]

    results = predict(records, log_to_db=not args.no_log, log_to_mlflow=not args.no_mlflow)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
