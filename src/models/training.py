"""Entraînement et évaluation du modèle baseline Rakuten, avec suivi MLflow."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
from datetime import UTC, datetime
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.config import RAW_DATA_DIR, ensure_dirs, settings, setup_logging
from src.features.build_features import (
    build_preprocessor,
    get_feature_columns,
    load_dataframe,
    split_train_test,
    split_xy,
)

logger = logging.getLogger(__name__)

CHAMPION_ALIAS = "champion"  # registry alias of the model the API serves
CHALLENGER_ALIAS = "challenger"  # registry alias of the latest rejected model
PRIMARY_METRIC = "weighted_f1"  # métrique utilisée pour départager les modèles (cf. OBJECTIVES.md)


def build_pipeline(feature_names: list[str], random_state: int) -> Pipeline:
    """Pipeline Rakuten : prétraitement texte + numérique + image + régression logistique."""
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(feature_names)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )


def evaluate(pipeline: Pipeline, X_test, y_test) -> dict[str, Any]:
    """Métriques d'évaluation multiclasse (Rakuten a ~27 classes, pas 2)."""
    y_pred = pipeline.predict(X_test)

    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "weighted_recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }


def _dataset_fingerprint() -> str:
    """Empreinte des fichiers sources bruts : trace quelle version des données a
    produit quel modèle (versioning minimal des données, sans outil dédié)."""
    hasher = hashlib.sha256()
    for filename in ("X_train_update.csv", "Y_train_CVw08PX.csv"):
        path = RAW_DATA_DIR / filename
        if path.exists():
            hasher.update(path.read_bytes())
    return hasher.hexdigest()[:12]


def _setup_mlflow() -> None:
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment_name)


def _get_champion_version(client: MlflowClient, model_name: str):
    """Return (version, metrics) of the model with the @champion alias, or (None, None).

    One-time migration: if no alias exists yet but a version is in the old
    "Production" stage, the alias is set on that version."""
    try:
        version = client.get_model_version_by_alias(model_name, CHAMPION_ALIAS)
    except Exception:
        version = None

    if version is None:
        try:
            legacy = client.get_latest_versions(model_name, stages=["Production"])
        except Exception:
            legacy = []
        if not legacy:
            return None, None
        version = legacy[0]
        client.set_registered_model_alias(model_name, CHAMPION_ALIAS, version.version)
        logger.info("Migration: @%s set on Production version %s", CHAMPION_ALIAS, version.version)

    run = client.get_run(version.run_id)
    return version.version, run.data.metrics


def _compare_and_promote(
    client: MlflowClient,
    model_name: str,
    new_version: str,
    new_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Charge la version Production actuelle, la compare au nouveau modèle sur
    PRIMARY_METRIC, et promeut/rétrograde en conséquence. Tag 'best_model' posé
    sur chaque version pour une lecture rapide sans interroger les stages."""
    prev_version, prev_metrics = _get_champion_version(client, model_name)
    new_score = new_metrics[PRIMARY_METRIC]
    prev_score = prev_metrics.get(PRIMARY_METRIC) if prev_metrics else None

    if prev_version is None:
        decision, reason, is_best = "promoted", "Aucun modèle en Production : première promotion.", True
    elif new_score > prev_score:
        decision = "promoted"
        reason = f"Nouveau {PRIMARY_METRIC}={new_score:.4f} > précédent {prev_score:.4f}."
        is_best = True
    else:
        decision = "rejected"
        reason = f"Nouveau {PRIMARY_METRIC}={new_score:.4f} <= précédent {prev_score:.4f}."
        is_best = False

    client.set_model_version_tag(model_name, new_version, "best_model", "true" if is_best else "false")
    client.set_model_version_tag(model_name, new_version, PRIMARY_METRIC, str(new_score))

    # Promotion moves the @champion alias; the API serves whatever @champion points to.
    alias = CHAMPION_ALIAS if is_best else CHALLENGER_ALIAS
    client.set_registered_model_alias(model_name, alias, new_version)

    logger.info("Comparaison MLflow : %s (version %s) — %s", decision, new_version, reason)

    return {
        "decision": decision,
        "reason": reason,
        "new_version": new_version,
        "new_score": new_score,
        "previous_version": prev_version,
        "previous_score": prev_score,
        "metric": PRIMARY_METRIC,
    }



CONTINUOUS_PIPELINE_NAME = "rakuten-continuous-training"
CONTINUOUS_BATCH_SIZE = 3000


def _get_cursor(session, pipeline_name: str = CONTINUOUS_PIPELINE_NAME) -> int:
    """Retourne last_processed_id pour ce pipeline (0 si absent, ligne créée à la volée)."""
    from src.data.db import PipelineCursor

    cursor = session.query(PipelineCursor).filter_by(pipeline_name=pipeline_name).first()
    if cursor is None:
        cursor = PipelineCursor(pipeline_name=pipeline_name, last_processed_id=0)
        session.add(cursor)
        session.commit()
        return 0
    return cursor.last_processed_id


def _update_cursor(session, new_id: int, pipeline_name: str = CONTINUOUS_PIPELINE_NAME) -> None:
    """Met à jour last_processed_id après un entraînement réussi."""
    from datetime import datetime, timezone

    from src.data.db import PipelineCursor

    cursor = session.query(PipelineCursor).filter_by(pipeline_name=pipeline_name).first()
    if cursor is None:
        cursor = PipelineCursor(pipeline_name=pipeline_name)
        session.add(cursor)
    cursor.last_processed_id = new_id
    cursor.last_run_at = datetime.now(timezone.utc)
    session.commit()


def train(
    test_size: float | None = None,
    random_state: int | None = None,
    limit: int | None = None,
    continuous: bool = False,
) -> dict[str, Any]:
    """Entraîne le modèle Rakuten, le journalise dans MLflow (Tracking + Registry),
    le compare à la version Production actuelle, et persiste l'artefact local
    (compatibilité avec predict.py)."""
    ensure_dirs()
    _setup_mlflow()

    test_size = test_size if test_size is not None else settings.test_size
    random_state = random_state if random_state is not None else settings.random_state

    min_id = None
    batch_max_id = None
    if continuous:
        from src.data.db import get_session

        with get_session() as session:
            min_id = _get_cursor(session)
        df = load_dataframe(limit=CONTINUOUS_BATCH_SIZE, min_id=min_id)
        batch_max_id = int(df["_id"].max())
        logger.info(
            "Mode continu : tranche (id > %d), %d lignes chargées, max id = %d",
            min_id, len(df), batch_max_id,
        )
    else:
        df = load_dataframe(limit=limit)

    feature_names = get_feature_columns(df)
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size, random_state)

    logger.info("Entraînement sur %d lignes (%d en test)", len(X_train), len(X_test))

    with mlflow.start_run() as run:
        pipeline = build_pipeline(feature_names, random_state)
        pipeline.fit(X_train, y_train)

        metrics = evaluate(pipeline, X_test, y_test)
        version = datetime.now(UTC).strftime("%Y%m%d%H%M%S")

        mlflow.log_params({
            "model_name": settings.model_name,
            "test_size": test_size,
            "random_state": random_state,
            "limit": limit,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_names),
            "classifier": "LogisticRegression",
        })
        mlflow.set_tags({
            "target_column": settings.target_column,
            "data_source": "postgresql:raw_samples",
            "data_fingerprint": _dataset_fingerprint(),
            "version": version,
        })
        mlflow.log_metrics({
            "accuracy": metrics["accuracy"],
            "weighted_f1": metrics["weighted_f1"],
            "weighted_precision": metrics["weighted_precision"],
            "weighted_recall": metrics["weighted_recall"],
        })
        
        if os.getenv("MLFLOW_LOG_ARTIFACTS", "true").lower() != "false":
             mlflow.log_dict(metrics["report"], "classification_report.json")
             mlflow.log_dict({"confusion_matrix": metrics["confusion_matrix"]}, "confusion_matrix.json")
        
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )
        run_id = run.info.run_id

    # Enregistrement dans le Model Registry (hors du bloc `with`, run déjà clos)
    registered = mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=settings.mlflow_registered_model_name)
    new_model_version = registered.version

    client = MlflowClient()
    comparison = _compare_and_promote(client, settings.mlflow_registered_model_name, new_model_version, metrics)
    new_model_version = str(new_model_version)

    if continuous and batch_max_id is not None:
        from src.data.db import get_session

        with get_session() as session:
            _update_cursor(session, batch_max_id)
        logger.info("Curseur mis à jour : last_processed_id = %d", batch_max_id)

    # Persistance locale : predict.py continue de fonctionner sans modification
    artifact = {
        "pipeline": pipeline,
        "feature_names": feature_names,
        "model_name": settings.model_name,
        "version": version,
        "trained_at": datetime.now(UTC).isoformat(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_model_version": new_model_version,
    }
    if comparison["decision"] == "promoted":
        joblib.dump(artifact, settings.model_path)
        logger.info("Modèle enregistré (local) : %s", settings.model_path)
    else:
        logger.info("Model not promoted: local copy %s left unchanged", settings.model_path)

    summary = {
        "model_name": settings.model_name,
        "version": version,
        "trained_at": artifact["trained_at"],
        "n_train": artifact["n_train"],
        "n_test": artifact["n_test"],
        "n_features": len(feature_names),
        "metrics": metrics,
        "mlflow_run_id": run_id,
        "mlflow_model_version": new_model_version,
        "comparison": comparison,
    }
    settings.metrics_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Métriques enregistrées : %s", settings.metrics_path)
    logger.info(
        "accuracy=%.4f | weighted_f1=%.4f | décision=%s",
        metrics["accuracy"], metrics["weighted_f1"], comparison["decision"],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Entraînement du modèle Rakuten")
    parser.add_argument("--test-size", type=float, default=None)
    parser.add_argument("--random-state", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Limiter à N lignes (test rapide)")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Mode versioning continu : traite les 3000 prochaines lignes depuis le curseur pipeline_cursor",
    )
    args = parser.parse_args()

    limit = args.limit if args.limit else None  # 0 ou None → pas de limite

    setup_logging()
    train(test_size=args.test_size, random_state=args.random_state, limit=limit, continuous=args.continuous)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())