"""Lecture des données depuis PostgreSQL et construction des features."""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sqlalchemy import select

from src.config import settings
from src.data.db import RawSample, get_session
from src.features.image_features import ImageEmbeddingTransformer

logger = logging.getLogger(__name__)

TARGET = settings.target_column


def load_dataframe(limit: int | None = None, min_id: int | None = None) -> pd.DataFrame:
    """Charge raw_samples et aplatit le JSONB en colonnes.

    min_id : si fourni, ne charge que les lignes avec id > min_id (utilisé par le
    versioning continu pour traiter la "prochaine tranche" de données). La colonne
    technique '_id' est conservée dans le DataFrame retourné pour permettre de
    calculer le nouveau curseur après entraînement ; elle n'est pas une feature.
    """
    stmt = select(RawSample.id, RawSample.features, RawSample.target).order_by(RawSample.id)
    if min_id is not None:
        stmt = stmt.where(RawSample.id > min_id)
    if limit:
        stmt = stmt.limit(limit)

    with get_session() as session:
        rows = session.execute(stmt).all()

    if not rows:
        raise RuntimeError(
            "Aucune donnée en base. Lancez d'abord : python -m src.data.make_dataset"
        )

    df = pd.DataFrame(
        [{"_id": row_id, **features, TARGET: target} for row_id, features, target in rows]
    )
    logger.info("Données chargées depuis PostgreSQL : %d lignes, %d colonnes", *df.shape)
    return df


#def get_feature_columns(df: pd.DataFrame) -> list[str]:
#    return [c for c in df.columns if c != TARGET]
def get_feature_columns(df: pd.DataFrame) -> list[str]:
    return ["designation", "description", "image_path"]


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare variables explicatives et cible, en respectant le type de chaque colonne."""
    features = get_feature_columns(df)
    X = df[features].copy()

    X["designation"] = X["designation"].fillna("").astype(str)
    X["description"] = X["description"].fillna("").astype(str)
    # image_path reste une chaîne (ou None) : gérée directement par ImageEmbeddingTransformer

    y = df[TARGET].astype(int)
    return X, y


def build_preprocessor(feature_names):
    return ColumnTransformer(
        transformers=[
            ("text_designation", TfidfVectorizer(max_features=5000), "designation"),
            ("text_description", TfidfVectorizer(max_features=5000), "description"),
            ("image", ImageEmbeddingTransformer(), ["image_path"]),
        ],
        remainder="drop",
    )


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Découpage stratifié train/test."""
    return train_test_split(
        X,
        y,
        test_size=test_size if test_size is not None else settings.test_size,
        random_state=random_state if random_state is not None else settings.random_state,
        stratify=y,
    )


def prepare_frame(records: list[dict], feature_names: list[str]) -> pd.DataFrame:
    """Aligne des enregistrements bruts sur les colonnes attendues par le modèle,
    en respectant le type de chaque colonne (texte vs numérique)."""
    df = pd.DataFrame(records).reindex(columns=feature_names)
    for col in ("productid", "imageid"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("designation", "description"):
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
    return df