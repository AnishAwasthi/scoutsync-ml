"""Database and model access for the Streamlit dashboard."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.config import get_settings
from src.db.session import MlbProjection, Player, check_db_connection, get_db_session
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES, build_training_frames
from src.ml.model import ScoutSyncTranslationModel
from src.ml.shap_engine import shap_for_player_row
from src.pipeline import load_tracking_dataframe, normalize_tracking_df


@dataclass
class PlayerProfile:
    player_id: int
    first_name: str
    last_name: str
    full_name: str
    birth_date: date
    age: int | None
    throws: str | None
    bats: str | None
    primary_position: str | None


@dataclass
class ProjectionView:
    player_id: int
    role: str
    target_season: int
    mean: float | None
    lower_90: float | None
    upper_90: float | None
    variance_delta: float | None
    shap_json: dict[str, Any] | None
    metric_label: str
    metric_unit: str


def db_status() -> tuple[bool, str]:
    from src.db.session import use_sqlite

    if check_db_connection():
        backend = "SQLite" if use_sqlite() else "PostgreSQL"
        return True, backend
    return False, "unavailable"


def list_players(session: Session) -> list[PlayerProfile]:
    players = session.query(Player).order_by(Player.last_name, Player.first_name).all()
    settings = get_settings()
    profiles = []
    for p in players:
        age = _age_at_season(p.birth_date, settings.validation_year)
        profiles.append(
            PlayerProfile(
                player_id=p.player_id,
                first_name=p.first_name,
                last_name=p.last_name,
                full_name=f"{p.first_name} {p.last_name}",
                birth_date=p.birth_date,
                age=age,
                throws=p.throws,
                bats=p.bats,
                primary_position=p.primary_position,
            )
        )
    return profiles


def _age_at_season(birth_date: date, context_year: int) -> int:
    age = context_year - birth_date.year
    if (birth_date.month, birth_date.day) > (7, 1):
        age -= 1
    return age


def get_player(session: Session, player_id: int) -> PlayerProfile | None:
    p = session.get(Player, player_id)
    if not p:
        return None
    settings = get_settings()
    return PlayerProfile(
        player_id=p.player_id,
        first_name=p.first_name,
        last_name=p.last_name,
        full_name=f"{p.first_name} {p.last_name}",
        birth_date=p.birth_date,
        age=_age_at_season(p.birth_date, settings.validation_year),
        throws=p.throws,
        bats=p.bats,
        primary_position=p.primary_position,
    )


def _parse_shap_json(raw: Any) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None
    if isinstance(raw, dict):
        return raw
    return None


def _projection_order_by(query):
    """Order projections by id (cloud schema) or calculation_date (full ORM schema)."""
    if hasattr(MlbProjection, "calculation_date"):
        return query.order_by(MlbProjection.calculation_date.desc())
    return query.order_by(MlbProjection.id.desc())


def get_projection(session: Session, player_id: int, role: str) -> ProjectionView | None:
    """Load latest projection for batter (wOBA) or pitcher (ERA)."""
    q = _projection_order_by(
        session.query(MlbProjection).filter(MlbProjection.player_id == player_id)
    )
    if role == "batter":
        q = q.filter(MlbProjection.proj_wOBA.isnot(None))
        row = q.first()
        if not row:
            return None
        return ProjectionView(
            player_id=player_id,
            role=role,
            target_season=row.target_season,
            mean=_to_float(row.proj_wOBA),
            lower_90=_to_float(row.proj_wOBA_lower_90),
            upper_90=_to_float(row.proj_wOBA_upper_90),
            variance_delta=_estimate_variance_delta(row),
            shap_json=_parse_shap_json(row.shap_explainability_json),
            metric_label="Projected MLB wOBA",
            metric_unit="wOBA",
        )

    q = _projection_order_by(
        session.query(MlbProjection).filter(
            MlbProjection.player_id == player_id,
            MlbProjection.proj_ERA.isnot(None),
        )
    )
    row = q.first()
    if not row:
        return None
    return ProjectionView(
        player_id=player_id,
        role=role,
        target_season=row.target_season,
        mean=_to_float(row.proj_ERA),
        lower_90=_to_float(row.proj_ERA_lower_90),
        upper_90=_to_float(row.proj_ERA_upper_90),
        variance_delta=_estimate_variance_delta(row),
        shap_json=_parse_shap_json(row.shap_explainability_json),
        metric_label="Projected MLB ERA",
        metric_unit="ERA",
    )


def _to_float(val) -> float | None:
    return float(val) if val is not None else None


def _estimate_variance_delta(proj: MlbProjection) -> float | None:
    """Derive sigma from 90% CI width when explicit variance is not stored."""
    if proj.proj_wOBA is not None and proj.proj_wOBA_lower_90 and proj.proj_wOBA_upper_90:
        return (float(proj.proj_wOBA_upper_90) - float(proj.proj_wOBA_lower_90)) / (2 * 1.645)
    if proj.proj_ERA is not None and proj.proj_ERA_lower_90 and proj.proj_ERA_upper_90:
        return (float(proj.proj_ERA_upper_90) - float(proj.proj_ERA_lower_90)) / (2 * 1.645)
    return None


def get_player_feature_row(session: Session, player_id: int, role: str) -> pd.Series | None:
    raw = load_tracking_dataframe(session)
    player_raw = raw[raw["player_id"] == player_id]
    if player_raw.empty:
        return None
    normalized = normalize_tracking_df(player_raw)
    _, pitchers, batters, _, _ = build_training_frames(normalized, pd.DataFrame())
    frame = pitchers if role == "pitcher" else batters
    if frame.empty:
        return None
    subset = frame[frame["player_id"] == player_id]
    if subset.empty:
        return None
    return subset.iloc[0]


@dataclass
class LoadedModel:
    model: ScoutSyncTranslationModel
    available: bool
    message: str


def load_translation_model() -> LoadedModel:
    model = ScoutSyncTranslationModel()
    try:
        model.load()
    except Exception as exc:
        return LoadedModel(model=model, available=False, message=str(exc))

    has_batter = model.batter_model is not None
    has_pitcher = model.pitcher_model is not None
    if not has_batter and not has_pitcher:
        return LoadedModel(
            model=model,
            available=False,
            message="No trained models found. Run: python main.py train",
        )
    return LoadedModel(model=model, available=True, message="ok")


def _shap_dict_to_contributions(shap_data: dict[str, Any], target_key: str) -> list[dict[str, Any]]:
    """Support nested SHAP lists or flat cloud mock dicts {label: impact}."""
    items = shap_data.get(target_key)
    if isinstance(items, list) and items:
        return items
    for value in shap_data.values():
        if isinstance(value, list) and value:
            return value
    contributions = []
    for key, val in shap_data.items():
        if isinstance(val, (int, float)):
            contributions.append(
                {"feature": key, "label": key, "impact": float(val)},
            )
    return contributions


def resolve_shap_contributions(
    session: Session,
    player_id: int,
    role: str,
    projection: ProjectionView | None,
    model_bundle: LoadedModel,
) -> list[dict[str, Any]]:
    """Return SHAP contributions from DB or recompute via saved model."""
    target_key = "proj_wOBA" if role == "batter" else "proj_ERA"
    if projection and projection.shap_json:
        items = _shap_dict_to_contributions(projection.shap_json, target_key)
        if items:
            return items

    if not model_bundle.available:
        return []

    features = BATTER_FEATURES if role == "batter" else PITCHER_FEATURES
    row = get_player_feature_row(session, player_id, role)
    if row is None:
        return []

    est = model_bundle.model.batter_model if role == "batter" else model_bundle.model.pitcher_model
    if est is None:
        return []

    try:
        payload = shap_for_player_row(est, row, features, role)
        return payload.get(target_key, [])
    except Exception:
        return []
