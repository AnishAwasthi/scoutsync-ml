"""End-to-end data pipeline: load, normalize, feature build, train, project."""

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, joinedload

from db.models import League, MlbProjection, Player, RawTrackingData, StadiumEnvironment
from src.config import get_settings
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES, build_training_frames
from src.ingestion.pybaseball_loader import (
    LEAGUE_BASELINES,
    PSEUDO_ROOKIE_MAP,
    build_pseudo_rookie_map,
    load_rookie_outcomes,
)
from src.db.session import has_tracking_schema
from src.logging_config import get_logger, log_filter_step
from src.ml.model import ScoutSyncTranslationModel, prepare_targets, train_models_from_frames
from src.ml.shap_engine import shap_for_player_row
from src.normalization.environment import apply_environmental_adjustments
from src.normalization.league_context import apply_league_normalization, compute_league_baselines


def load_tracking_dataframe(session: Session) -> pd.DataFrame:
    """Load raw tracking joined with player, stadium, league context."""
    if not has_tracking_schema():
        get_logger().info("load_tracking_skipped reason=no_raw_tracking_table")
        return pd.DataFrame()

    rows = (
        session.query(RawTrackingData)
        .options(
            joinedload(RawTrackingData.player),
            joinedload(RawTrackingData.stadium).joinedload(StadiumEnvironment.league),
        )
        .all()
    )
    records = []
    for r in rows:
        stadium = r.stadium
        league = stadium.league if stadium else None
        player = r.player
        records.append(
            {
                "tracking_id": r.tracking_id,
                "player_id": r.player_id,
                "stadium_id": r.stadium_id,
                "game_date": r.game_date,
                "context_year": r.context_year,
                "pitch_type": r.pitch_type,
                "release_speed": float(r.release_speed) if r.release_speed is not None else None,
                "spin_rate": r.spin_rate,
                "vertical_break": float(r.vertical_break) if r.vertical_break is not None else None,
                "horizontal_break": float(r.horizontal_break) if r.horizontal_break is not None else None,
                "vertical_approach_angle": float(r.vertical_approach_angle)
                if r.vertical_approach_angle is not None
                else None,
                "extension": float(r.extension) if r.extension is not None else None,
                "plate_x": float(r.plate_x) if r.plate_x is not None else None,
                "plate_z": float(r.plate_z) if r.plate_z is not None else None,
                "exit_velocity": float(r.exit_velocity) if r.exit_velocity is not None else None,
                "launch_angle": float(r.launch_angle) if r.launch_angle is not None else None,
                "hit_distance": r.hit_distance,
                "is_strikeout": r.is_strikeout,
                "is_walk": r.is_walk,
                "is_chase": r.is_chase,
                "birth_date": player.birth_date if player else date(2000, 1, 1),
                "altitude": stadium.altitude if stadium else get_settings().alt_std_ft,
                "temperature_mean": float(stadium.temperature_mean)
                if stadium and stadium.temperature_mean is not None
                else 22.0,
                "humidity_mean": float(stadium.humidity_mean)
                if stadium and stadium.humidity_mean is not None
                else 50.0,
                "league_id": league.league_id if league else None,
                "competition_tier": league.competition_tier if league else 4,
                "base_run_environment": float(league.base_run_environment) if league else 5.0,
                "abbreviation": league.abbreviation if league else "NCAA",
            }
        )
    df = pd.DataFrame(records)
    log_filter_step("load_tracking", 0, len(df))
    return df


def normalize_tracking_df(df: pd.DataFrame) -> pd.DataFrame:
    """Apply environmental then league normalization."""
    if df.empty:
        return df
    env = apply_environmental_adjustments(df)
    baselines = compute_league_baselines(
        env,
        ["adj_velocity", "adj_vertical_break", "exit_velocity"],
        group_cols=["league_id"],
    )
    if not LEAGUE_BASELINES.empty:
        mlb_bl = LEAGUE_BASELINES.copy()
        mlb_bl["league_id"] = 1
        baselines = pd.concat([baselines, mlb_bl], ignore_index=True).drop_duplicates(
            subset=["league_id", "metric"], keep="first"
        )
    for metric, col in [
        ("adj_velocity", "adj_velocity"),
        ("adj_vertical_break", "adj_vertical_break"),
        ("exit_velocity", "exit_velocity"),
    ]:
        if col.replace("adj_", "") in env.columns or col in env.columns:
            src = col if col in env.columns else col.replace("adj_", "")
            env = apply_league_normalization(env, baselines, src, output_col=f"{src}_norm")
    return env


def attach_training_labels(
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    amateur_ids: list[int],
    validation_year: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rookies = load_rookie_outcomes(validation_year)
    pseudo = build_pseudo_rookie_map(amateur_ids, rookies)
    if pseudo.empty:
        return pitcher_df, batter_df

    labels = pseudo.rename(
        columns={
            "true_wOBA": "true_wOBA",
            "true_ERA": "true_ERA",
        }
    )
    labels["proj_mlb_wOBA"] = labels["true_wOBA"]
    labels["proj_mlb_ERA"] = labels["true_ERA"]
    labels["expected_variance_delta"] = 0.02

    p = pitcher_df.merge(labels[labels["role"] == "pitcher"], on="player_id", how="left")
    b = batter_df.merge(labels[labels["role"] == "batter"], on="player_id", how="left")
    return p, b


def run_train_pipeline(session: Session) -> ScoutSyncTranslationModel:
    """Full train: load -> normalize -> features -> fit -> persist projections."""
    settings = get_settings()
    raw = load_tracking_dataframe(session)
    if raw.empty:
        raise RuntimeError("No tracking data. Run: python main.py seed")

    normalized = normalize_tracking_df(raw)
    combined, pitchers, batters, _, _ = build_training_frames(normalized, pd.DataFrame())

    amateur_ids = (
        raw[raw["competition_tier"] >= 4]["player_id"].dropna().unique().tolist()
    )
    pitchers, batters = attach_training_labels(
        pitchers, batters, amateur_ids, settings.validation_year
    )

    model = train_models_from_frames(pitchers, batters)
    persist_projections(session, model, pitchers, batters)
    session.commit()
    return model


def persist_projections(
    session: Session,
    model: ScoutSyncTranslationModel,
    pitchers: pd.DataFrame,
    batters: pd.DataFrame,
) -> None:
    settings = get_settings()
    z = 1.645

    for role, frame, features in [
        ("pitcher", pitchers, PITCHER_FEATURES),
        ("batter", batters, BATTER_FEATURES),
    ]:
        est = model.pitcher_model if role == "pitcher" else model.batter_model
        if est is None or frame.empty:
            continue
        valid = frame.dropna(subset=features, how="any")
        if valid.empty:
            continue
        preds = est.predict(valid[features].values)
        std = model.residual_std.get(role, {"woba": 0.03, "era": 0.5, "variance": 0.02})

        for idx, (_, row) in enumerate(valid.iterrows()):
            pred = preds[idx]
            shap_json = shap_for_player_row(est, row, features, role)
            woba = float(pred[0]) if role == "batter" else None
            era = float(pred[1]) if role == "pitcher" else float(pred[1])
            if role == "batter":
                era = None
                woba_ci = (
                    woba - z * std["woba"],
                    woba + z * std["woba"],
                )
                proj = MlbProjection(
                    player_id=int(row["player_id"]),
                    target_season=settings.validation_year,
                    proj_wOBA=woba,
                    proj_wOBA_lower_90=woba_ci[0],
                    proj_wOBA_upper_90=woba_ci[1],
                    proj_ERA=None,
                    shap_explainability_json=shap_json,
                )
            else:
                era_ci = (era - z * std["era"], era + z * std["era"])
                proj = MlbProjection(
                    player_id=int(row["player_id"]),
                    target_season=settings.validation_year,
                    proj_wOBA=None,
                    proj_ERA=era,
                    proj_ERA_lower_90=era_ci[0],
                    proj_ERA_upper_90=era_ci[1],
                    shap_explainability_json=shap_json,
                )
            session.add(proj)
    get_logger().info("projections_persisted")


def get_player_breakdown(session: Session, player_id: int) -> dict:
    """Raw vs adjusted distributions for charting API."""
    if not has_tracking_schema():
        return {
            "player_id": player_id,
            "error": "no_tracking_data",
            "raw_distribution": {"name": "raw", "bins": [], "counts": []},
            "adjusted_distribution": {"name": "adjusted", "bins": [], "counts": []},
            "shap_contributions": _latest_shap(session, player_id),
        }

    raw = load_tracking_dataframe(session)
    player_raw = raw[raw["player_id"] == player_id]
    if player_raw.empty:
        return {"player_id": player_id, "error": "not_found"}
    adjusted = normalize_tracking_df(player_raw)

    def bins(series: pd.Series, name: str) -> dict:
        clean = series.dropna()
        if clean.empty:
            return {"name": name, "bins": [], "counts": []}
        counts, edges = np.histogram(clean, bins=10)
        return {
            "name": name,
            "bins": [float(e) for e in edges[:-1]],
            "counts": [int(c) for c in counts],
        }

    return {
        "player_id": player_id,
        "raw_distribution": bins(player_raw.get("release_speed", player_raw.get("exit_velocity", pd.Series())), "raw"),
        "adjusted_distribution": bins(
            adjusted.get("adj_velocity", adjusted.get("adj_velocity_norm", pd.Series())),
            "adjusted",
        ),
        "shap_contributions": _latest_shap(session, player_id),
    }


def _latest_shap(session: Session, player_id: int) -> dict:
    """Load SHAP JSON using schema compatible with cloud lite or full ORM."""
    from src.db.session import MlbProjection as SessionProjection
    from src.db.session import use_sqlite
    import json

    Model = SessionProjection if use_sqlite() else MlbProjection
    q = session.query(Model).filter_by(player_id=player_id)
    if hasattr(Model, "calculation_date"):
        proj = q.order_by(Model.calculation_date.desc()).first()
    else:
        proj = q.order_by(Model.id.desc()).first()
    if not proj or not proj.shap_explainability_json:
        return {}
    raw = proj.shap_explainability_json
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return raw if isinstance(raw, dict) else {}
