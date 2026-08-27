"""End-to-end data pipeline: load, normalize, feature build, train, project."""

from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session, joinedload

from db.models import MlbProjection, PlayerGroundTruth, RawTrackingData, StadiumEnvironment
from src.config import get_settings
from src.db.session import has_tracking_schema
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES, build_training_frames
from src.ingestion import pybaseball_loader
from src.logging_config import get_logger, log_filter_step
from src.ml.model import TARGET_COLUMN, ScoutSyncTranslationModel, train_models_from_frames
from src.ml.shap_engine import shap_for_player_row
from src.normalization.environment import apply_environmental_adjustments
from src.normalization.league_context import apply_league_normalization, compute_league_baselines

Z_90 = 1.645


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
                "horizontal_break": float(r.horizontal_break)
                if r.horizontal_break is not None
                else None,
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
    # Read through the module: ``LEAGUE_BASELINES`` is reassigned during seeding, and a
    # `from ... import LEAGUE_BASELINES` would freeze the empty frame captured at import
    # time -- which silently dropped the MLB tier-1 baselines from every normalization.
    mlb_reference = pybaseball_loader.LEAGUE_BASELINES
    if not mlb_reference.empty:
        mlb_baselines = mlb_reference.copy()
        mlb_baselines["league_id"] = 1
        baselines = pd.concat([baselines, mlb_baselines], ignore_index=True).drop_duplicates(
            subset=["league_id", "metric"], keep="first"
        )
    for col in ("adj_velocity", "adj_vertical_break", "exit_velocity"):
        source = col if col in env.columns else col.replace("adj_", "")
        if source in env.columns:
            env = apply_league_normalization(env, baselines, source, output_col=f"{source}_norm")
    return env


def load_ground_truth(session: Session) -> pd.DataFrame:
    """
    Simulation labels for the synthetic cohort.

    Only synthetic players have rows here. Real players seeded from Statcast are
    reference data for league baselines and are deliberately left unlabeled -- there is
    no public amateur-to-MLB player linkage to build real labels from.
    """
    rows = session.query(PlayerGroundTruth).all()
    if not rows:
        return pd.DataFrame(
            columns=["player_id", "role", "true_wOBA", "true_ERA", "latent_talent"]
        )
    return pd.DataFrame(
        [
            {
                "player_id": r.player_id,
                "role": r.role,
                "true_wOBA": float(r.true_wOBA) if r.true_wOBA is not None else np.nan,
                "true_ERA": float(r.true_ERA) if r.true_ERA is not None else np.nan,
                "latent_talent": float(r.latent_talent),
            }
            for r in rows
        ]
    )


def attach_training_labels(
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    ground_truth: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Merge each player's own ground-truth outcome onto its feature row."""
    logger = get_logger()
    if ground_truth.empty:
        logger.warning("no_ground_truth_rows; models cannot be trained")
        return pitcher_df, batter_df

    labeled = []
    for role, frame in (("pitcher", pitcher_df), ("batter", batter_df)):
        if frame.empty:
            labeled.append(frame)
            continue
        subset = ground_truth[ground_truth["role"] == role][
            ["player_id", "true_wOBA", "true_ERA", "latent_talent"]
        ]
        merged = frame.merge(subset, on="player_id", how="left")
        matched = int(merged[TARGET_COLUMN[role]].notna().sum())
        logger.info(f"labels_attached role={role} rows={len(merged)} labeled={matched}")
        labeled.append(merged)
    return labeled[0], labeled[1]


def run_train_pipeline(session: Session) -> ScoutSyncTranslationModel:
    """Full train: load -> normalize -> features -> fit -> persist projections."""
    raw = load_tracking_dataframe(session)
    if raw.empty:
        raise RuntimeError("No tracking data. Run: python main.py seed")

    normalized = normalize_tracking_df(raw)
    _, pitchers, batters, _, _ = build_training_frames(normalized, pd.DataFrame())

    ground_truth = load_ground_truth(session)
    pitchers, batters = attach_training_labels(pitchers, batters, ground_truth)

    # Tier 1 is MLB; anything at tier 4+ is the amateur cohort this tool exists to translate.
    amateur_ids = set(
        raw.loc[raw["competition_tier"] >= 4, "player_id"].dropna().astype(int).tolist()
    )
    get_logger().info(f"amateur_cohort players={len(amateur_ids)}")

    model = train_models_from_frames(pitchers, batters)
    persist_projections(session, model, pitchers, batters, eligible_player_ids=amateur_ids)
    session.commit()
    return model


def persist_projections(
    session: Session,
    model: ScoutSyncTranslationModel,
    pitchers: pd.DataFrame,
    batters: pd.DataFrame,
    eligible_player_ids: set[int] | None = None,
) -> int:
    """
    Write one projection per player-season, replacing any previous run.

    ``eligible_player_ids`` restricts output to the amateur cohort. Projecting an MLB
    reference player *to* MLB is incoherent -- those rows exist to set league baselines,
    not to be translated -- and letting them through put them in the dashboard's player
    picker as though they were prospects.

    Re-running training used to append a fresh set of rows every time, so the dashboard
    read whichever row happened to sort first out of a steadily growing pile. Clearing
    the target season first keeps exactly one current projection per player.
    """
    settings = get_settings()
    season = settings.validation_year
    logger = get_logger()

    deleted = (
        session.query(MlbProjection)
        .filter(MlbProjection.target_season == season)
        .delete(synchronize_session=False)
    )
    session.flush()
    if deleted:
        logger.info(f"projections_cleared season={season} rows={deleted}")

    written = 0
    for role, frame, features in [
        ("pitcher", pitchers, PITCHER_FEATURES),
        ("batter", batters, BATTER_FEATURES),
    ]:
        estimator = model.get_estimator(role)
        if estimator is None or frame.empty:
            continue
        valid = frame.dropna(subset=list(features), how="any")
        if eligible_player_ids is not None:
            valid = valid[valid["player_id"].isin(eligible_player_ids)]
        if valid.empty:
            continue

        preds = model.predict(valid[features], role)
        std = model.residual_for(role)
        background = valid[features]

        for offset, (_, row) in enumerate(valid.iterrows()):
            mean = float(preds[offset])
            lower, upper = mean - Z_90 * std, mean + Z_90 * std
            shap_json = shap_for_player_row(
                estimator, row, list(features), role, background=background
            )
            fields = dict(
                player_id=int(row["player_id"]),
                target_season=season,
                shap_explainability_json=shap_json,
            )
            if role == "batter":
                fields.update(
                    proj_wOBA=round(mean, 3),
                    proj_wOBA_lower_90=round(lower, 3),
                    proj_wOBA_upper_90=round(upper, 3),
                )
            else:
                fields.update(
                    proj_ERA=round(mean, 2),
                    proj_ERA_lower_90=round(lower, 2),
                    proj_ERA_upper_90=round(upper, 2),
                )
            session.add(MlbProjection(**fields))
            written += 1

    logger.info(f"projections_persisted season={season} rows={written}")
    return written


def get_player_breakdown(session: Session, player_id: int, role: str | None = None) -> dict:
    """Raw vs adjusted distributions for charting."""
    if not has_tracking_schema():
        return {
            "player_id": player_id,
            "error": "no_tracking_data",
            "raw_distribution": {"name": "raw", "bins": [], "counts": []},
            "adjusted_distribution": {"name": "adjusted", "bins": [], "counts": []},
            "shap_contributions": latest_shap(session, player_id),
        }

    raw = load_tracking_dataframe(session)
    player_raw = raw[raw["player_id"] == player_id]
    if player_raw.empty:
        return {"player_id": player_id, "error": "not_found"}
    adjusted = normalize_tracking_df(player_raw)

    def bins(series: pd.Series, name: str) -> dict:
        clean = pd.Series(series).dropna()
        if clean.empty:
            return {"name": name, "bins": [], "counts": []}
        counts, edges = np.histogram(clean, bins=10)
        return {
            "name": name,
            "bins": [float(e) for e in edges[:-1]],
            "counts": [int(c) for c in counts],
        }

    # Prefer the caller's role. Falling back to column sniffing mislabels any player
    # who has both pitch and batted-ball rows.
    if role is not None:
        is_pitcher = role == "pitcher"
    else:
        is_pitcher = player_raw["release_speed"].notna().any()
    raw_col = "release_speed" if is_pitcher else "exit_velocity"
    adj_col = "adj_velocity" if is_pitcher else "adj_exit_velocity"

    return {
        "player_id": player_id,
        "metric": "Release Speed (mph)" if is_pitcher else "Exit Velocity (mph)",
        "raw_distribution": bins(player_raw.get(raw_col, pd.Series(dtype=float)), "raw"),
        "adjusted_distribution": bins(adjusted.get(adj_col, pd.Series(dtype=float)), "adjusted"),
        "shap_contributions": latest_shap(session, player_id),
    }


def latest_shap(session: Session, player_id: int) -> dict:
    """Most recent stored SHAP payload for a player, or an empty dict."""
    import json

    proj = (
        session.query(MlbProjection)
        .filter(MlbProjection.player_id == player_id)
        .order_by(MlbProjection.calculation_date.desc())
        .first()
    )
    if not proj or not proj.shap_explainability_json:
        return {}
    payload = proj.shap_explainability_json
    if isinstance(payload, str):
        try:
            return json.loads(payload)
        except json.JSONDecodeError:
            return {}
    return payload if isinstance(payload, dict) else {}


# Backwards-compatible alias for the previous private name.
_latest_shap = latest_shap
