"""Fast offline seeding for Streamlit Cloud (no pybaseball downloads)."""

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from db.models import League, Player, RawTrackingData, StadiumEnvironment
from src.config import PROJECT_ROOT, get_settings
from src.ingestion.pybaseball_loader import (
    LEAGUE_BASELINES,
    ROOKIE_OUTCOMES,
    compute_league_baselines_from_statcast,
    load_offline_statcast_lite,
    load_rookie_outcomes_lite,
)
from src.ingestion.synthetic_seed import seed_synthetic_data
from src.logging_config import get_logger
from src.runtime import is_streamlit_cloud


def seed_cloud_lite(session: Session, num_amateur_players: int = 8) -> dict:
    """
    Populate DB for dashboard demo without network or large DataFrames.

    ~5 MLB sample players from statcast_lite.csv + 8 amateur synthetic players.
    """
    if not is_streamlit_cloud():
        get_logger().warning("seed_cloud_lite called outside cloud; proceeding anyway")

    mlb_stats = seed_mlb_from_offline_csv(session)
    amateur_stats = seed_synthetic_data(session, num_players=num_amateur_players, lite=True)
    get_logger().info(f"cloud_lite_seed_complete mlb={mlb_stats} amateur={amateur_stats}")
    return {"mlb": mlb_stats, "synthetic": amateur_stats, "mode": "cloud_lite"}


def seed_mlb_from_offline_csv(session: Session) -> dict:
    """Insert MLB tracking from pre-baked CSV; sets global baselines and rookie outcomes."""
    global LEAGUE_BASELINES, ROOKIE_OUTCOMES

    settings = get_settings()
    statcast_df = load_offline_statcast_lite()
    LEAGUE_BASELINES = compute_league_baselines_from_statcast(statcast_df)
    ROOKIE_OUTCOMES = load_rookie_outcomes_lite(settings.validation_year)

    if session.query(League).count() == 0:
        from src.db.session import seed_reference_leagues

        seed_reference_leagues()

    mlb = session.query(League).filter_by(abbreviation="MLB").first()
    if not mlb:
        raise RuntimeError("MLB league row missing. Run init-db first.")

    stadium = (
        session.query(StadiumEnvironment)
        .filter_by(league_id=mlb.league_id, stadium_name="MLB Reference Park")
        .first()
    )
    if not stadium:
        stadium = StadiumEnvironment(
            league_id=mlb.league_id,
            stadium_name="MLB Reference Park",
            altitude=500,
            temperature_mean=22.0,
            humidity_mean=50.0,
        )
        session.add(stadium)
        session.flush()

    inserted = 0
    players_by_name: dict[str, Player] = {}

    for _, row in statcast_df.iterrows():
        name = str(row.get("player_name", "MLB Player"))
        if name not in players_by_name:
            parts = name.split()
            players_by_name[name] = Player(
                first_name=parts[0],
                last_name=parts[-1] if len(parts) > 1 else "Player",
                birth_date=date(1996, 6, 15),
                throws="R",
                bats="R",
                primary_position=["P", "OF", "SS", "C", "1B"][len(players_by_name) % 5],
            )
            session.add(players_by_name[name])
            session.flush()

        player = players_by_name[name]
        year = int(row.get("game_year", settings.validation_year))
        gd = pd.to_datetime(row["game_date"]).date()

        if pd.notna(row.get("release_speed")):
            session.add(
                RawTrackingData(
                    player_id=player.player_id,
                    stadium_id=stadium.stadium_id,
                    game_date=gd,
                    context_year=year,
                    pitch_type=str(row.get("pitch_type", "FF"))[:3],
                    release_speed=float(row["release_speed"]),
                    spin_rate=int(row["release_spin_rate"])
                    if pd.notna(row.get("release_spin_rate"))
                    else None,
                    vertical_break=float(row["pfx_z"] * 12) if pd.notna(row.get("pfx_z")) else None,
                    horizontal_break=float(row["pfx_x"] * 12) if pd.notna(row.get("pfx_x")) else None,
                    vertical_approach_angle=-5.0,
                    extension=6.0,
                    plate_x=0.0,
                    plate_z=2.5,
                )
            )
            inserted += 1

        if pd.notna(row.get("launch_speed")):
            session.add(
                RawTrackingData(
                    player_id=player.player_id,
                    stadium_id=stadium.stadium_id,
                    game_date=gd,
                    context_year=year,
                    exit_velocity=float(row["launch_speed"]),
                    launch_angle=float(row.get("launch_angle", 15)),
                    hit_distance=300,
                    plate_x=0.0,
                    plate_z=2.5,
                    is_strikeout=False,
                    is_chase=False,
                )
            )
            inserted += 1

    session.commit()
    get_logger().info(
        f"mlb_offline_seed players={len(players_by_name)} tracking_rows={inserted} "
        f"path={PROJECT_ROOT / 'data' / 'samples' / 'statcast_lite.csv'}"
    )
    return {
        "mlb_tracking_rows": inserted,
        "mlb_players": len(players_by_name),
        "baselines": len(LEAGUE_BASELINES),
    }
