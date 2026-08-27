"""
Statcast ingestion for MLB reference data.

MLB rows exist to establish the tier-1 league baselines (mu/sigma per metric) that the
league-normalization step normalizes against. They are deliberately *not* training
labels: linking an amateur player to their eventual major-league outcome requires an
identity mapping that no public dataset provides. Training labels come from the
simulation ground truth in ``src.ingestion.synthetic_seed`` instead.

An earlier version of this module fabricated labels by pairing amateur player *i* with
MLB rookie *i mod N* -- a round-robin with no relationship between the two players. Any
accuracy measured against those labels was meaningless, so the whole mechanism is gone.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import League, Player, RawTrackingData, StadiumEnvironment
from src.config import PROJECT_ROOT, get_settings
from src.logging_config import get_logger, log_filter_step
from src.runtime import is_streamlit_cloud

# Tier-1 baselines derived from the Statcast sample, consumed by normalize_tracking_df.
LEAGUE_BASELINES: pd.DataFrame = pd.DataFrame()

MAX_REFERENCE_PLAYERS = 60


def _ensure_cache_dir() -> Path:
    path = Path(get_settings().pybaseball_cache_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_offline_statcast_lite() -> pd.DataFrame:
    """Pre-baked Statcast sample for Streamlit Cloud (no network)."""
    path = PROJECT_ROOT / "data" / "samples" / "statcast_lite.csv"
    logger = get_logger()
    if path.exists():
        df = pd.read_csv(path, parse_dates=["game_date"])
        logger.info(f"statcast_lite_csv rows={len(df)} path={path}")
        return df
    logger.warning(f"statcast_lite_csv missing path={path}; using synthetic stand-in")
    return _synthetic_mlb_statcast(2023, 2024, max_rows=40)


def load_statcast_sample(start_year: int, end_year: int, max_rows: int = 8000) -> pd.DataFrame:
    """
    Load MLB Statcast data via pybaseball with a local CSV cache.

    Falls back to synthetic MLB-like rows when pybaseball or the network is unavailable,
    logging a warning so the substitution is visible. On Streamlit Cloud this never hits
    the network -- it reads the committed lite CSV.
    """
    if is_streamlit_cloud():
        return load_offline_statcast_lite()

    logger = get_logger()
    cache = _ensure_cache_dir() / f"statcast_{start_year}_{end_year}.csv"

    if cache.exists():
        logger.info(f"statcast_cache_hit path={cache}")
        return pd.read_csv(cache, parse_dates=["game_date"], nrows=max_rows)

    try:
        from pybaseball import statcast

        logger.info(f"statcast_download start={start_year} end={end_year}")
        df = statcast(start_dt=f"{start_year}-04-01", end_dt=f"{end_year}-10-01")
        if df is None or df.empty:
            raise ValueError("Empty statcast response")
        df = df.head(max_rows)
        df.to_csv(cache, index=False)
        logger.info(f"statcast_cached rows={len(df)} path={cache}")
        return df
    except Exception as exc:
        logger.warning(f"statcast_fallback reason={exc}")
        return _synthetic_mlb_statcast(start_year, end_year, max_rows)


def _synthetic_mlb_statcast(start_year: int, end_year: int, max_rows: int) -> pd.DataFrame:
    """MLB-shaped stand-in used only when the real feed cannot be reached."""
    rng = np.random.default_rng(7)
    n = min(max_rows, 5000)
    years = list(range(start_year, end_year + 1))
    return pd.DataFrame(
        {
            "game_date": pd.date_range(f"{years[0]}-04-01", periods=n, freq="h"),
            "release_speed": rng.normal(93, 3, n),
            "release_spin_rate": rng.integers(2000, 2800, n),
            "pfx_z": rng.normal(1.2, 0.4, n),
            "pfx_x": rng.normal(-0.5, 0.3, n),
            "launch_speed": rng.normal(88, 8, n),
            "launch_angle": rng.normal(12, 15, n),
            "player_name": [f"MLB Player {i % 200}" for i in range(n)],
            "pitcher": [10000 + i % 100 for i in range(n)],
            "batter": [20000 + i % 100 for i in range(n)],
            "game_year": rng.choice(years, n),
            "pitch_type": rng.choice(["FF", "SL", "CH", "CU"], n),
        }
    )


def compute_league_baselines_from_statcast(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Derive mu/sigma for MLB (tier 1) metrics."""
    records = []
    mappings = [
        ("release_speed", "release_speed"),
        ("release_spin_rate", "spin_rate"),
        ("launch_speed", "exit_velocity"),
    ]
    for source, metric in mappings:
        if source not in statcast_df.columns:
            continue
        series = statcast_df[source].dropna()
        if series.empty:
            continue
        records.append(
            {
                "league_id": 1,
                "abbreviation": "MLB",
                "competition_tier": 1,
                "metric": metric,
                "mu_league": float(series.mean()),
                "sigma_league": float(series.std(ddof=0)) or 1.0,
            }
        )
    return pd.DataFrame(records)


def _reference_stadium(session: Session, mlb: League) -> StadiumEnvironment:
    stadium = (
        session.query(StadiumEnvironment)
        .filter_by(league_id=mlb.league_id, stadium_name="MLB Reference Park")
        .first()
    )
    if stadium:
        return stadium
    stadium = StadiumEnvironment(
        league_id=mlb.league_id,
        stadium_name="MLB Reference Park",
        altitude=500,
        temperature_mean=22.0,
        humidity_mean=50.0,
    )
    session.add(stadium)
    session.flush()
    return stadium


def seed_mlb_statcast(session: Session, validation_year: int | None = None) -> dict:
    """Load a Statcast sample into the DB and populate the tier-1 baselines."""
    if is_streamlit_cloud():
        from src.ingestion.lite_seed import seed_mlb_from_offline_csv

        return seed_mlb_from_offline_csv(session)

    global LEAGUE_BASELINES
    settings = get_settings()
    validation_year = validation_year or settings.validation_year

    statcast_df = load_statcast_sample(validation_year - 3, validation_year, max_rows=6000)
    LEAGUE_BASELINES = compute_league_baselines_from_statcast(statcast_df)

    mlb = session.query(League).filter_by(abbreviation="MLB").first()
    if not mlb:
        raise RuntimeError("MLB league row missing. Run: python main.py init-db")
    stadium = _reference_stadium(session, mlb)

    inserted = 0
    # One Player row per distinct name. The previous version created a new player for
    # every pitch, producing hundreds of one-row "players" that polluted the roster.
    players_by_name: dict[str, Player] = {}

    for i, row in statcast_df.head(500).iterrows():
        name = str(row.get("player_name") or f"MLB Player {i}")
        player = players_by_name.get(name)
        if player is None:
            if len(players_by_name) >= MAX_REFERENCE_PLAYERS:
                continue
            parts = name.split()
            player = Player(
                first_name=parts[0] if parts else "MLB",
                last_name=parts[-1] if len(parts) > 1 else "Player",
                birth_date=date(1995, 1, 1),
                throws="R",
                bats="R",
                primary_position=["P", "OF", "SS", "C", "1B"][len(players_by_name) % 5],
            )
            session.add(player)
            session.flush()
            players_by_name[name] = player

        year = int(row.get("game_year", validation_year))
        game_date = row.get("game_date", date(year, 6, 1))
        if hasattr(game_date, "date"):
            game_date = game_date.date()
        elif isinstance(game_date, str):
            game_date = pd.to_datetime(game_date).date()

        if pd.notna(row.get("release_speed")):
            session.add(
                RawTrackingData(
                    player_id=player.player_id,
                    stadium_id=stadium.stadium_id,
                    game_date=game_date,
                    context_year=year,
                    pitch_type=str(row.get("pitch_type", "FF"))[:3],
                    release_speed=float(row["release_speed"]),
                    spin_rate=int(row["release_spin_rate"])
                    if pd.notna(row.get("release_spin_rate"))
                    else None,
                    vertical_break=float(row["pfx_z"] * 12) if pd.notna(row.get("pfx_z")) else None,
                    horizontal_break=float(row["pfx_x"] * 12)
                    if pd.notna(row.get("pfx_x"))
                    else None,
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
                    game_date=game_date,
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
    log_filter_step("mlb_statcast_insert", len(statcast_df.head(500)), inserted)
    get_logger().info(
        f"mlb_seed_complete players={len(players_by_name)} tracking_rows={inserted} "
        f"baselines={len(LEAGUE_BASELINES)}"
    )
    return {
        "mlb_players": len(players_by_name),
        "mlb_tracking_rows": inserted,
        "baselines": len(LEAGUE_BASELINES),
    }
