"""pybaseball Statcast ingestion and rookie outcome tables."""

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import League, Player, RawTrackingData, StadiumEnvironment
from src.config import PROJECT_ROOT, get_settings
from src.logging_config import get_logger, log_filter_step
from src.runtime import is_streamlit_cloud

# Module-level caches for baselines and rookie outcomes
LEAGUE_BASELINES: pd.DataFrame = pd.DataFrame()
ROOKIE_OUTCOMES: pd.DataFrame = pd.DataFrame()
PSEUDO_ROOKIE_MAP: pd.DataFrame = pd.DataFrame()


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
    logger.warning(f"statcast_lite_csv missing path={path}; using tiny synthetic")
    return _synthetic_mlb_statcast(2023, 2024, max_rows=40)


def load_rookie_outcomes_lite(validation_year: int) -> pd.DataFrame:
    """Small synthetic rookie table for cloud backtest labels."""
    return _synthetic_rookie_outcomes(validation_year, n_batters=6, n_pitchers=4)


def load_statcast_sample(start_year: int, end_year: int, max_rows: int = 8000) -> pd.DataFrame:
    """
    Load MLB Statcast data via pybaseball with local CSV cache.

    Falls back to synthetic MLB-like data if pybaseball/network unavailable.
    On Streamlit Cloud, never calls pybaseball (offline lite CSV only).
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
    for src, metric in mappings:
        if src not in statcast_df.columns:
            continue
        series = statcast_df[src].dropna()
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


def load_rookie_outcomes(validation_year: int) -> pd.DataFrame:
    """Build rookie wOBA/ERA table for backtest labels."""
    global ROOKIE_OUTCOMES
    if is_streamlit_cloud():
        ROOKIE_OUTCOMES = load_rookie_outcomes_lite(validation_year)
        return ROOKIE_OUTCOMES

    cache = _ensure_cache_dir() / f"rookies_{validation_year}.csv"
    if cache.exists():
        ROOKIE_OUTCOMES = pd.read_csv(cache)
        return ROOKIE_OUTCOMES

    try:
        from pybaseball import batting_stats, pitching_stats

        bat = batting_stats(validation_year, qual=0)
        pit = pitching_stats(validation_year, qual=0)
        bat_rookies = bat[bat["PA"] < 130].copy() if "PA" in bat.columns else bat.head(50)
        pit_rookies = pit[pit["IP"] < 50].copy() if "ID" in pit.columns else pit.head(30)

        rows = []
        for _, r in bat_rookies.iterrows():
            woba_col = "wOBA" if "wOBA" in r else "OBP"
            rows.append(
                {
                    "mlb_player_key": str(r.get("ID", r.get("Name", ""))),
                    "role": "batter",
                    "true_wOBA": float(r.get(woba_col, 0.31)),
                    "true_ERA": np.nan,
                    "context_year": validation_year - 1,
                }
            )
        for _, r in pit_rookies.iterrows():
            rows.append(
                {
                    "mlb_player_key": str(r.get("ID", r.get("Name", ""))),
                    "role": "pitcher",
                    "true_wOBA": np.nan,
                    "true_ERA": float(r.get("ERA", 4.50)),
                    "context_year": validation_year - 1,
                }
            )
        ROOKIE_OUTCOMES = pd.DataFrame(rows)
    except Exception as exc:
        get_logger().warning(f"rookie_outcomes_fallback reason={exc}")
        ROOKIE_OUTCOMES = _synthetic_rookie_outcomes(validation_year)

    ROOKIE_OUTCOMES.to_csv(cache, index=False)
    return ROOKIE_OUTCOMES


def _synthetic_rookie_outcomes(
    validation_year: int,
    n_batters: int = 50,
    n_pitchers: int = 30,
) -> pd.DataFrame:
    rng = np.random.default_rng(99)
    n = n_batters + n_pitchers
    roles = ["batter"] * n_batters + ["pitcher"] * n_pitchers
    return pd.DataFrame(
        {
            "mlb_player_key": [f"mlb_{i}" for i in range(n)],
            "role": roles,
            "true_wOBA": np.where(
                np.array(roles) == "batter",
                rng.normal(0.31, 0.04, n),
                np.nan,
            ),
            "true_ERA": np.where(
                np.array(roles) == "pitcher",
                rng.normal(4.2, 0.8, n),
                np.nan,
            ),
            "context_year": validation_year - 1,
        }
    )


def build_pseudo_rookie_map(amateur_player_ids: list[int], rookie_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic amateur -> MLB rookie mapping for backtest."""
    global PSEUDO_ROOKIE_MAP
    if rookie_df.empty:
        return pd.DataFrame()
    keys = rookie_df["mlb_player_key"].tolist()
    rows = []
    for i, pid in enumerate(amateur_player_ids):
        key = keys[i % len(keys)]
        row = rookie_df[rookie_df["mlb_player_key"] == key].iloc[0]
        rows.append(
            {
                "player_id": pid,
                "mlb_player_key": key,
                "role": row["role"],
                "true_wOBA": row.get("true_wOBA"),
                "true_ERA": row.get("true_ERA"),
                "context_year": int(row.get("context_year", 2023)),
            }
        )
    PSEUDO_ROOKIE_MAP = pd.DataFrame(rows)
    return PSEUDO_ROOKIE_MAP


def seed_mlb_statcast(session: Session, validation_year: int | None = None) -> dict:
    """Load Statcast sample into DB and populate global baselines."""
    if is_streamlit_cloud():
        from src.ingestion.lite_seed import seed_mlb_from_offline_csv

        return seed_mlb_from_offline_csv(session)

    global LEAGUE_BASELINES
    settings = get_settings()
    validation_year = validation_year or settings.validation_year
    start_year = validation_year - 3

    statcast_df = load_statcast_sample(start_year, validation_year, max_rows=6000)
    LEAGUE_BASELINES = compute_league_baselines_from_statcast(statcast_df)
    load_rookie_outcomes(validation_year)

    mlb = session.query(League).filter_by(abbreviation="MLB").first()
    if not mlb:
        raise RuntimeError("MLB league row missing")

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
    sample = statcast_df.head(500)
    for i, row in sample.iterrows():
        name = str(row.get("player_name", f"MLB {i}"))
        parts = name.split()
        player = Player(
            first_name=parts[0] if parts else "MLB",
            last_name=parts[-1] if len(parts) > 1 else str(i),
            birth_date=date(1995, 1, 1),
            throws="R",
            bats="R",
            primary_position="P" if "pitcher" in str(row).lower() else "OF",
        )
        session.add(player)
        session.flush()

        year = int(row.get("game_year", validation_year))
        gd = row.get("game_date", date(year, 6, 1))
        if hasattr(gd, "date"):
            gd = gd.date()
        elif isinstance(gd, str):
            gd = pd.to_datetime(gd).date()

        if pd.notna(row.get("release_speed")):
            session.add(
                RawTrackingData(
                    player_id=player.player_id,
                    stadium_id=stadium.stadium_id,
                    game_date=gd,
                    context_year=year,
                    pitch_type=str(row.get("pitch_type", "FF"))[:3],
                    release_speed=float(row["release_speed"]),
                    spin_rate=int(row["release_spin_rate"]) if pd.notna(row.get("release_spin_rate")) else None,
                    vertical_break=float(row["pfx_z"] * 12) if pd.notna(row.get("pfx_z")) else None,
                    horizontal_break=float(row["pfx_x"] * 12) if pd.notna(row.get("pfx_x")) else None,
                    vertical_approach_angle=-5.0,
                    extension=6.0,
                    plate_x=float(np.random.normal(0, 0.5)),
                    plate_z=float(np.random.normal(2.5, 0.4)),
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
    log_filter_step("mlb_statcast_insert", len(sample), inserted)
    get_logger().info(f"mlb_seed_complete tracking_rows={inserted} baselines={len(LEAGUE_BASELINES)}")
    return {"mlb_tracking_rows": inserted, "baselines": len(LEAGUE_BASELINES)}
