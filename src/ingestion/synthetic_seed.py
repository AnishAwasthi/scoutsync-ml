"""Synthetic NCAA/Cape Cod amateur tracking data generator."""

from datetime import date, timedelta
import random

import numpy as np
import pandas as pd
from sqlalchemy.orm import Session

from db.models import League, Player, RawTrackingData, StadiumEnvironment
from src.logging_config import get_logger, log_filter_step

RNG = np.random.default_rng(42)
FIRST_NAMES = ["Alex", "Jordan", "Casey", "Riley", "Morgan", "Taylor", "Drew", "Quinn"]
LAST_NAMES = ["Smith", "Johnson", "Williams", "Brown", "Davis", "Miller", "Wilson", "Moore"]


def _random_birth_date() -> date:
    year = RNG.integers(1998, 2005)
    month = int(RNG.integers(1, 13))
    day = int(RNG.integers(1, 29))
    return date(int(year), month, day)


def seed_synthetic_data(session: Session, num_players: int = 50, lite: bool = False) -> dict:
    """Insert synthetic players, stadiums, and tracking rows."""
    logger = get_logger()
    if lite:
        num_players = min(num_players, 10)
        logger.info(f"synthetic_seed_lite num_players={num_players}")
    leagues = {l.abbreviation: l for l in session.query(League).all()}
    ncaa = leagues.get("NCAA")
    ccl = leagues.get("CCL")
    if not ncaa or not ccl:
        raise RuntimeError("Reference leagues NCAA and CCL must exist. Run init-db first.")

    stadiums = []
    for league, name, alt in [
        (ncaa, "Rosenblatt Field", 1100),
        (ncaa, "Baum-Walker Stadium", 400),
        (ccl, "Eldredge Park", 50),
        (ccl, "Spillane Field", 30),
    ]:
        stadium = StadiumEnvironment(
            league_id=league.league_id,
            stadium_name=name,
            altitude=alt,
            park_factor_hr=float(RNG.uniform(0.85, 1.15)),
            park_factor_obp=float(RNG.uniform(0.90, 1.10)),
            temperature_mean=float(RNG.uniform(18, 32)),
            humidity_mean=float(RNG.uniform(35, 75)),
        )
        session.add(stadium)
        stadiums.append(stadium)
    session.flush()

    players = []
    for i in range(num_players):
        p = Player(
            first_name=FIRST_NAMES[i % len(FIRST_NAMES)],
            last_name=f"{LAST_NAMES[i % len(LAST_NAMES)]}{i}",
            birth_date=_random_birth_date(),
            throws=random.choice(["L", "R"]),
            bats=random.choice(["L", "R", "S"]),
            primary_position=random.choice(["P", "SS", "OF", "1B", "C"]),
        )
        session.add(p)
        players.append(p)
    session.flush()

    tracking_count = 0
    context_year = 2023
    for idx, player in enumerate(players):
        league = ncaa if idx % 2 == 0 else ccl
        stadium = RNG.choice([s for s in stadiums if s.league_id == league.league_id])
        is_pitcher = player.primary_position == "P"

        if lite:
            n_pitch = int(RNG.integers(12, 22)) if is_pitcher else int(RNG.integers(6, 12))
            n_hit = int(RNG.integers(10, 18)) if not is_pitcher else int(RNG.integers(4, 8))
        else:
            n_pitch = int(RNG.integers(80, 120)) if is_pitcher else int(RNG.integers(20, 40))
            n_hit = int(RNG.integers(30, 60)) if not is_pitcher else int(RNG.integers(5, 15))

        base_date = date(context_year, 3, 1) + timedelta(days=int(RNG.integers(0, 120)))

        for j in range(n_pitch):
            session.add(
                RawTrackingData(
                    player_id=player.player_id,
                    stadium_id=stadium.stadium_id,
                    game_date=base_date + timedelta(days=j % 30),
                    context_year=context_year,
                    pitch_type=random.choice(["FF", "SL", "CH", "CU"]),
                    release_speed=float(RNG.normal(88 if league.abbreviation == "NCAA" else 90, 3)),
                    spin_rate=int(RNG.integers(1800, 2600)),
                    vertical_break=float(RNG.normal(16, 4)),
                    horizontal_break=float(RNG.normal(-8, 3)),
                    vertical_approach_angle=float(RNG.normal(-5.5, 1.2)),
                    extension=float(RNG.uniform(5.5, 6.8)),
                    plate_x=float(RNG.normal(0, 0.6)),
                    plate_z=float(RNG.normal(2.5, 0.5)),
                )
            )
            tracking_count += 1

        for j in range(n_hit):
            session.add(
                RawTrackingData(
                    player_id=player.player_id,
                    stadium_id=stadium.stadium_id,
                    game_date=base_date + timedelta(days=j % 30),
                    context_year=context_year,
                    exit_velocity=float(RNG.normal(92 if league.abbreviation == "NCAA" else 89, 5)),
                    launch_angle=float(RNG.normal(18, 12)),
                    hit_distance=int(RNG.integers(180, 420)),
                    plate_x=float(RNG.normal(0, 0.5)),
                    plate_z=float(RNG.normal(2.4, 0.4)),
                    is_strikeout=bool(RNG.random() < 0.22),
                    is_walk=bool(RNG.random() < 0.08),
                    is_chase=bool(RNG.random() < 0.28),
                )
            )
            tracking_count += 1

    session.commit()
    log_filter_step("synthetic_tracking_insert", 0, tracking_count)
    logger.info(
        f"synthetic_seed_complete players={num_players} tracking_rows={tracking_count} "
        f"stadiums={len(stadiums)}"
    )
    return {
        "players": num_players,
        "tracking_rows": tracking_count,
        "player_ids": [p.player_id for p in players],
        "league_ids": {"NCAA": ncaa.league_id, "CCL": ccl.league_id},
    }
