"""Feature matrix construction for pitcher and batter models."""

from datetime import date

import pandas as pd

from src.logging_config import get_logger, log_filter_step

PITCHER_FEATURES = [
    "adj_velocity",
    "spin_rate",
    "vertical_break",
    "horizontal_break",
    "vaa",
    "extension",
    "strike_zone_command_rate",
    "age_relative_to_league",
    "tier_coefficient",
]

BATTER_FEATURES = [
    "max_exit_velocity",
    "pct_90th_exit_velocity",
    "launch_angle_sweetspot_rate",
    "zone_contact_rate",
    "out_of_zone_chase_rate",
    "age_relative_to_league",
    "conference_strength_factor",
]


def _player_age_at_season(birth_date: date, context_year: int) -> float:
    return context_year - birth_date.year - (
        1 if (birth_date.month, birth_date.day) > (7, 1) else 0
    )


def build_pitcher_features(df: pd.DataFrame, league_median_age: float = 21.0) -> pd.DataFrame:
    """Aggregate pitch-level rows to player-season pitcher features."""
    pitch_df = df[df["release_speed"].notna()].copy()
    log_filter_step("pitcher_pitch_rows", len(df), len(pitch_df))

    if pitch_df.empty:
        return pd.DataFrame()

    in_zone = (
        pitch_df["plate_x"].abs().le(0.83)
        & pitch_df["plate_z"].between(1.5, 3.5, inclusive="both")
    )

    grouped = pitch_df.groupby(["player_id", "context_year"])
    rows = []
    for (player_id, year), grp in grouped:
        rows.append(
            {
                "player_id": player_id,
                "context_year": year,
                "role": "pitcher",
                "adj_velocity": grp["adj_velocity"].mean(),
                "spin_rate": grp.get("adj_spin_rate", grp["spin_rate"]).mean(),
                "vertical_break": grp.get("adj_vertical_break", grp["vertical_break"]).mean(),
                "horizontal_break": grp.get(
                    "adj_horizontal_break", grp["horizontal_break"]
                ).mean(),
                "vaa": grp["vertical_approach_angle"].mean(),
                "extension": grp["extension"].mean(),
                "strike_zone_command_rate": float(in_zone.loc[grp.index].mean()),
                "age_relative_to_league": grp["age_at_season"].iloc[0] - league_median_age,
                "tier_coefficient": grp["competition_tier"].iloc[0],
            }
        )
    result = pd.DataFrame(rows)
    get_logger().info(f"pitcher_features built rows={len(result)}")
    return result


def build_batter_features(df: pd.DataFrame, league_median_age: float = 21.0) -> pd.DataFrame:
    """Aggregate batted-ball rows to player-season batter features."""
    hit_df = df[df["exit_velocity"].notna()].copy()
    log_filter_step("batter_hit_rows", len(df), len(hit_df))

    if hit_df.empty:
        return pd.DataFrame()

    sweet = hit_df["launch_angle"].between(8, 32, inclusive="both")
    in_zone = hit_df["plate_x"].abs().le(0.83) & hit_df["plate_z"].between(1.5, 3.5)
    chase = hit_df["is_chase"].fillna(False)

    grouped = hit_df.groupby(["player_id", "context_year"])
    rows = []
    for (player_id, year), grp in grouped:
        # Prefer the park-adjusted series so altitude bias does not reach the model.
        ev = grp["adj_exit_velocity"] if "adj_exit_velocity" in grp.columns else grp["exit_velocity"]
        rows.append(
            {
                "player_id": player_id,
                "context_year": year,
                "role": "batter",
                "max_exit_velocity": ev.max(),
                "pct_90th_exit_velocity": float(ev.quantile(0.9)),
                "launch_angle_sweetspot_rate": float(sweet.loc[grp.index].mean()),
                "zone_contact_rate": float(
                    (
                        ~grp["is_strikeout"].fillna(True).astype(bool)
                        & in_zone.loc[grp.index].astype(bool)
                    ).mean()
                ),
                "out_of_zone_chase_rate": float(chase.loc[grp.index].mean()),
                "age_relative_to_league": grp["age_at_season"].iloc[0] - league_median_age,
                "conference_strength_factor": float(
                    (5.0 - grp["base_run_environment"].iloc[0]) / 2.0
                ),
            }
        )
    result = pd.DataFrame(rows)
    get_logger().info(f"batter_features built rows={len(result)}")
    return result


def enrich_tracking_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add age_at_season and birth_year helpers."""
    out = df.copy()
    if "birth_date" in out.columns:
        out["age_at_season"] = out.apply(
            lambda r: _player_age_at_season(r["birth_date"], int(r["context_year"])),
            axis=1,
        )
    return out


def build_training_frames(
    normalized_df: pd.DataFrame,
    labels_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Build per-role feature matrices.

    Returns ``(combined, pitchers, batters, PITCHER_FEATURES, BATTER_FEATURES)``.
    """
    norm = enrich_tracking_dataframe(normalized_df)
    league_median_age = float(norm["age_at_season"].median()) if "age_at_season" in norm else 21.0

    pitchers = build_pitcher_features(norm, league_median_age)
    batters = build_batter_features(norm, league_median_age)

    if not labels_df.empty:
        pitchers = pitchers.merge(
            labels_df,
            on=["player_id", "context_year"],
            how="left",
            suffixes=("", "_label"),
        )
        batters = batters.merge(
            labels_df,
            on=["player_id", "context_year"],
            how="left",
            suffixes=("", "_label"),
        )

    combined = pd.concat([pitchers, batters], ignore_index=True)
    return combined, pitchers, batters, PITCHER_FEATURES, BATTER_FEATURES
