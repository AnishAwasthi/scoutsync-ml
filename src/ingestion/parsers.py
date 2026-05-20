"""CSV/JSON parsers for TrackMan-style tracking uploads."""

import json
from datetime import datetime
from io import StringIO
from typing import Any

import pandas as pd

from src.logging_config import get_logger, log_filter_step

TRACKMAN_COLUMN_MAP = {
    "PitcherId": "player_id",
    "player_id": "player_id",
    "GameDate": "game_date",
    "game_date": "game_date",
    "PitchType": "pitch_type",
    "pitch_type": "pitch_type",
    "RelSpeed": "release_speed",
    "release_speed": "release_speed",
    "SpinRate": "spin_rate",
    "spin_rate": "spin_rate",
    "VertBreak": "vertical_break",
    "vertical_break": "vertical_break",
    "HorzBreak": "horizontal_break",
    "horizontal_break": "horizontal_break",
    "VAA": "vertical_approach_angle",
    "vertical_approach_angle": "vertical_approach_angle",
    "Extension": "extension",
    "extension": "extension",
    "PlateLocSide": "plate_x",
    "plate_x": "plate_x",
    "PlateLocHeight": "plate_z",
    "plate_z": "plate_z",
    "ExitSpeed": "exit_velocity",
    "exit_velocity": "exit_velocity",
    "Angle": "launch_angle",
    "launch_angle": "launch_angle",
    "Distance": "hit_distance",
    "hit_distance": "hit_distance",
    "StadiumId": "stadium_id",
    "stadium_id": "stadium_id",
    "ContextYear": "context_year",
    "context_year": "context_year",
}


def parse_trackman_csv(content: str | bytes, default_year: int | None = None) -> pd.DataFrame:
    """Parse TrackMan CSV into normalized column names."""
    if isinstance(content, bytes):
        content = content.decode("utf-8")
    raw = pd.read_csv(StringIO(content))
    before = len(raw)
    renamed = raw.rename(columns={k: v for k, v in TRACKMAN_COLUMN_MAP.items() if k in raw.columns})
    log_filter_step("parse_trackman_csv", before, len(renamed))

    if "game_date" in renamed.columns:
        renamed["game_date"] = pd.to_datetime(renamed["game_date"]).dt.date
    if "context_year" not in renamed.columns:
        renamed["context_year"] = default_year or datetime.now().year

    get_logger().info(f"trackman_csv_parsed columns={list(renamed.columns)}")
    return renamed


def parse_json_tracking(payload: list[dict[str, Any]] | str) -> pd.DataFrame:
    """Parse JSON array of tracking records."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    before = len(payload)
    df = pd.DataFrame(payload)
    log_filter_step("parse_json_tracking", before, len(df))
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.date
    return df


def dataframe_to_tracking_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert DataFrame rows to dicts suitable for ORM insertion."""
    records = []
    for _, row in df.iterrows():
        rec = {
            "player_id": int(row["player_id"]) if pd.notna(row.get("player_id")) else None,
            "stadium_id": int(row["stadium_id"]) if pd.notna(row.get("stadium_id")) else None,
            "game_date": row.get("game_date"),
            "context_year": int(row.get("context_year", datetime.now().year)),
            "pitch_type": row.get("pitch_type"),
            "release_speed": _float_or_none(row.get("release_speed")),
            "spin_rate": _int_or_none(row.get("spin_rate")),
            "vertical_break": _float_or_none(row.get("vertical_break")),
            "horizontal_break": _float_or_none(row.get("horizontal_break")),
            "vertical_approach_angle": _float_or_none(row.get("vertical_approach_angle")),
            "extension": _float_or_none(row.get("extension")),
            "plate_x": _float_or_none(row.get("plate_x")),
            "plate_z": _float_or_none(row.get("plate_z")),
            "exit_velocity": _float_or_none(row.get("exit_velocity")),
            "launch_angle": _float_or_none(row.get("launch_angle")),
            "hit_distance": _int_or_none(row.get("hit_distance")),
            "is_strikeout": _bool_or_none(row.get("is_strikeout")),
            "is_walk": _bool_or_none(row.get("is_walk")),
            "is_chase": _bool_or_none(row.get("is_chase")),
        }
        records.append(rec)
    return records


def _float_or_none(val) -> float | None:
    return None if val is None or (isinstance(val, float) and pd.isna(val)) else float(val)


def _int_or_none(val) -> int | None:
    return None if val is None or (isinstance(val, float) and pd.isna(val)) else int(val)


def _bool_or_none(val) -> bool | None:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return bool(val)
