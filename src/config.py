"""Application configuration."""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql://scoutsync:scoutsync@localhost:5432/scoutsync"
    use_sqlite: bool = False
    log_level: str = "INFO"

    # Reference park that every metric is normalized to. Air density at these conditions
    # is the denominator for the break and carry adjustments.
    alt_std_ft: int = 500
    ref_temperature_c: float = 22.0
    ref_humidity_pct: float = 50.0

    # Carry sensitivity to air density: a ~18% density drop (sea level -> Coors) buys
    # about 5% of batted-ball distance, so 0.05 / 0.18 ~= 0.28.
    carry_density_sensitivity: float = 0.28

    validation_year: int = 2024
    pybaseball_cache_dir: str = str(PROJECT_ROOT / "data" / "pybaseball_cache")
    model_dir: str = str(PROJECT_ROOT / "data" / "models")

    gamma_tier_map: dict[int, float] = {
        1: 1.0,
        2: 0.85,
        3: 0.75,
        4: 0.55,
        5: 0.45,
    }

    # Molar masses (kg/mol) and the universal gas constant, for moist-air density.
    m_dry: float = 0.0289652
    m_vapor: float = 0.018016
    gas_constant: float = 8.31446


def _use_sqlite_env() -> bool:
    return os.getenv("USE_SQLITE", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if _use_sqlite_env():
        return settings.model_copy(update={"database_url": "sqlite:///scoutsync.db"})
    return settings
