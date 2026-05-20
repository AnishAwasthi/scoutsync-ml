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
    alpha: float = 0.003
    alt_std_ft: int = 500
    rho_std: float = 1.225
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

    # Ideal gas constants for air density (simplified moist air)
    pressure_pa: float = 101325.0
    m_dry: float = 0.02897
    gas_constant: float = 8.314


def _use_sqlite_env() -> bool:
    return os.getenv("USE_SQLITE", "false").lower() == "true"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if _use_sqlite_env():
        return settings.model_copy(update={"database_url": "sqlite:///scoutsync.db"})
    return settings
