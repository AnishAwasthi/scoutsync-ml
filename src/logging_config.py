"""Structured logging for ScoutSync ML."""

import logging
import sys
from typing import Any

from src.config import get_settings

LOGGER_NAME = "scoutsync"


def setup_logging() -> logging.Logger:
    settings = get_settings()
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return logger

    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    logger.addHandler(handler)
    return logger


def get_logger() -> logging.Logger:
    return setup_logging()


def log_filter_step(name: str, before: int, after: int, **kwargs: Any) -> None:
    dropped = before - after
    extras = " ".join(f"{k}={v}" for k, v in kwargs.items())
    get_logger().info(
        f"filter_step={name} rows_before={before} rows_after={after} rows_dropped={dropped} {extras}".strip()
    )


def log_variance_params(
    player_id: int,
    sigma: dict[str, float],
    ci_bounds: dict[str, tuple[float, float]],
) -> None:
    get_logger().info(
        f"variance_generation player_id={player_id} sigma={sigma} ci_bounds={ci_bounds}"
    )


def log_env_batch(
    mean_delta_alt: float,
    mean_density_ratio: float,
    imputed_count: int,
) -> None:
    get_logger().info(
        f"env_normalization mean_delta_alt={mean_delta_alt:.2f} "
        f"mean_density_ratio={mean_density_ratio:.4f} imputed_env_fields={imputed_count}"
    )
