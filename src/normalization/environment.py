"""Environmental physical vector adjustment for tracking metrics."""

import numpy as np
import pandas as pd

from src.config import get_settings
from src.logging_config import get_logger, log_env_batch


def air_density_kg_m3(temperature_c: float, humidity_pct: float) -> float:
    """Simplified ideal-gas moist air density (kg/m^3)."""
    settings = get_settings()
    t_k = temperature_c + 273.15
    rho = (settings.pressure_pa * settings.m_dry) / (settings.gas_constant * t_k)
    rho *= 1.0 - 0.378 * (humidity_pct / 100.0)
    return float(rho)


def adjust_velocity(raw_velocity: pd.Series, altitude_ft: pd.Series) -> pd.Series:
    settings = get_settings()
    delta_alt = altitude_ft - settings.alt_std_ft
    scalar = 1.0 + settings.alpha * (delta_alt / 1000.0)
    return raw_velocity * scalar


def adjust_break(raw_break: pd.Series, rho_stadium: pd.Series) -> pd.Series:
    settings = get_settings()
    ratio = settings.rho_std / rho_stadium.replace(0, np.nan)
    return raw_break * ratio.fillna(1.0)


def apply_environmental_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize tracking metrics for altitude, temperature, and humidity.

    Expects columns: release_speed, vertical_break, horizontal_break, spin_rate,
    altitude, temperature_mean, humidity_mean (nullable; imputed when missing).
    """
    logger = get_logger()
    out = df.copy()
    imputed = 0

    if "altitude" not in out.columns:
        out["altitude"] = get_settings().alt_std_ft
        imputed += len(out)
    if "temperature_mean" not in out.columns:
        out["temperature_mean"] = 22.0
        imputed += len(out)
    if "humidity_mean" not in out.columns:
        out["humidity_mean"] = 50.0
        imputed += len(out)

    for col in ("altitude", "temperature_mean", "humidity_mean"):
        null_mask = out[col].isna()
        if null_mask.any():
            imputed += int(null_mask.sum())
            if col == "altitude":
                out.loc[null_mask, col] = get_settings().alt_std_ft
            elif col == "temperature_mean":
                out.loc[null_mask, col] = 22.0
            else:
                out.loc[null_mask, col] = 50.0

    out["rho_stadium"] = out.apply(
        lambda r: air_density_kg_m3(float(r["temperature_mean"]), float(r["humidity_mean"])),
        axis=1,
    )
    out["altitude_scalar"] = 1.0 + get_settings().alpha * (
        (out["altitude"] - get_settings().alt_std_ft) / 1000.0
    )

    if "release_speed" in out.columns:
        out["adj_velocity"] = adjust_velocity(out["release_speed"], out["altitude"])
    if "vertical_break" in out.columns:
        out["adj_vertical_break"] = adjust_break(out["vertical_break"], out["rho_stadium"])
    if "horizontal_break" in out.columns:
        out["adj_horizontal_break"] = adjust_break(out["horizontal_break"], out["rho_stadium"])
    if "spin_rate" in out.columns:
        out["adj_spin_rate"] = adjust_break(
            out["spin_rate"].astype(float), out["rho_stadium"]
        )

    mean_delta = float((out["altitude"] - get_settings().alt_std_ft).mean())
    mean_ratio = float((get_settings().rho_std / out["rho_stadium"]).mean())
    log_env_batch(mean_delta, mean_ratio, imputed)
    logger.debug(f"environmental_adjustment rows={len(out)} imputed={imputed}")
    return out
