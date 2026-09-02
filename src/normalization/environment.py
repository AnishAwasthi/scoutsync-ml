"""
Environmental normalization for tracking metrics.

The physics, and what follows from it:

* **Release speed is not an air-density effect.** Statcast reports ``release_speed`` out
  of the hand, and how fast a pitcher's arm is does not depend on the air. Drag acts on
  the ball *after* release, so the ball arrives at the plate slightly faster in thin air
  — a baseball loses about 8% of its speed at Coors versus 10% at Fenway, roughly 1 mph.
  That is a plate-speed effect on a release-speed measurement, and it is tiny. Release
  speed is therefore passed through unchanged.

* **Break is the large, real altitude effect.** Movement comes from the Magnus force,
  which scales with air density. Coors sits at roughly 82% of sea-level density, so an
  identically thrown pitch breaks about 82% as much — 14-15 inches where Fenway would
  give 18. Normalizing to a reference density means scaling break *up* in thin air, which
  is what ``adjust_break`` does.

* **Exit velocity is not an air-density effect either.** It is measured off the bat; the
  bat-ball collision does not care about the surrounding air.

* **Carry is.** Reduced drag lets batted balls travel about 5% farther at Coors than at
  Fenway for the same contact. That is the effect the "420 feet at altitude" intuition is
  actually about, and it applies to distance, not to exit velocity.

Sources: Alan Nathan, "Baseball At High Altitude" (baseball.physics.illinois.edu/Denver.html).

An earlier version of this module scaled *release speed* by an altitude term (upward at
altitude, compounding the bias rather than removing it), left exit velocity alone
entirely, and computed air density at a fixed sea-level pressure — so density never
varied with altitude at all, and the only thing moving it was a humidity term that was
roughly 38x too strong. Density now uses the barometric formula and a proper
vapour-pressure humidity correction, and reproduces the published Coors ratio (0.83
against a reported 0.82).
"""

import math

import numpy as np
import pandas as pd

from src.config import get_settings
from src.logging_config import get_logger, log_env_batch

FEET_PER_METER = 0.3048

# Barometric formula constants for the troposphere (ISA).
SEA_LEVEL_PRESSURE_PA = 101325.0
BAROMETRIC_LAPSE = 2.25577e-5
BAROMETRIC_EXPONENT = 5.25588

# Tetens coefficients for saturation vapour pressure over water.
TETENS_A = 610.78
TETENS_B = 17.27
TETENS_C = 237.3


def saturation_vapor_pressure_pa(temperature_c: float) -> float:
    """Saturation vapour pressure over water (Tetens equation), in pascals."""
    return TETENS_A * math.exp(TETENS_B * temperature_c / (temperature_c + TETENS_C))


def pressure_at_altitude_pa(altitude_ft: float) -> float:
    """Ambient pressure from the ISA barometric formula."""
    altitude_m = altitude_ft * FEET_PER_METER
    return SEA_LEVEL_PRESSURE_PA * (1.0 - BAROMETRIC_LAPSE * altitude_m) ** BAROMETRIC_EXPONENT


def air_density_kg_m3(
    temperature_c: float,
    humidity_pct: float,
    altitude_ft: float = 0.0,
) -> float:
    """
    Moist-air density from partial pressures of dry air and water vapour.

    ``altitude_ft`` defaults to sea level. Humid air is *less* dense than dry air because
    water vapour (18 g/mol) is lighter than dry air (29 g/mol), but the effect is about
    1% from bone dry to saturated — far smaller than the ~18% altitude effect at Coors.
    """
    pressure = pressure_at_altitude_pa(altitude_ft)
    vapor_pressure = (humidity_pct / 100.0) * saturation_vapor_pressure_pa(temperature_c)
    vapor_pressure = min(vapor_pressure, pressure)
    dry_pressure = pressure - vapor_pressure

    settings = get_settings()
    temperature_k = temperature_c + 273.15
    return float(
        (dry_pressure * settings.m_dry + vapor_pressure * settings.m_vapor)
        / (settings.gas_constant * temperature_k)
    )


def reference_air_density() -> float:
    """Density at the configured reference park, the baseline everything normalizes to."""
    settings = get_settings()
    return air_density_kg_m3(
        settings.ref_temperature_c, settings.ref_humidity_pct, settings.alt_std_ft
    )


def adjust_release_speed(raw_velocity: pd.Series) -> pd.Series:
    """
    Release speed needs no environmental correction — see the module docstring.

    Kept as an explicit identity rather than silently omitted so the pipeline documents
    the decision at the point a reader would look for the adjustment.
    """
    return raw_velocity.astype(float)


def adjust_break(raw_break: pd.Series, rho_stadium: pd.Series) -> pd.Series:
    """Scale break to reference density. Magnus force is proportional to air density."""
    ratio = reference_air_density() / rho_stadium.replace(0, np.nan)
    return raw_break * ratio.fillna(1.0)


def carry_factor(rho_stadium: pd.Series) -> pd.Series:
    """
    How much farther a batted ball carries at this park than at the reference park.

    Linearized around the reference density: a ~18% density drop (sea level to Coors)
    buys about 5% of carry, so the sensitivity is 0.05 / 0.18 ≈ 0.28.
    """
    rho_ref = reference_air_density()
    sensitivity = get_settings().carry_density_sensitivity
    return 1.0 + sensitivity * (1.0 - rho_stadium / rho_ref)


def adjust_hit_distance(raw_distance: pd.Series, rho_stadium: pd.Series) -> pd.Series:
    """Remove park carry from a batted-ball distance."""
    factor = carry_factor(rho_stadium).replace(0, np.nan)
    return raw_distance.astype(float) / factor.fillna(1.0)


def apply_environmental_adjustments(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize tracking metrics for altitude, temperature, and humidity.

    Expects columns: release_speed, vertical_break, horizontal_break, spin_rate,
    hit_distance, altitude, temperature_mean, humidity_mean (nullable; imputed).
    """
    settings = get_settings()
    logger = get_logger()
    out = df.copy()
    imputed = 0

    defaults = {
        "altitude": settings.alt_std_ft,
        "temperature_mean": settings.ref_temperature_c,
        "humidity_mean": settings.ref_humidity_pct,
    }
    for col, default in defaults.items():
        if col not in out.columns:
            out[col] = default
            imputed += len(out)
            continue
        missing = out[col].isna()
        if missing.any():
            imputed += int(missing.sum())
            out.loc[missing, col] = default

    out["rho_stadium"] = [
        air_density_kg_m3(float(t), float(h), float(a))
        for t, h, a in zip(
            out["temperature_mean"], out["humidity_mean"], out["altitude"], strict=True
        )
    ]
    out["carry_factor"] = carry_factor(out["rho_stadium"])

    if "release_speed" in out.columns:
        out["adj_velocity"] = adjust_release_speed(out["release_speed"])
    if "vertical_break" in out.columns:
        out["adj_vertical_break"] = adjust_break(out["vertical_break"], out["rho_stadium"])
    if "horizontal_break" in out.columns:
        out["adj_horizontal_break"] = adjust_break(out["horizontal_break"], out["rho_stadium"])
    if "spin_rate" in out.columns:
        # Spin rate is imparted by the hand, not the air; carried through unchanged so
        # downstream code that reads adj_spin_rate stays consistent with adj_velocity.
        out["adj_spin_rate"] = out["spin_rate"].astype(float)
    if "hit_distance" in out.columns:
        out["adj_hit_distance"] = adjust_hit_distance(out["hit_distance"], out["rho_stadium"])

    rho_ref = reference_air_density()
    mean_delta = float((out["altitude"] - settings.alt_std_ft).mean())
    mean_ratio = float((rho_ref / out["rho_stadium"]).mean())
    log_env_batch(mean_delta, mean_ratio, imputed)
    logger.debug(
        f"environmental_adjustment rows={len(out)} imputed={imputed} rho_ref={rho_ref:.4f}"
    )
    return out
