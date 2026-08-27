"""Unit tests for normalization math (no network)."""

import pandas as pd

from src.config import get_settings
from src.normalization.environment import (
    adjust_velocity,
    air_density_kg_m3,
    apply_environmental_adjustments,
)
from src.normalization.league_context import gamma_for_tier, normalize_metric


def test_air_density_positive():
    rho = air_density_kg_m3(22.0, 50.0)
    assert 0.9 < rho < 1.4


def test_adjust_velocity_altitude_effect():
    settings = get_settings()
    raw = pd.Series([90.0, 90.0])
    low = pd.Series([settings.alt_std_ft, settings.alt_std_ft])
    high = pd.Series([settings.alt_std_ft + 1000, settings.alt_std_ft + 1000])
    base = adjust_velocity(raw, low).iloc[0]
    elevated = adjust_velocity(raw, high).iloc[0]
    assert elevated > base


def test_environmental_adjustments_columns():
    df = pd.DataFrame(
        {
            "release_speed": [92.0],
            "vertical_break": [18.0],
            "horizontal_break": [-6.0],
            "spin_rate": [2200],
            "altitude": [1000],
            "temperature_mean": [25.0],
            "humidity_mean": [60.0],
        }
    )
    out = apply_environmental_adjustments(df)
    assert "adj_velocity" in out.columns
    assert "rho_stadium" in out.columns
    assert out["adj_velocity"].iloc[0] > 92.0


def test_league_normalize_metric():
    values = pd.Series([95.0, 100.0])
    normed = normalize_metric(values, mu=90.0, sigma=5.0, tier=4)
    gamma = gamma_for_tier(4)
    expected_first = ((95.0 - 90.0) / 5.0) * gamma
    assert abs(normed.iloc[0] - expected_first) < 1e-6


def test_gamma_tier_ordering():
    assert gamma_for_tier(1) > gamma_for_tier(5)
