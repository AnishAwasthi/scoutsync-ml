"""
Normalization physics tests.

These pin the *direction and magnitude* of each environmental correction against
published values, because an earlier version had them wrong in ways that unit tests
asserting only "something changed" would not have caught:

- air density ignored altitude entirely (fixed sea-level pressure), so the one thing
  driving it was a humidity term roughly 38x too strong;
- release speed was scaled *up* at altitude, compounding bias rather than removing it;
- exit velocity had no correction, while carry -- the effect that actually exists -- had
  none either.

Reference: Alan Nathan, "Baseball At High Altitude".
https://baseball.physics.illinois.edu/Denver.html
"""

import pandas as pd
import pytest

from src.config import get_settings
from src.normalization.environment import (
    adjust_break,
    adjust_hit_distance,
    adjust_release_speed,
    air_density_kg_m3,
    apply_environmental_adjustments,
    carry_factor,
    pressure_at_altitude_pa,
    reference_air_density,
)
from src.normalization.league_context import gamma_for_tier, normalize_metric

COORS_ALTITUDE_FT = 5200
SEA_LEVEL_FT = 0


def test_air_density_is_physically_plausible_at_sea_level():
    rho = air_density_kg_m3(22.0, 50.0, SEA_LEVEL_FT)
    assert 1.15 < rho < 1.25


def test_air_density_falls_with_altitude():
    sea_level = air_density_kg_m3(22.0, 50.0, SEA_LEVEL_FT)
    coors = air_density_kg_m3(22.0, 50.0, COORS_ALTITUDE_FT)
    assert coors < sea_level


def test_coors_density_ratio_matches_published_value():
    """Coors sits at roughly 82% of sea-level air density."""
    ratio = air_density_kg_m3(22.0, 50.0, COORS_ALTITUDE_FT) / air_density_kg_m3(
        22.0, 50.0, SEA_LEVEL_FT
    )
    assert ratio == pytest.approx(0.82, abs=0.02)


def test_humid_air_is_slightly_less_dense_than_dry_air():
    """
    Water vapour (18 g/mol) is lighter than dry air (29 g/mol), so humidity lowers
    density -- but only by about 1% from bone dry to saturated, not tens of percent.
    """
    dry = air_density_kg_m3(22.0, 0.0, SEA_LEVEL_FT)
    saturated = air_density_kg_m3(22.0, 100.0, SEA_LEVEL_FT)
    assert saturated < dry
    assert (1.0 - saturated / dry) < 0.03


def test_warm_air_is_less_dense_than_cold_air():
    assert air_density_kg_m3(35.0, 50.0, SEA_LEVEL_FT) < air_density_kg_m3(5.0, 50.0, SEA_LEVEL_FT)


def test_pressure_decreases_with_altitude():
    assert pressure_at_altitude_pa(COORS_ALTITUDE_FT) < pressure_at_altitude_pa(SEA_LEVEL_FT)


def test_release_speed_is_not_environmentally_adjusted():
    """
    Release speed is measured out of the hand. Air density acts on the ball after
    release, so there is nothing for a park correction to remove.
    """
    raw = pd.Series([88.0, 94.5, 101.2])
    pd.testing.assert_series_equal(adjust_release_speed(raw), raw.astype(float))


def test_break_is_scaled_up_in_thin_air():
    """
    Magnus force scales with air density, so an identical pitch breaks *less* at
    altitude. Normalizing to a reference density must scale the observed break up.
    """
    raw = pd.Series([18.0])
    thin = pd.Series([reference_air_density() * 0.82])
    adjusted = adjust_break(raw, thin)
    assert adjusted.iloc[0] > raw.iloc[0]
    # 18 inches at sea level -> ~14.8 observed at Coors -> restored to ~18.
    assert adjusted.iloc[0] == pytest.approx(18.0 / 0.82, rel=0.01)


def test_break_is_unchanged_at_reference_density():
    raw = pd.Series([18.0])
    adjusted = adjust_break(raw, pd.Series([reference_air_density()]))
    assert adjusted.iloc[0] == pytest.approx(18.0)


def test_carry_factor_exceeds_one_in_thin_air():
    """Reduced drag lets a batted ball travel about 5% farther at Coors."""
    thin = pd.Series([reference_air_density() * 0.82])
    factor = carry_factor(thin).iloc[0]
    assert factor > 1.0
    assert factor == pytest.approx(1.05, abs=0.01)


def test_hit_distance_is_deflated_in_thin_air():
    """A 420-foot drive at altitude was not a 420-foot drive at the reference park."""
    raw = pd.Series([420.0])
    thin = pd.Series([reference_air_density() * 0.82])
    assert adjust_hit_distance(raw, thin).iloc[0] < 420.0


def test_apply_environmental_adjustments_produces_expected_columns():
    df = pd.DataFrame(
        {
            "release_speed": [92.0],
            "vertical_break": [18.0],
            "horizontal_break": [-6.0],
            "spin_rate": [2200],
            "exit_velocity": [98.0],
            "hit_distance": [400],
            "altitude": [COORS_ALTITUDE_FT],
            "temperature_mean": [25.0],
            "humidity_mean": [60.0],
        }
    )
    out = apply_environmental_adjustments(df)

    for column in ("rho_stadium", "carry_factor", "adj_velocity", "adj_vertical_break"):
        assert column in out.columns

    # Velocity passes through; break is restored upward; carry is removed downward.
    assert out["adj_velocity"].iloc[0] == pytest.approx(92.0)
    assert out["adj_vertical_break"].iloc[0] > 18.0
    assert out["adj_hit_distance"].iloc[0] < 400.0


def test_environmental_adjustment_imputes_missing_park_data():
    settings = get_settings()
    df = pd.DataFrame({"release_speed": [92.0], "vertical_break": [18.0]})
    out = apply_environmental_adjustments(df)
    assert out["altitude"].iloc[0] == settings.alt_std_ft
    assert out["rho_stadium"].iloc[0] == pytest.approx(reference_air_density())


def test_league_normalize_metric():
    values = pd.Series([95.0, 100.0])
    normed = normalize_metric(values, mu=90.0, sigma=5.0, tier=4)
    expected_first = ((95.0 - 90.0) / 5.0) * gamma_for_tier(4)
    assert normed.iloc[0] == pytest.approx(expected_first)


def test_gamma_tier_ordering():
    assert gamma_for_tier(1) > gamma_for_tier(5)
