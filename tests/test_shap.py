"""SHAP payload tests. The sign of a contribution is the product feature, so it is tested."""

import numpy as np
import pandas as pd

from src.ml.shap_engine import build_shap_payload, label_for, shap_for_player_row, summarize_shap

FEATURES = ["adj_velocity", "spin_rate", "tier_coefficient", "extension"]


def test_regression_payload_preserves_negative_contributions():
    """
    The old ``build_shap_payload`` took ``np.abs(vals).mean(axis=0)``, so every impact
    came back non-negative and the chart could not show direction. Negative SHAP values
    must survive as negative numbers.
    """
    values = np.array([[0.05, -0.03, 0.01, -0.08]])
    payload = build_shap_payload(values, FEATURES, "proj_wOBA")
    impacts = {c["feature"]: c["impact"] for c in payload["proj_wOBA"]}

    assert impacts["spin_rate"] < 0
    assert impacts["extension"] < 0
    assert impacts["adj_velocity"] > 0


def test_regression_tier_coefficient_is_not_sign_flipped():
    """
    The old code hard-coded ``-impact if fname == "tier_coefficient"``, manufacturing a
    negative bar for one feature regardless of its real contribution. A positive SHAP
    value for tier_coefficient must stay positive.
    """
    values = np.array([[0.0, 0.0, 0.042, 0.0]])
    payload = build_shap_payload(values, FEATURES, "proj_ERA")
    impacts = {c["feature"]: c["impact"] for c in payload["proj_ERA"]}

    assert impacts["tier_coefficient"] == 0.042


def test_payload_is_ranked_by_absolute_magnitude():
    values = np.array([[0.01, -0.09, 0.04, -0.02]])
    payload = build_shap_payload(values, FEATURES, "proj_wOBA")
    magnitudes = [abs(c["impact"]) for c in payload["proj_wOBA"]]

    assert magnitudes == sorted(magnitudes, reverse=True)
    assert payload["proj_wOBA"][0]["feature"] == "spin_rate"


def test_payload_respects_top_n():
    values = np.array([[0.05, -0.03, 0.01, -0.08]])
    payload = build_shap_payload(values, FEATURES, "proj_wOBA", top_n=2)
    assert len(payload["proj_wOBA"]) == 2


def test_empty_values_produce_empty_payload():
    payload = build_shap_payload(np.zeros((0, 4)), FEATURES, "proj_wOBA")
    assert payload["proj_wOBA"] == []


def test_summarize_shap_reports_globally_important_features():
    values = np.array([[0.01, -0.50, 0.02, 0.0], [0.02, 0.48, -0.01, 0.0]])
    assert summarize_shap(values, FEATURES, top_n=1) == ["spin_rate"]


def test_label_for_falls_back_to_prettified_name():
    assert label_for("adj_velocity") == "Adjusted Velocity"
    assert label_for("some_new_feature") == "Some New Feature"


def test_row_explanation_is_signed_for_a_real_model():
    """End-to-end: a fitted tree model must yield signed per-player contributions."""
    from sklearn.ensemble import GradientBoostingRegressor

    rng = np.random.default_rng(0)
    n = 80
    X = pd.DataFrame({f: rng.normal(0, 1, n) for f in FEATURES})
    y = 2.0 * X["adj_velocity"] - 1.5 * X["extension"] + rng.normal(0, 0.05, n)

    estimator = GradientBoostingRegressor(random_state=0).fit(X.values, y.values)

    # A player far above average on adj_velocity and far below on extension should show
    # a positive contribution for the first and a negative one for the second.
    row = pd.Series({f: 0.0 for f in FEATURES})
    row["adj_velocity"] = 2.0
    row["extension"] = -2.0

    payload = shap_for_player_row(estimator, row, FEATURES, "batter")
    impacts = {c["feature"]: c["impact"] for c in payload["proj_wOBA"]}

    assert impacts["adj_velocity"] > 0
    assert impacts["extension"] > 0  # negative feature value x negative coefficient
    assert any(v != 0 for v in impacts.values())
