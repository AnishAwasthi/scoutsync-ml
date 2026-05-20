"""SHAP explainability for ScoutSync models."""

import numpy as np
import pandas as pd
import shap
from sklearn.multioutput import MultiOutputRegressor

from src.logging_config import get_logger

def _fallback_shap(estimator, sample: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """Feature-importance weighted pseudo-SHAP when TreeExplainer unavailable."""
    if hasattr(estimator, "feature_importances_"):
        imp = np.array(estimator.feature_importances_)
    else:
        imp = np.ones(len(feature_names)) / len(feature_names)
    row = sample.iloc[0].values if len(sample) else np.zeros(len(feature_names))
    centered = row - row.mean()
    vals = centered * imp
    return np.tile(vals, (len(sample), 1))


FEATURE_LABELS = {
    "adj_velocity": "Adjusted Velocity",
    "spin_rate": "Spin Rate",
    "vertical_break": "Induced Vertical Break",
    "horizontal_break": "Horizontal Break",
    "vaa": "Vertical Approach Angle",
    "extension": "Extension",
    "strike_zone_command_rate": "Strike Zone Command",
    "age_relative_to_league": "Age vs League Median",
    "tier_coefficient": "Competition Tier",
    "max_exit_velocity": "Max Exit Velocity",
    "pct_90th_exit_velocity": "90th Percentile Exit Velo",
    "launch_angle_sweetspot_rate": "Launch Angle Sweet Spot",
    "zone_contact_rate": "Zone Contact Rate",
    "out_of_zone_chase_rate": "Chase Rate (O-Zone)",
    "conference_strength_factor": "Conference Strength",
}


def explain_model(
    model: MultiOutputRegressor,
    X_val: pd.DataFrame,
    role: str,
) -> np.ndarray:
    """Compute SHAP values using TreeExplainer on first estimator."""
    logger = get_logger()
    base_estimator = model.estimators_[0]
    sample = X_val.head(min(200, len(X_val)))
    try:
        explainer = shap.TreeExplainer(base_estimator)
        shap_values = explainer(sample.values)
        logger.info(f"shap_computed role={role} samples={len(sample)}")
        return shap_values
    except Exception as exc:
        logger.warning(f"shap_fallback role={role} reason={exc}")
        return _fallback_shap(base_estimator, sample, list(X_val.columns))


def build_shap_payload(
    shap_values: np.ndarray,
    feature_names: list[str],
    target_key: str = "proj_wOBA",
    top_n: int = 8,
) -> dict:
    """Aggregate mean |SHAP| into ranked contribution list."""
    if hasattr(shap_values, "values"):
        vals = shap_values.values
    else:
        vals = shap_values
    if vals.ndim == 3:
        vals = vals[:, :, 0]
    mean_abs = np.abs(vals).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:top_n]
    contributions = []
    for idx in order:
        fname = feature_names[idx]
        impact = float(mean_abs[idx])
        contributions.append(
            {
                "feature": fname,
                "impact": round(-impact if fname == "tier_coefficient" else impact, 4),
                "label": FEATURE_LABELS.get(fname, fname.replace("_", " ").title()),
            }
        )
    return {target_key: contributions}


def shap_for_player_row(
    model: MultiOutputRegressor,
    row: pd.Series,
    feature_names: list[str],
    role: str,
) -> dict:
    """Single-row SHAP explanation."""
    X = pd.DataFrame([row[feature_names].values], columns=feature_names)
    base_estimator = model.estimators_[0]
    try:
        explainer = shap.TreeExplainer(base_estimator)
        sv = explainer(X.values)
    except Exception:
        sv = _fallback_shap(base_estimator, X, feature_names)
    target = "proj_wOBA" if role == "batter" else "proj_ERA"
    return build_shap_payload(sv, feature_names, target_key=target)
