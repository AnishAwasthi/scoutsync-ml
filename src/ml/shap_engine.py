"""SHAP explainability for ScoutSync models.

Per-player explanations keep the **sign** of each SHAP value, because the sign is the
whole point: it says whether a factor pushed this player's projection up or down. An
earlier version collapsed everything to ``mean(|shap|)`` -- a global importance measure
that cannot express direction -- and then re-introduced a fake minus sign for one
hard-coded feature. Both are gone.

``summarize_shap`` still aggregates magnitude, but only for logging global importance
during training, never for the per-player chart.
"""

import numpy as np
import pandas as pd
import shap

from src.logging_config import get_logger

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


def label_for(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").title())


def _as_array(shap_values) -> np.ndarray:
    """Unwrap a shap Explanation into a plain 2-D (rows, features) array."""
    values = shap_values.values if hasattr(shap_values, "values") else shap_values
    values = np.asarray(values, dtype=float)
    if values.ndim == 3:  # (rows, features, outputs) for multi-output explainers
        values = values[:, :, 0]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def _fallback_shap(estimator, sample: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    """
    Importance-weighted approximation used only when TreeExplainer fails.

    This is NOT a Shapley decomposition. It signs each feature's importance by how far
    that row sits from the sample mean, which preserves direction well enough to be
    readable but does not satisfy the additivity guarantee. Callers surface it as
    approximate so it is never mistaken for exact SHAP.
    """
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype=float)
    else:
        importance = np.ones(len(feature_names), dtype=float) / len(feature_names)

    values = sample.to_numpy(dtype=float)
    centered = values - values.mean(axis=0, keepdims=True)
    return centered * importance


def explain_model(estimator, X_val: pd.DataFrame, max_rows: int = 200):
    """SHAP values for a validation sample; used for global importance logging."""
    sample = X_val.head(min(max_rows, len(X_val)))
    if sample.empty:
        return np.zeros((0, X_val.shape[1]))
    try:
        return shap.TreeExplainer(estimator)(sample.values)
    except Exception as exc:
        get_logger().warning(f"shap_fallback scope=model reason={exc}")
        return _fallback_shap(estimator, sample, list(X_val.columns))


def summarize_shap(shap_values, feature_names: list[str], top_n: int = 5) -> list[str]:
    """Most globally important features by mean |SHAP|, for training logs only."""
    values = _as_array(shap_values)
    if values.size == 0:
        return []
    mean_abs = np.abs(values).mean(axis=0)
    return [feature_names[i] for i in np.argsort(mean_abs)[::-1][:top_n]]


def build_shap_payload(
    shap_values,
    feature_names: list[str],
    target_key: str,
    top_n: int = 8,
    approximate: bool = False,
) -> dict:
    """
    Build the signed per-player contribution list stored on a projection.

    Ranked by absolute magnitude so the biggest movers come first, but ``impact`` keeps
    its sign: positive pushed the projection up, negative pushed it down.
    """
    values = _as_array(shap_values)
    if values.size == 0:
        return {target_key: []}

    row = values[0] if values.shape[0] == 1 else values.mean(axis=0)
    order = np.argsort(np.abs(row))[::-1][:top_n]

    contributions = [
        {
            "feature": feature_names[i],
            "label": label_for(feature_names[i]),
            "impact": round(float(row[i]), 5),
        }
        for i in order
    ]
    return {target_key: contributions, "approximate": approximate}


def shap_for_player_row(
    estimator,
    row: pd.Series,
    feature_names: list[str],
    role: str,
    background: pd.DataFrame | None = None,
) -> dict:
    """Signed SHAP explanation for one player-season."""
    X = pd.DataFrame([row[feature_names].to_numpy()], columns=feature_names)
    target_key = "proj_wOBA" if role == "batter" else "proj_ERA"

    try:
        values = shap.TreeExplainer(estimator)(X.values)
        return build_shap_payload(values, feature_names, target_key)
    except Exception as exc:
        get_logger().warning(f"shap_fallback scope=row role={role} reason={exc}")

    # The approximation needs a population to centre against. With only the one row
    # there is no reference, and centring it against itself would emit a chart of
    # zeros -- report nothing rather than something meaningless.
    if background is None or len(background) < 2:
        return {target_key: [], "approximate": True}

    reference = pd.concat([X, background[feature_names]], ignore_index=True)
    values = _fallback_shap(estimator, reference, feature_names)[:1]
    return build_shap_payload(values, feature_names, target_key, approximate=True)
