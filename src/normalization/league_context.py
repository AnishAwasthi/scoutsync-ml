"""Quality-of-competition league normalization."""

import numpy as np
import pandas as pd

from src.config import get_settings
from src.logging_config import get_logger


def gamma_for_tier(tier: int) -> float:
    return get_settings().gamma_tier_map.get(tier, 0.5)


def compute_league_baselines(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Rolling positional baseline mean/std per league container."""
    group_cols = group_cols or ["league_id"]
    records = []
    for keys, grp in df.groupby(group_cols):
        key_dict = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,)))
        for col in metric_cols:
            if col not in grp.columns:
                continue
            series = grp[col].dropna()
            if series.empty:
                continue
            records.append(
                {
                    **key_dict,
                    "metric": col,
                    "mu_league": float(series.mean()),
                    "sigma_league": float(series.std(ddof=0)) or 1.0,
                }
            )
    return pd.DataFrame(records)


def normalize_metric(
    values: pd.Series,
    mu: float,
    sigma: float,
    tier: int,
) -> pd.Series:
    gamma = gamma_for_tier(tier)
    sigma = sigma if sigma > 1e-6 else 1.0
    return ((values - mu) / sigma) * gamma


def apply_league_normalization(
    df: pd.DataFrame,
    baselines: pd.DataFrame,
    metric_col: str,
    output_col: str | None = None,
) -> pd.DataFrame:
    """Apply M_norm = ((M_adj - mu_league) / sigma_league) * gamma_tier."""
    out = df.copy()
    output_col = output_col or f"{metric_col}_norm"
    out[output_col] = np.nan

    if baselines.empty or metric_col not in out.columns:
        get_logger().warning(f"league_normalization skipped metric={metric_col}")
        return out

    for _, row in baselines[baselines["metric"] == metric_col].iterrows():
        mask = out["league_id"] == row["league_id"]
        if "pitch_type" in baselines.columns and "pitch_type" in row.index:
            mask &= out.get("pitch_type", pd.Series(dtype=object)) == row.get("pitch_type")
        tier = int(out.loc[mask, "competition_tier"].iloc[0]) if mask.any() else 4
        out.loc[mask, output_col] = normalize_metric(
            out.loc[mask, metric_col],
            float(row["mu_league"]),
            float(row["sigma_league"]),
            tier,
        )

    # Fallback: league-only baseline without pitch_type split
    if out[output_col].isna().any():
        league_only = baselines[baselines["metric"] == metric_col].drop_duplicates(
            subset=["league_id"], keep="first"
        )
        for _, row in league_only.iterrows():
            mask = (out["league_id"] == row["league_id"]) & out[output_col].isna()
            if not mask.any():
                continue
            tier = int(out.loc[mask, "competition_tier"].iloc[0])
            out.loc[mask, output_col] = normalize_metric(
                out.loc[mask, metric_col],
                float(row["mu_league"]),
                float(row["sigma_league"]),
                tier,
            )

    get_logger().info(
        f"league_normalization metric={metric_col} normalized_rows={out[output_col].notna().sum()}"
    )
    return out
