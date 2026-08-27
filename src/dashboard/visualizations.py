"""Plotly chart builders for ScoutSync dashboard."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from src.dashboard.data_access import ProjectionView


def projection_density_figure(
    projection: ProjectionView,
    n_points: int = 200,
) -> go.Figure | None:
    """Bell curve (normal PDF) for projected outcome distribution."""
    if projection.mean is None:
        return None

    if projection.variance_delta and projection.variance_delta > 0:
        sigma = projection.variance_delta
    elif projection.lower_90 is not None and projection.upper_90 is not None:
        sigma = (projection.upper_90 - projection.lower_90) / (2 * 1.645)
    else:
        sigma = 0.03 if projection.role == "batter" else 0.5

    sigma = max(sigma, 1e-6)
    x_min = projection.mean - 4 * sigma
    x_max = projection.mean + 4 * sigma
    x = np.linspace(x_min, x_max, n_points)
    pdf = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - projection.mean) / sigma) ** 2)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=pdf,
            mode="lines",
            name="Density",
            fill="tozeroy",
            line=dict(color="#1f77b4", width=2),
        )
    )
    fig.add_vline(x=projection.mean, line_dash="solid", line_color="#d62728", annotation_text="Mean")
    if projection.lower_90 is not None:
        fig.add_vline(
            x=projection.lower_90,
            line_dash="dash",
            line_color="#2ca02c",
            annotation_text="Lower 90%",
        )
    if projection.upper_90 is not None:
        fig.add_vline(
            x=projection.upper_90,
            line_dash="dash",
            line_color="#ff7f0e",
            annotation_text="Upper 90%",
        )

    fig.update_layout(
        title=f"Projected {projection.metric_unit} Distribution (90% CI)",
        xaxis_title=projection.metric_unit,
        yaxis_title="Probability Density",
        template="plotly_white",
        height=400,
        showlegend=False,
    )
    return fig


def shap_bar_figure(
    contributions: list[dict],
    title: str = "Feature Contributions",
    lower_is_better: bool = False,
    metric_unit: str = "",
) -> go.Figure | None:
    """
    Signed SHAP contributions, coloured by whether they helped the player.

    For ERA a negative contribution is *good* (it lowers the projected ERA), so colour
    keys off ``lower_is_better`` rather than the raw sign -- green always means "this
    factor improved the projection".
    """
    if not contributions:
        return None

    labels = [c.get("label", c.get("feature", "")) for c in contributions]
    impacts = [float(c.get("impact", 0)) for c in contributions]
    helped = [(v < 0) if lower_is_better else (v >= 0) for v in impacts]
    colors = ["#2ca02c" if good else "#d62728" for good in helped]

    labels, impacts, colors = zip(
        *sorted(zip(labels, impacts, colors, strict=True), key=lambda t: abs(t[1])),
        strict=True,
    )

    decimals = 2 if lower_is_better else 4
    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=list(labels),
            orientation="h",
            marker_color=list(colors),
            text=[f"{v:+.{decimals}f}" for v in impacts],
            textposition="outside",
            hovertemplate="%{y}: %{x:+." + str(decimals) + "f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#888")
    fig.update_layout(
        title=title,
        xaxis_title=f"Impact on projected {metric_unit}" if metric_unit else "Impact on projection",
        yaxis_title="",
        template="plotly_white",
        height=max(320, 42 * len(labels)),
        margin=dict(l=190, r=70),
    )
    return fig


def tracking_histogram_figure(breakdown: dict, title: str) -> go.Figure | None:
    """
    Raw vs park-adjusted distribution, overlaid.

    The gap between the two series is the environmental correction the normalization
    layer applied, so showing them together is the point -- one series alone says
    nothing about what the adjustment did.
    """
    raw = breakdown.get("raw_distribution") or {}
    adjusted = breakdown.get("adjusted_distribution") or {}
    if not raw.get("bins") and not adjusted.get("bins"):
        return None

    fig = go.Figure()
    if raw.get("bins"):
        fig.add_trace(
            go.Bar(x=raw["bins"], y=raw["counts"], name="Raw", marker_color="#9aa7ff", opacity=0.75)
        )
    if adjusted.get("bins"):
        fig.add_trace(
            go.Bar(
                x=adjusted["bins"],
                y=adjusted["counts"],
                name="Park-adjusted",
                marker_color="#1f3fd6",
                opacity=0.75,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=breakdown.get("metric", "Value"),
        yaxis_title="Count",
        barmode="overlay",
        template="plotly_white",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
