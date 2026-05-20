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


def shap_bar_figure(contributions: list[dict], title: str = "Feature Contributions") -> go.Figure | None:
    if not contributions:
        return None

    labels = [c.get("label", c.get("feature", "")) for c in contributions]
    impacts = [float(c.get("impact", 0)) for c in contributions]
    colors = ["#2ca02c" if v >= 0 else "#d62728" for v in impacts]

    sorted_pairs = sorted(zip(labels, impacts, colors), key=lambda t: abs(t[1]))
    labels, impacts, colors = zip(*sorted_pairs)

    fig = go.Figure(
        go.Bar(
            x=impacts,
            y=list(labels),
            orientation="h",
            marker_color=list(colors),
            text=[f"{v:+.4f}" for v in impacts],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Impact on Projection",
        yaxis_title="",
        template="plotly_white",
        height=max(320, 40 * len(labels)),
        margin=dict(l=180),
    )
    return fig


def tracking_histogram_figure(breakdown: dict, title: str) -> go.Figure | None:
    """Optional raw vs adjusted histogram from pipeline breakdown."""
    dist = breakdown.get("raw_distribution") or breakdown.get("adjusted_distribution")
    if not dist or not dist.get("bins"):
        return None
    fig = go.Figure(
        go.Bar(
            x=dist["bins"],
            y=dist["counts"],
            marker_color="#636efa",
        )
    )
    fig.update_layout(title=title, xaxis_title=dist.get("name", "value"), yaxis_title="Count", template="plotly_white")
    return fig
