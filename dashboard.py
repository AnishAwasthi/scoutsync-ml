#!/usr/bin/env python3
"""
ScoutSync ML — interactive Streamlit dashboard.

Run locally:
    USE_SQLITE=true streamlit run dashboard.py

Everything shown here is computed. On a cold start with an empty database the app seeds
a small synthetic cohort and trains the models before rendering, so every projection,
confidence interval, and SHAP bar on the page comes from a model fit at runtime.

An earlier version of this file inserted hard-coded projections for three real major
league players and rendered them as if they were model output. They were not: no model
ran on the hosted demo at all. Nothing on this page is hard-coded now, and the synthetic
nature of the underlying data is stated in the UI rather than left for the reader to
discover in the source.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.config import get_settings
from src.dashboard.data_access import (
    LoadedModel,
    PlayerProfile,
    ProjectionView,
    db_status,
    get_player,
    get_projection,
    list_players,
    load_translation_model,
    projected_player_ids,
    resolve_shap_contributions,
    shap_is_approximate,
)
from src.dashboard.visualizations import (
    projection_density_figure,
    shap_bar_figure,
    tracking_histogram_figure,
)
from src.db.session import Player, get_db_session, init_db, use_sqlite
from src.ml.model import backend_name
from src.pipeline import get_player_breakdown, run_train_pipeline

DEMO_COHORT_SIZE = 60

st.set_page_config(
    page_title="ScoutSync ML",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Preparing demo database (seeding and training)...")
def bootstrap_demo_database() -> dict:
    """
    Make sure the app has a populated database and trained models.

    Cached as a resource so it runs once per process, not on every rerun. Returns a
    status dict the sidebar renders, including any error, so a failure surfaces in the
    UI instead of leaving the page mysteriously empty.
    """
    status: dict = {"seeded": False, "trained": False, "error": None}

    try:
        init_db()
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        status["error"] = f"Database init failed: {exc}"
        return status

    session = get_db_session()
    try:
        if session.query(Player).count() == 0:
            from src.ingestion.lite_seed import seed_cloud_lite

            seed_cloud_lite(session, num_amateur_players=DEMO_COHORT_SIZE)
            status["seeded"] = True

        model = load_translation_model()
        if not model.available:
            run_train_pipeline(session)
            status["trained"] = True
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        status["error"] = f"Demo bootstrap failed: {exc}"
        session.rollback()
    finally:
        session.close()

    return status


@st.cache_resource
def get_model_bundle() -> LoadedModel:
    return load_translation_model()


@st.cache_data(ttl=60)
def cached_players(role: str) -> list[dict]:
    """Players with a projection for ``role``, ready for the sidebar picker."""
    session = get_db_session()
    try:
        projectable = projected_player_ids(session, role)
        return [
            {
                "player_id": p.player_id,
                "full_name": p.full_name,
                "position": p.primary_position or "—",
            }
            for p in list_players(session)
            if p.player_id in projectable
        ]
    finally:
        session.close()


def render_data_provenance() -> None:
    """State plainly what the numbers on this page are, before showing any of them."""
    st.info(
        "**Demo data is synthetic.** The amateur cohort is generated from a latent "
        "talent parameter that drives both each player's tracking metrics and their "
        "stored ground-truth outcome, so the backtest measures whether the pipeline "
        "recovers a known signal. It is **not** evidence of real predictive accuracy — "
        "no public dataset links amateur tracking data to major-league outcomes at the "
        "player level. Statcast rows are real, and are used only to set league baselines.",
        icon="🧪",
    )


def render_metric_cards(projection: ProjectionView) -> None:
    if projection.mean is None:
        st.warning("Projection mean is unavailable for this player.")
        return

    fmt = "{:.3f}" if projection.role == "batter" else "{:.2f}"
    lower = projection.lower_90 if projection.lower_90 is not None else projection.mean
    upper = projection.upper_90 if projection.upper_90 is not None else projection.mean

    c1, c2, c3 = st.columns(3)
    c1.metric(projection.metric_label, fmt.format(projection.mean))
    c2.metric("Lower 90% CI", fmt.format(lower))
    c3.metric("Upper 90% CI", fmt.format(upper))

    if projection.variance_delta is not None:
        st.caption(
            f"Interval half-width is 1.645σ, with σ = **{projection.variance_delta:.4f}** "
            "estimated from held-out residuals at training time."
        )


def render_player_profile(profile: PlayerProfile, role: str) -> None:
    st.subheader("Player Profile")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(f"**Name**  \n{profile.full_name}")
    col_b.markdown(f"**Age**  \n{profile.age if profile.age is not None else '—'}")
    col_c.markdown(f"**Bats / Throws**  \n{profile.bats or '—'} / {profile.throws or '—'}")
    col_d.markdown(f"**Position**  \n{profile.primary_position or '—'}")
    st.caption(
        f"Viewing **{role}** projections · target season {get_settings().validation_year}"
    )


def render_sidebar(bootstrap: dict) -> tuple[int | None, str]:
    st.sidebar.title("ScoutSync ML")

    if bootstrap.get("error"):
        st.sidebar.error(bootstrap["error"])

    ok, backend = db_status()
    if ok:
        st.sidebar.success(f"Database connected ({backend})")
    else:
        st.sidebar.error("Database unavailable. Run `python main.py init-db`.")

    model_bundle = get_model_bundle()
    if model_bundle.available:
        st.sidebar.info(f"Models loaded · gradient boosting via `{backend_name()}`")
    else:
        st.sidebar.warning(model_bundle.message)

    role_choice = st.sidebar.radio(
        "Projection type",
        options=["Batter Projections", "Pitcher Projections"],
        horizontal=True,
    )
    role = "batter" if role_choice.startswith("Batter") else "pitcher"

    try:
        players = cached_players(role)
    except Exception as exc:  # noqa: BLE001 - surfaced in the UI
        st.sidebar.error(f"Failed to load players: {exc}")
        return None, role

    if not players:
        st.sidebar.warning(
            f"No {role} projections stored. Run `python main.py seed` then "
            "`python main.py train`."
        )
        return None, role

    search = st.sidebar.text_input("Search players", placeholder="Type a name...")
    filtered = players
    if search.strip():
        query = search.strip().lower()
        filtered = [p for p in players if query in p["full_name"].lower()]

    if not filtered:
        st.sidebar.warning("No players match your search.")
        return None, role

    options = {
        p["player_id"]: f"{p['full_name']} ({p['position']}) — ID {p['player_id']}"
        for p in filtered
    }
    player_id = st.sidebar.selectbox(
        "Select player",
        options=list(options.keys()),
        format_func=lambda pid: options[pid],
    )
    st.sidebar.caption(
        f"{len(players)} projectable {role}s. MLB reference players are excluded — they "
        "set league baselines rather than being translated."
    )
    return player_id, role


def render_explainability(session, player_id: int, role: str, projection: ProjectionView) -> None:
    st.subheader("SHAP Explainability")
    contributions = resolve_shap_contributions(
        session, player_id, role, projection, get_model_bundle()
    )
    if not contributions:
        st.info(
            "No SHAP payload stored for this player and live recalculation was not "
            "possible. Re-run `python main.py train` to refresh explainability."
        )
        return

    lower_is_better = role == "pitcher"
    fig = shap_bar_figure(
        contributions,
        title=f"Drivers of {projection.metric_label}",
        lower_is_better=lower_is_better,
        metric_unit=projection.metric_unit,
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Green improves the projection, red worsens it. Values are signed SHAP "
        "contributions in units of "
        f"{projection.metric_unit}"
        + (" (negative lowers ERA, which is better)." if lower_is_better else ".")
    )
    if shap_is_approximate(projection.shap_json):
        st.warning(
            "These are importance-weighted approximations, not exact Shapley values — "
            "the tree explainer was unavailable for this model.",
            icon="⚠️",
        )


def main() -> None:
    bootstrap = bootstrap_demo_database() if use_sqlite() else {}

    st.title("ScoutSync ML — Cross-League Translation")
    st.markdown(
        "Translate amateur tracking data into **MLB baseline projections** with "
        "confidence intervals and SHAP explainability."
    )
    render_data_provenance()

    player_id, role = render_sidebar(bootstrap)
    if player_id is None:
        st.info(
            "No projections available yet. Initialize, seed, and train first:\n\n"
            "```bash\nUSE_SQLITE=true python main.py init-db\n"
            "USE_SQLITE=true python main.py seed\n"
            "USE_SQLITE=true python main.py train\n```"
        )
        return

    session = get_db_session()
    try:
        profile = get_player(session, player_id)
        if not profile:
            st.error(f"Player ID **{player_id}** was not found.")
            return

        render_player_profile(profile, role)
        projection = get_projection(session, player_id, role)

        if projection is None:
            st.warning(f"No **{role}** projection stored for this player.")
            return

        st.divider()
        st.subheader("MLB Projection Summary")
        render_metric_cards(projection)

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Outcome Distribution")
            fig_density = projection_density_figure(projection)
            if fig_density:
                st.plotly_chart(fig_density, width="stretch")
            else:
                st.info("Insufficient data to render the distribution curve.")

        with col_right:
            render_explainability(session, player_id, role, projection)

        st.divider()
        st.subheader("Environmental Normalization")
        breakdown = get_player_breakdown(session, player_id, role=role)
        fig_tracking = tracking_histogram_figure(
            breakdown, "Raw vs park-adjusted tracking distribution"
        )
        if fig_tracking:
            st.plotly_chart(fig_tracking, width="stretch")
            st.caption(
                "The shift between the two series is the altitude and air-density "
                "correction applied before the metric reaches the model."
            )
        else:
            st.caption("No pitch-level tracking rows available for this player.")
    finally:
        session.close()


main()
