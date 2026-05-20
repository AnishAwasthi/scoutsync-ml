#!/usr/bin/env python3
"""
ScoutSync ML — Interactive Streamlit dashboard.

Run locally (no FastAPI required):
  USE_SQLITE=true streamlit run dashboard.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("USE_SQLITE", "").lower() in ("1", "true", "yes"):
    os.environ["USE_SQLITE"] = "true"


def bootstrap_sqlite_database() -> None:
    """
    On first run with USE_SQLITE, create and seed the database if missing.

    Mirrors `python main.py init-db` and `python main.py seed` before any UI loads.
    """
    from src.db.session import get_db_session, init_db, sqlite_db_path, use_sqlite
    from src.ingestion.lite_seed import seed_cloud_lite
    from src.ingestion.pybaseball_loader import seed_mlb_statcast
    from src.ingestion.synthetic_seed import seed_synthetic_data
    from src.logging_config import setup_logging
    from src.runtime import is_streamlit_cloud

    if not use_sqlite():
        return

    db_path = sqlite_db_path()
    if db_path.exists():
        return

    setup_logging()
    init_db()

    session = get_db_session()
    try:
        if is_streamlit_cloud():
            seed_cloud_lite(session, num_amateur_players=8)
        else:
            from src.config import get_settings

            settings = get_settings()
            seed_mlb_statcast(session, validation_year=settings.validation_year)
            seed_synthetic_data(session, num_players=50)
    finally:
        session.close()


bootstrap_sqlite_database()

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
    resolve_shap_contributions,
)
from src.dashboard.visualizations import (
    projection_density_figure,
    shap_bar_figure,
    tracking_histogram_figure,
)
from src.db.session import get_db_session
from src.pipeline import get_player_breakdown

st.set_page_config(
    page_title="ScoutSync ML",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_model_bundle() -> LoadedModel:
    return load_translation_model()


@st.cache_data(ttl=60)
def cached_players() -> list[dict]:
    session = get_db_session()
    try:
        return [
            {
                "player_id": p.player_id,
                "full_name": p.full_name,
                "position": p.primary_position or "—",
            }
            for p in list_players(session)
        ]
    finally:
        session.close()


def render_metric_cards(projection: ProjectionView) -> None:
    if projection.mean is None:
        st.warning("Projection mean is unavailable for this player.")
        return

    c1, c2, c3 = st.columns(3)
    lower = projection.lower_90 if projection.lower_90 is not None else projection.mean
    upper = projection.upper_90 if projection.upper_90 is not None else projection.mean

    c1.metric(
        projection.metric_label,
        f"{projection.mean:.3f}" if projection.role == "batter" else f"{projection.mean:.2f}",
    )
    c2.metric(
        "Lower 90% CI",
        f"{lower:.3f}" if projection.role == "batter" else f"{lower:.2f}",
    )
    c3.metric(
        "Upper 90% CI",
        f"{upper:.3f}" if projection.role == "batter" else f"{upper:.2f}",
    )

    if projection.variance_delta is not None:
        st.caption(f"Estimated σ (variance delta): **{projection.variance_delta:.4f}**")


def render_player_profile(profile: PlayerProfile, role: str) -> None:
    st.subheader("Player Profile")
    col_a, col_b, col_c, col_d = st.columns(4)
    col_a.markdown(f"**Name**  \n{profile.full_name}")
    col_b.markdown(f"**Age**  \n{profile.age if profile.age is not None else '—'}")
    col_c.markdown(f"**Bats / Throws**  \n{profile.bats or '—'} / {profile.throws or '—'}")
    col_d.markdown(f"**Position**  \n{profile.primary_position or '—'}")
    st.caption(f"Viewing **{role.title()}** projections · Target season {get_settings().validation_year}")


def render_sidebar(players: list[dict]) -> tuple[int | None, str]:
    st.sidebar.title("ScoutSync ML")
    ok, backend = db_status()
    if ok:
        st.sidebar.success(f"Database connected ({backend})")
    else:
        st.sidebar.error("Database unavailable. Run `python main.py init-db`.")

    model_bundle = get_model_bundle()
    if model_bundle.available:
        st.sidebar.info("Translation models loaded")
    else:
        st.sidebar.warning(model_bundle.message)

    role = st.sidebar.radio(
        "Projection type",
        options=["Batter Projections", "Pitcher Projections"],
        horizontal=True,
    )
    role_key = "batter" if role.startswith("Batter") else "pitcher"

    if not players:
        st.sidebar.warning("No players in database. Run `python main.py seed`.")
        return None, role_key

    search = st.sidebar.text_input("Search players", placeholder="Type a name...")
    filtered = players
    if search.strip():
        q = search.strip().lower()
        filtered = [p for p in players if q in p["full_name"].lower()]

    if not filtered:
        st.sidebar.warning("No players match your search.")
        return None, role_key

    options = {p["player_id"]: f"{p['full_name']} ({p['position']}) — ID {p['player_id']}" for p in filtered}
    player_id = st.sidebar.selectbox(
        "Select player",
        options=list(options.keys()),
        format_func=lambda pid: options[pid],
    )
    return player_id, role_key


def main() -> None:
    st.title("ScoutSync ML — Cross-League Translation")
    st.markdown(
        "Translate amateur and international tracking data into **MLB baseline projections** "
        "with confidence intervals and SHAP explainability."
    )

    try:
        players = cached_players()
    except Exception as exc:
        st.error(f"Failed to load players: {exc}")
        st.info("Ensure the database is initialized (`python main.py init-db`) and seeded (`python main.py seed`).")
        return

    player_id, role = render_sidebar(players)
    if player_id is None:
        st.info("Add players via `python main.py seed` to explore projections.")
        return

    session = get_db_session()
    try:
        profile = get_player(session, player_id)
        if not profile:
            st.error(f"Player ID **{player_id}** was not found.")
            return

        projection = get_projection(session, player_id, role)
        model_bundle = get_model_bundle()
        breakdown = get_player_breakdown(session, player_id)

        render_player_profile(profile, role)

        if projection is None:
            st.warning(
                f"No **{role}** projection found for this player. "
                "Run `python main.py train` after seeding tracking data."
            )
            if breakdown.get("error"):
                st.caption("No tracking rows available for distribution charts.")
            return

        st.divider()
        st.subheader("MLB Projection Summary")
        render_metric_cards(projection)

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Outcome Distribution")
            fig_density = projection_density_figure(projection)
            if fig_density:
                st.plotly_chart(fig_density, use_container_width=True)
            else:
                st.info("Insufficient data to render the distribution curve.")

        with col_right:
            st.subheader("SHAP Explainability")
            contributions = resolve_shap_contributions(
                session, player_id, role, projection, model_bundle
            )
            shap_title = (
                f"Drivers of {projection.metric_label}"
                if contributions
                else "SHAP contributions unavailable"
            )
            fig_shap = shap_bar_figure(contributions, title=shap_title)
            if fig_shap:
                st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info(
                    "No SHAP data in the database and live recalculation failed. "
                    "Re-run `python main.py train` to refresh explainability payloads."
                )

        if not breakdown.get("error"):
            st.divider()
            st.subheader("Raw vs Adjusted Tracking")
            c1, c2 = st.columns(2)
            raw_fig = tracking_histogram_figure(breakdown, "Raw Metric Distribution")
            adj_fig = tracking_histogram_figure(
                {"adjusted_distribution": breakdown.get("adjusted_distribution")},
                "Environment-Adjusted Distribution",
            )
            with c1:
                if raw_fig:
                    st.plotly_chart(raw_fig, use_container_width=True)
                else:
                    st.caption("No raw distribution bins.")
            with c2:
                if adj_fig:
                    st.plotly_chart(adj_fig, use_container_width=True)
                else:
                    st.caption("No adjusted distribution bins.")

    except Exception as exc:
        st.error(f"An unexpected error occurred: {exc}")
        st.exception(exc)
    finally:
        session.close()


if __name__ == "__main__":
    main()
