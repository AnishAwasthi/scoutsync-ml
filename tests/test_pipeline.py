"""End-to-end pipeline tests against a throwaway SQLite database."""

import numpy as np
import pandas as pd

from db.models import MlbProjection, Player, PlayerGroundTruth
from src.features.builder import build_training_frames
from src.pipeline import (
    attach_training_labels,
    load_ground_truth,
    load_tracking_dataframe,
    normalize_tracking_df,
    run_train_pipeline,
)


def test_init_db_creates_the_full_schema(scratch_db):
    """
    ``init-db`` used to create only ``players`` and ``mlb_projections`` under SQLite, so
    the documented ``init-db && seed`` flow died on "no such table: leagues".
    """
    from sqlalchemy import inspect

    from src.db.session import get_engine

    tables = set(inspect(get_engine()).get_table_names())
    assert {
        "leagues",
        "players",
        "stadium_environments",
        "raw_tracking_data",
        "mlb_projections",
        "backtest_runs",
        "player_ground_truth",
    } <= tables


def test_init_db_seeds_reference_leagues(scratch_db):
    from db.models import League

    abbreviations = {abbr for (abbr,) in scratch_db.query(League.abbreviation).all()}
    assert {"MLB", "NPB", "KBO", "NCAA", "CCL"} == abbreviations


def test_seeding_produces_ground_truth_for_every_synthetic_player(seeded_db):
    players = seeded_db.query(Player).count()
    truths = seeded_db.query(PlayerGroundTruth).count()
    assert players == truths == 40


def test_ground_truth_labels_belong_to_their_own_player(seeded_db):
    """
    Labels used to be assigned round-robin (amateur *i* got MLB rookie *i mod N*), which
    is not a relationship at all. Each label must now key off the player's own id.
    """
    truth = load_ground_truth(seeded_db)
    assert len(truth) == 40
    assert truth["player_id"].is_unique

    batters = truth[truth["role"] == "batter"]
    pitchers = truth[truth["role"] == "pitcher"]
    assert batters["true_wOBA"].notna().all()
    assert batters["true_ERA"].isna().all()
    assert pitchers["true_ERA"].notna().all()
    assert pitchers["true_wOBA"].isna().all()


def test_labels_correlate_with_latent_talent(seeded_db):
    """The generator must actually encode a recoverable signal."""
    truth = load_ground_truth(seeded_db)
    batters = truth[truth["role"] == "batter"]
    pitchers = truth[truth["role"] == "pitcher"]

    woba_corr = np.corrcoef(batters["latent_talent"], batters["true_wOBA"])[0, 1]
    era_corr = np.corrcoef(pitchers["latent_talent"], pitchers["true_ERA"])[0, 1]

    assert woba_corr > 0.5, "higher talent should mean higher wOBA"
    assert era_corr < -0.5, "higher talent should mean lower ERA"


def test_normalization_recovers_the_latent_metric(seeded_db):
    """
    The seeder injects park bias as the inverse of the adjustment, so the adjusted
    series must track latent talent more tightly than the raw one does. If this fails
    the environmental layer is decorative.
    """
    raw = load_tracking_dataframe(seeded_db)
    normalized = normalize_tracking_df(raw)
    truth = load_ground_truth(seeded_db).set_index("player_id")["latent_talent"]

    pitches = normalized[normalized["release_speed"].notna()]
    grouped = pitches.groupby("player_id").agg(
        raw_mean=("release_speed", "mean"), adj_mean=("adj_velocity", "mean")
    )
    grouped["talent"] = truth.reindex(grouped.index)
    grouped = grouped.dropna()

    raw_corr = abs(np.corrcoef(grouped["raw_mean"], grouped["talent"])[0, 1])
    adj_corr = abs(np.corrcoef(grouped["adj_mean"], grouped["talent"])[0, 1])

    assert adj_corr >= raw_corr, f"adjustment lost signal: raw={raw_corr:.4f} adj={adj_corr:.4f}"
    assert adj_corr > 0.8


def test_exit_velocity_gets_a_park_adjustment(seeded_db):
    raw = load_tracking_dataframe(seeded_db)
    normalized = normalize_tracking_df(raw)
    assert "adj_exit_velocity" in normalized.columns

    hits = normalized[normalized["exit_velocity"].notna()]
    differing = (hits["adj_exit_velocity"] - hits["exit_velocity"]).abs() > 1e-9
    assert differing.any(), "no batted ball was park-adjusted"


def test_attach_training_labels_matches_on_player_id(seeded_db):
    raw = load_tracking_dataframe(seeded_db)
    normalized = normalize_tracking_df(raw)
    _, pitchers, batters, _, _ = build_training_frames(normalized, pd.DataFrame())

    pitchers, batters = attach_training_labels(pitchers, batters, load_ground_truth(seeded_db))

    assert pitchers["true_ERA"].notna().any()
    assert batters["true_wOBA"].notna().any()


def test_full_train_pipeline_produces_varied_projections(seeded_db, isolated_model_dir):
    """
    The headline regression: every pitcher projection used to be exactly 4.50 with a
    zero-width interval. Both roles must now produce a spread of values.
    """
    model = run_train_pipeline(seeded_db)
    assert model.pitcher_model is not None
    assert model.batter_model is not None

    rows = seeded_db.query(MlbProjection).all()
    assert rows, "no projections were written"

    eras = [float(r.proj_ERA) for r in rows if r.proj_ERA is not None]
    wobas = [float(r.proj_wOBA) for r in rows if r.proj_wOBA is not None]

    assert len(set(eras)) > 1, "pitcher projections collapsed to a single value"
    assert len(set(wobas)) > 1, "batter projections collapsed to a single value"


def test_confidence_intervals_have_nonzero_width(seeded_db, isolated_model_dir):
    run_train_pipeline(seeded_db)
    for row in seeded_db.query(MlbProjection).all():
        if row.proj_ERA is not None:
            assert float(row.proj_ERA_upper_90) > float(row.proj_ERA_lower_90)
        if row.proj_wOBA is not None:
            assert float(row.proj_wOBA_upper_90) > float(row.proj_wOBA_lower_90)


def test_stored_shap_payloads_contain_signed_values(seeded_db, isolated_model_dir):
    run_train_pipeline(seeded_db)

    saw_negative = False
    for row in seeded_db.query(MlbProjection).all():
        payload = row.shap_explainability_json or {}
        for key in ("proj_wOBA", "proj_ERA"):
            for contribution in payload.get(key, []):
                if contribution["impact"] < 0:
                    saw_negative = True
    assert saw_negative, "no negative SHAP contribution was ever stored"


def test_retraining_replaces_projections_instead_of_appending(seeded_db, isolated_model_dir):
    """Re-running training used to append a whole new set of rows every time."""
    run_train_pipeline(seeded_db)
    first = seeded_db.query(MlbProjection).count()

    run_train_pipeline(seeded_db)
    second = seeded_db.query(MlbProjection).count()

    assert first == second, f"projections accumulated: {first} -> {second}"


def test_projections_cover_only_the_amateur_cohort(seeded_db, isolated_model_dir):
    run_train_pipeline(seeded_db)
    projected = {r.player_id for r in seeded_db.query(MlbProjection).all()}
    amateurs = {r.player_id for r in seeded_db.query(PlayerGroundTruth).all()}
    assert projected <= amateurs


def test_synthetic_seed_is_deterministic(fresh_database_factory):
    """Reproducible numbers matter: the README quotes backtest output."""
    from src.ingestion.synthetic_seed import seed_synthetic_data

    def cohort(name):
        db = fresh_database_factory(name)
        seed_synthetic_data(db, num_players=10)
        return [(float(r.latent_talent), r.role) for r in db.query(PlayerGroundTruth).all()]

    assert cohort("first") == cohort("second")
