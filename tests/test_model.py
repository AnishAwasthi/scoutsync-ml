"""
Model-layer tests, most of them regression guards.

Each test named ``test_regression_*`` pins a specific bug that shipped in an earlier
version. They are the reason the ERA side of the product works at all, so they should
fail loudly rather than be relaxed.
"""

import numpy as np
import pandas as pd
import pytest

from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES
from src.ml.model import (
    TARGET_COLUMN,
    ScoutSyncTranslationModel,
    prepare_targets,
    train_models_from_frames,
)


def _frame(features, n=12, seed=0, **extra):
    rng = np.random.default_rng(seed)
    data = {f: rng.normal(0, 1, n) for f in features}
    data["player_id"] = np.arange(1, n + 1)
    data.update(extra)
    return pd.DataFrame(data)


def test_prepare_targets_returns_one_dimensional_vector():
    frame = _frame(BATTER_FEATURES, n=6, true_wOBA=np.linspace(0.28, 0.36, 6))
    y = prepare_targets(frame, "batter")
    assert y.ndim == 1
    assert len(y) == 6


def test_regression_prepare_targets_refuses_missing_label_column():
    """
    The old implementation did ``df.get("true_wOBA", df.get(..., 0.31))`` and filled the
    whole column with a constant when the label was absent. That trained a model to
    predict one number for every player. Missing labels must raise instead.
    """
    frame = _frame(BATTER_FEATURES, n=6)  # no true_wOBA column
    with pytest.raises(KeyError, match="true_wOBA"):
        prepare_targets(frame, "batter")


def test_regression_prepare_targets_refuses_partial_labels():
    frame = _frame(BATTER_FEATURES, n=4, true_wOBA=[0.30, np.nan, 0.32, 0.33])
    with pytest.raises(ValueError, match="unlabeled"):
        prepare_targets(frame, "batter")


def test_regression_single_player_frame_does_not_raise_unbound_local(isolated_model_dir):
    """
    With one unique player the old split branch assigned ``train_idx``/``val_idx`` and
    then referenced ``X_train``, raising UnboundLocalError. It must train instead.
    """
    frame = _frame(PITCHER_FEATURES, n=8, true_ERA=np.linspace(3.0, 5.0, 8))
    frame["player_id"] = 1  # a single player across all rows

    model = train_models_from_frames(frame, pd.DataFrame())
    assert model.pitcher_model is not None


def test_regression_pitcher_predictions_are_not_constant(isolated_model_dir):
    """
    The pitcher model used to emit exactly 4.50 for every player because its target was
    a constant. Predictions must vary with the features.
    """
    n = 40
    rng = np.random.default_rng(3)
    talent = rng.normal(0, 1, n)
    frame = pd.DataFrame({f: rng.normal(0, 0.4, n) for f in PITCHER_FEATURES})
    frame["adj_velocity"] = 88 + 2.5 * talent
    frame["player_id"] = np.arange(1, n + 1)
    frame["true_ERA"] = 4.3 - 0.7 * talent + rng.normal(0, 0.1, n)

    model = train_models_from_frames(frame, pd.DataFrame())
    preds = model.predict(frame[PITCHER_FEATURES], "pitcher")

    assert len(np.unique(np.round(preds, 4))) > 1, "pitcher model collapsed to a constant"
    assert preds.std() > 0.05


def test_regression_residual_std_is_positive_so_intervals_have_width(isolated_model_dir):
    """A zero residual std produced a 90% CI of [x, x], which is not an interval."""
    n = 40
    rng = np.random.default_rng(5)
    frame = pd.DataFrame({f: rng.normal(0, 1, n) for f in PITCHER_FEATURES})
    frame["player_id"] = np.arange(1, n + 1)
    frame["true_ERA"] = 4.0 + rng.normal(0, 0.8, n)

    model = train_models_from_frames(frame, pd.DataFrame())
    assert model.residual_for("pitcher") > 0


def test_model_round_trips_through_disk(isolated_model_dir):
    n = 30
    rng = np.random.default_rng(11)
    frame = pd.DataFrame({f: rng.normal(0, 1, n) for f in BATTER_FEATURES})
    frame["player_id"] = np.arange(1, n + 1)
    frame["true_wOBA"] = 0.31 + rng.normal(0, 0.03, n)

    trained = train_models_from_frames(pd.DataFrame(), frame)
    trained.save(isolated_model_dir)

    reloaded = ScoutSyncTranslationModel()
    reloaded.load(isolated_model_dir)

    assert reloaded.batter_model is not None
    assert reloaded.residual_for("batter") == pytest.approx(trained.residual_for("batter"))
    assert reloaded.validation_players.get("batter") == trained.validation_players.get("batter")

    np.testing.assert_allclose(
        reloaded.predict(frame[BATTER_FEATURES], "batter"),
        trained.predict(frame[BATTER_FEATURES], "batter"),
    )


def test_held_out_players_are_excluded_from_training(isolated_model_dir):
    n = 40
    rng = np.random.default_rng(7)
    frame = pd.DataFrame({f: rng.normal(0, 1, n) for f in BATTER_FEATURES})
    frame["player_id"] = np.arange(1, n + 1)
    frame["true_wOBA"] = 0.31 + rng.normal(0, 0.03, n)

    model = train_models_from_frames(frame if False else pd.DataFrame(), frame)
    held_out = set(model.validation_players["batter"])

    assert held_out, "no players were held out, so the backtest would be in-sample"
    assert held_out.issubset(set(frame["player_id"]))


def test_target_columns_are_distinct_per_role():
    assert TARGET_COLUMN["batter"] != TARGET_COLUMN["pitcher"]
