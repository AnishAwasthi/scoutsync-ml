"""
Held-out validation for the translation models.

What this measures: the synthetic cohort is generated from a latent talent parameter
that drives both the tracking metrics and the stored ground-truth outcome (see
``src.ingestion.synthetic_seed``). The backtest asks whether the pipeline recovers that
signal on players it did not train on.

What it does NOT measure: real predictive accuracy against major-league outcomes. That
would need an amateur-to-MLB player linkage that is not publicly available. Every number
this prints is a simulation-recovery number and is labelled as such.

The baseline comparison matters more than the raw error. A model that always predicts
the cohort mean gets a respectable-looking RMSE on its own; ``skill_vs_mean`` reports
how much better than that the model actually does, and it is the number to quote.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy.orm import Session

from db.models import BacktestRun
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES, build_training_frames
from src.logging_config import get_logger
from src.ml.model import TARGET_COLUMN, ScoutSyncTranslationModel
from src.pipeline import (
    attach_training_labels,
    load_ground_truth,
    load_tracking_dataframe,
    normalize_tracking_df,
)

METRIC_NAME = {"batter": "wOBA", "pitcher": "ERA"}


def _role_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    """Error plus skill against a predict-the-mean baseline."""
    rmse = float(np.sqrt(mean_squared_error(true, pred)))
    mae = float(mean_absolute_error(true, pred))
    baseline = np.full_like(true, true.mean())
    baseline_rmse = float(np.sqrt(mean_squared_error(true, baseline)))
    skill = float(1.0 - (rmse / baseline_rmse)) if baseline_rmse > 0 else 0.0
    return {
        "n": int(len(true)),
        "rmse": rmse,
        "mae": mae,
        "baseline_rmse": baseline_rmse,
        "skill_vs_mean": skill,
    }


def run_historical_validation(
    model: ScoutSyncTranslationModel,
    session: Session,
    validation_year: int = 2024,
) -> dict:
    """Score both roles against stored ground truth and record the run."""
    logger = get_logger()

    raw = load_tracking_dataframe(session)
    if raw.empty:
        raise RuntimeError("No tracking data. Run: python main.py seed")

    normalized = normalize_tracking_df(raw)
    _, pitchers, batters, _, _ = build_training_frames(normalized, pd.DataFrame())
    pitchers, batters = attach_training_labels(pitchers, batters, load_ground_truth(session))

    results: dict = {}
    per_role: dict = {}

    for role, frame, features in [
        ("batter", batters, BATTER_FEATURES),
        ("pitcher", pitchers, PITCHER_FEATURES),
    ]:
        estimator = model.get_estimator(role)
        target = TARGET_COLUMN[role]

        if estimator is None:
            logger.warning(f"backtest_skipped role={role} reason=model_not_trained")
            per_role[role] = {"skipped": "model not trained"}
            continue
        if frame.empty or target not in frame.columns:
            logger.warning(f"backtest_skipped role={role} reason=no_labeled_frame")
            per_role[role] = {"skipped": "no labeled rows"}
            continue

        labeled = frame.dropna(subset=list(features) + [target], how="any")
        if labeled.empty:
            logger.warning(f"backtest_skipped role={role} reason=no_rows_after_dropna")
            per_role[role] = {"skipped": "no labeled rows"}
            continue

        # Score only players the model never saw. Without this the "validation" number
        # is mostly in-sample fit and overstates skill.
        held_out = model.validation_players.get(role) or []
        if held_out:
            evaluated = labeled[labeled["player_id"].isin(held_out)]
            in_sample = False
        else:
            evaluated = labeled
            in_sample = True
            logger.warning(
                f"backtest_in_sample role={role}; no held-out player list on the model "
                "artifact, so this score includes training players"
            )

        if evaluated.empty:
            logger.warning(f"backtest_skipped role={role} reason=no_held_out_rows")
            per_role[role] = {"skipped": "no held-out rows"}
            continue

        true = evaluated[target].astype(float).to_numpy()
        pred = model.predict(evaluated[features], role)
        metrics = _role_metrics(true, pred)
        metrics["held_out_players"] = len(held_out)
        metrics["in_sample"] = in_sample
        per_role[role] = metrics

        suffix = "woba" if role == "batter" else "era"
        results[f"rmse_{suffix}"] = metrics["rmse"]
        results[f"mae_{suffix}"] = metrics["mae"]
        results[f"skill_{suffix}"] = metrics["skill_vs_mean"]

    _print_report(validation_year, per_role)
    logger.info(f"backtest_complete year={validation_year} results={results}")

    session.add(
        BacktestRun(
            validation_year=validation_year,
            rmse_woba=results.get("rmse_woba"),
            mae_woba=results.get("mae_woba"),
            rmse_era=results.get("rmse_era"),
            mae_era=results.get("mae_era"),
        )
    )
    session.commit()
    return {"validation_year": validation_year, "roles": per_role, **results}


def _print_report(validation_year: int, per_role: dict) -> None:
    print(f"\n--- SIMULATION RECOVERY METRICS — SEASON {validation_year} ---")
    print("Synthetic cohort with known ground truth; not real MLB predictive accuracy.\n")

    for role, metrics in per_role.items():
        name = METRIC_NAME[role]
        if "skipped" in metrics:
            print(f"{name:>5}: skipped ({metrics['skipped']})")
            continue
        scope = "IN-SAMPLE" if metrics.get("in_sample") else "held-out"
        print(
            f"{name:>5}: n={metrics['n']:<4} ({scope})  "
            f"RMSE={metrics['rmse']:.4f}  MAE={metrics['mae']:.4f}  "
            f"baseline(mean)={metrics['baseline_rmse']:.4f}  "
            f"skill={metrics['skill_vs_mean']:+.1%}"
        )
    print()
    print(json.dumps({k: v for k, v in per_role.items()}, indent=2, default=float))
