"""Historical out-of-sample backtest engine."""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sqlalchemy.orm import Session

from db.models import BacktestRun
from src.config import get_settings
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES
from src.ingestion.pybaseball_loader import load_rookie_outcomes
from src.logging_config import get_logger
from src.ml.model import ScoutSyncTranslationModel
from src.pipeline import attach_training_labels, load_tracking_dataframe, normalize_tracking_df
from src.features.builder import build_training_frames


def run_historical_validation(
    model: ScoutSyncTranslationModel,
    session: Session,
    validation_year: int = 2024,
) -> dict:
    """
    Validate model by projecting amateur data and comparing to MLB rookie outcomes.
    """
    logger = get_logger()
    raw = load_tracking_dataframe(session)
    amateur = raw[raw["competition_tier"] >= 4]
    amateur = amateur[amateur["context_year"] < validation_year]
    if amateur.empty:
        logger.warning("no_amateur_data_for_backtest")
        amateur = raw[raw["competition_tier"] >= 4]

    normalized = normalize_tracking_df(amateur)
    _, pitchers, batters, _, _ = build_training_frames(normalized, pd.DataFrame())

    amateur_ids = amateur["player_id"].dropna().unique().tolist()
    pitchers, batters = attach_training_labels(
        pitchers, batters, amateur_ids, validation_year
    )
    actual = load_rookie_outcomes(validation_year)

    results: dict = {}
    z_90 = 1.645

    for role, frame, features in [
        ("batter", batters, BATTER_FEATURES),
        ("pitcher", pitchers, PITCHER_FEATURES),
    ]:
        est = model.pitcher_model if role == "pitcher" else model.batter_model
        if est is None:
            continue
        labeled = frame.dropna(subset=features + (["true_wOBA"] if role == "batter" else ["true_ERA"]), how="any")
        if labeled.empty:
            continue
        preds = est.predict(labeled[features].values)
        if role == "batter":
            true = labeled["true_wOBA"].astype(float).values
            pred = preds[:, 0]
            rmse = float(np.sqrt(mean_squared_error(true, pred)))
            mae = float(mean_absolute_error(true, pred))
            results["rmse_woba"] = rmse
            results["mae_woba"] = mae
        else:
            true = labeled["true_ERA"].astype(float).values
            pred = preds[:, 1]
            rmse = float(np.sqrt(mean_squared_error(true, pred)))
            mae = float(mean_absolute_error(true, pred))
            results["rmse_era"] = rmse
            results["mae_era"] = mae

    print(f"--- BACKTEST VALIDATION METRICS FOR SEASON {validation_year} ---")
    if "rmse_woba" in results:
        print(f"Root Mean Squared Error vs. True MLB wOBA: {results['rmse_woba']:.4f}")
        print(f"Mean Absolute Error vs. True MLB wOBA: {results['mae_woba']:.4f}")
    if "rmse_era" in results:
        print(f"Root Mean Squared Error vs. True MLB ERA: {results['rmse_era']:.4f}")
        print(f"Mean Absolute Error vs. True MLB ERA: {results['mae_era']:.4f}")

    logger.info(f"backtest_complete year={validation_year} results={results}")
    print(json.dumps(results))

    run = BacktestRun(
        validation_year=validation_year,
        rmse_woba=results.get("rmse_woba"),
        mae_woba=results.get("mae_woba"),
        rmse_era=results.get("rmse_era"),
        mae_era=results.get("mae_era"),
    )
    session.add(run)
    session.commit()
    return results
