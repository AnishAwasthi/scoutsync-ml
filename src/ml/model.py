"""ScoutSync XGBoost translation models."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor

from src.config import get_settings
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES
from src.logging_config import get_logger
from src.ml.shap_engine import explain_model

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.7,
    random_state=42,
)

GB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    random_state=42,
)


def _build_base_estimator():
    try:
        import xgboost as xgb

        return xgb.XGBRegressor(**XGB_PARAMS)
    except Exception as exc:
        get_logger().warning(f"xgboost_unavailable fallback=sklearn reason={exc}")
        return GradientBoostingRegressor(**GB_PARAMS)


class ScoutSyncTranslationModel:
    def __init__(self):
        self.pitcher_model: MultiOutputRegressor | None = None
        self.batter_model: MultiOutputRegressor | None = None
        self.residual_std: dict[str, float] = {}
        self.feature_names: dict[str, list[str]] = {
            "pitcher": PITCHER_FEATURES,
            "batter": BATTER_FEATURES,
        }

    def _make_estimator(self) -> MultiOutputRegressor:
        return MultiOutputRegressor(_build_base_estimator())

    def train_and_explain(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        role: str,
    ) -> tuple[MultiOutputRegressor, object, dict]:
        model = self._make_estimator()
        logger = get_logger()
        logger.info(
            f"training_start role={role} train_rows={len(X_train)} val_rows={len(X_val)} "
            f"params={XGB_PARAMS}"
        )
        model.fit(X_train.values, y_train)
        preds = model.predict(X_val.values)
        residuals = y_val - preds
        self.residual_std[role] = {
            "woba": float(np.nanstd(residuals[:, 0])) if residuals.shape[1] > 0 else 0.03,
            "era": float(np.nanstd(residuals[:, 1])) if residuals.shape[1] > 1 else 0.5,
            "variance": float(np.nanstd(residuals[:, 2])) if residuals.shape[1] > 2 else 0.02,
        }
        shap_values = explain_model(model, X_val, role)
        metrics = {
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "residual_std": self.residual_std[role],
        }
        logger.info(f"training_complete role={role} metrics={metrics}")
        if role == "pitcher":
            self.pitcher_model = model
        else:
            self.batter_model = model
        return model, shap_values, metrics

    def predict(self, X: pd.DataFrame, role: str) -> np.ndarray:
        model = self.pitcher_model if role == "pitcher" else self.batter_model
        if model is None:
            raise RuntimeError(f"Model for role={role} not trained")
        return model.predict(X.values)

    def save(self, path: Path | None = None) -> Path:
        path = path or Path(get_settings().model_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.pitcher_model:
            joblib.dump(self.pitcher_model, path / "pitcher_model.joblib")
        if self.batter_model:
            joblib.dump(self.batter_model, path / "batter_model.joblib")
        with open(path / "residual_std.json", "w", encoding="utf-8") as f:
            json.dump(self.residual_std, f)
        get_logger().info(f"models_saved path={path}")
        return path

    def load(self, path: Path | None = None) -> None:
        path = path or Path(get_settings().model_dir)
        pitcher_path = path / "pitcher_model.joblib"
        batter_path = path / "batter_model.joblib"
        if pitcher_path.exists():
            self.pitcher_model = joblib.load(pitcher_path)
        if batter_path.exists():
            self.batter_model = joblib.load(batter_path)
        std_path = path / "residual_std.json"
        if std_path.exists():
            with open(std_path, encoding="utf-8") as f:
                self.residual_std = json.load(f)


def prepare_targets(df: pd.DataFrame, role: str) -> np.ndarray:
    """Build Y matrix [proj_wOBA, proj_ERA, variance_delta]."""
    y = np.zeros((len(df), 3))
    if role == "batter":
        y[:, 0] = df.get("true_wOBA", df.get("proj_mlb_wOBA", 0.31)).fillna(0.31).values
        y[:, 1] = 4.5
        y[:, 2] = 0.02
    else:
        y[:, 0] = 0.31
        y[:, 1] = df.get("true_ERA", df.get("proj_mlb_ERA", 4.5)).fillna(4.5).values
        y[:, 2] = 0.02
    if "expected_variance_delta" in df.columns:
        y[:, 2] = df["expected_variance_delta"].fillna(0.02).values
    return y


def train_models_from_frames(
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    test_size: float = 0.2,
) -> ScoutSyncTranslationModel:
    """Train pitcher and batter models with player-level split."""
    translator = ScoutSyncTranslationModel()
    logger = get_logger()

    for role, frame, features in [
        ("pitcher", pitcher_df, PITCHER_FEATURES),
        ("batter", batter_df, BATTER_FEATURES),
    ]:
        labeled = frame.dropna(subset=features, how="any")
        if len(labeled) < 5:
            logger.warning(f"skipping_train role={role} insufficient_rows={len(labeled)}")
            continue
        y = prepare_targets(labeled, role)
        X = labeled[features]
        players = labeled["player_id"].values
        unique_players = np.unique(players)
        if len(unique_players) < 2:
            train_idx = labeled.index
            val_idx = labeled.index
        else:
            train_players, val_players = train_test_split(
                unique_players, test_size=test_size, random_state=42
            )
            train_mask = labeled["player_id"].isin(train_players)
            val_mask = labeled["player_id"].isin(val_players)
            X_train, X_val = X[train_mask], X[val_mask]
            y_train, y_val = y[train_mask.values], y[val_mask.values]

        translator.train_and_explain(X_train, y_train, X_val, y_val, role)

    translator.save()
    return translator
