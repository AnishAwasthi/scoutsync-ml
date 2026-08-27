"""ScoutSync translation models: one regressor per role.

Each role predicts exactly one metric -- batters predict wOBA, pitchers predict ERA.
An earlier version wrapped both in a ``MultiOutputRegressor`` with a 3-column target
where two columns were constants; that produced a pitcher model that returned 4.50 for
every player, a zero-width confidence interval, and all-zero SHAP values (SHAP read
``estimators_[0]``, which for pitchers was fit on a constant). One estimator per role
removes that whole class of bug.
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from src.config import get_settings
from src.features.builder import BATTER_FEATURES, PITCHER_FEATURES
from src.logging_config import get_logger
from src.ml.shap_engine import explain_model, summarize_shap

ROLES = ("pitcher", "batter")

# Label column each role is trained against.
TARGET_COLUMN = {"batter": "true_wOBA", "pitcher": "true_ERA"}

# Fallback residual spread when a role has too few validation rows to estimate one.
DEFAULT_RESIDUAL_STD = {"batter": 0.030, "pitcher": 0.500}

MIN_TRAIN_ROWS = 5

XGB_PARAMS = dict(
    objective="reg:squarederror",
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
)

GB_PARAMS = dict(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    random_state=42,
)


def _build_estimator():
    """XGBoost when importable, else sklearn gradient boosting.

    The fallback is not silent: it logs at WARNING and ``backend_name()`` reports which
    one is in use, so the dashboard and CLI can say so rather than implying XGBoost ran.
    """
    try:
        import xgboost as xgb

        return xgb.XGBRegressor(**XGB_PARAMS), "xgboost"
    except Exception as exc:
        get_logger().warning(f"xgboost_unavailable fallback=sklearn reason={exc}")
        return GradientBoostingRegressor(**GB_PARAMS), "sklearn_gbr"


def backend_name() -> str:
    """Which gradient-boosting implementation is actually available here."""
    return _build_estimator()[1]


class ScoutSyncTranslationModel:
    def __init__(self):
        self.pitcher_model = None
        self.batter_model = None
        self.residual_std: dict[str, float] = {}
        self.backend: str | None = None
        # Player ids held out of training, per role. The backtest scores only these.
        self.validation_players: dict[str, list[int]] = {}
        self.feature_names: dict[str, list[str]] = {
            "pitcher": PITCHER_FEATURES,
            "batter": BATTER_FEATURES,
        }

    def get_estimator(self, role: str):
        return self.pitcher_model if role == "pitcher" else self.batter_model

    def set_estimator(self, role: str, estimator) -> None:
        if role == "pitcher":
            self.pitcher_model = estimator
        else:
            self.batter_model = estimator

    def train_and_explain(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: pd.DataFrame,
        y_val: np.ndarray,
        role: str,
    ) -> tuple[object, object, dict]:
        estimator, backend = _build_estimator()
        self.backend = backend
        logger = get_logger()
        logger.info(
            f"training_start role={role} backend={backend} target={TARGET_COLUMN[role]} "
            f"train_rows={len(X_train)} val_rows={len(X_val)}"
        )
        estimator.fit(X_train.values, y_train)

        preds = estimator.predict(X_val.values)
        residuals = np.asarray(y_val, dtype=float) - np.asarray(preds, dtype=float)
        std = float(np.nanstd(residuals))
        if not np.isfinite(std) or std <= 0:
            std = DEFAULT_RESIDUAL_STD[role]
            logger.warning(f"residual_std_degenerate role={role} using_default={std}")
        self.residual_std[role] = std

        shap_values = explain_model(estimator, X_val)
        metrics = {
            "backend": backend,
            "train_rows": len(X_train),
            "val_rows": len(X_val),
            "residual_std": std,
            "val_target_std": float(np.nanstd(y_val)),
            "top_features": summarize_shap(shap_values, list(X_val.columns), top_n=3),
        }
        logger.info(f"training_complete role={role} metrics={metrics}")
        self.set_estimator(role, estimator)
        return estimator, shap_values, metrics

    def predict(self, X: pd.DataFrame, role: str) -> np.ndarray:
        estimator = self.get_estimator(role)
        if estimator is None:
            raise RuntimeError(f"Model for role={role} is not trained")
        return np.asarray(estimator.predict(X.values), dtype=float)

    def residual_for(self, role: str) -> float:
        return self.residual_std.get(role, DEFAULT_RESIDUAL_STD[role])

    def save(self, path: Path | None = None) -> Path:
        path = path or Path(get_settings().model_dir)
        path.mkdir(parents=True, exist_ok=True)
        if self.pitcher_model is not None:
            joblib.dump(self.pitcher_model, path / "pitcher_model.joblib")
        if self.batter_model is not None:
            joblib.dump(self.batter_model, path / "batter_model.joblib")
        with open(path / "residual_std.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "residual_std": self.residual_std,
                    "backend": self.backend,
                    "validation_players": self.validation_players,
                },
                f,
                indent=2,
            )
        get_logger().info(f"models_saved path={path}")
        return path

    def load(self, path: Path | None = None) -> None:
        path = path or Path(get_settings().model_dir)
        for role, filename in (("pitcher", "pitcher_model.joblib"), ("batter", "batter_model.joblib")):
            model_path = path / filename
            if model_path.exists():
                self.set_estimator(role, joblib.load(model_path))

        std_path = path / "residual_std.json"
        if not std_path.exists():
            return
        with open(std_path, encoding="utf-8") as f:
            payload = json.load(f)
        # Tolerate the pre-refactor file shape, which stored a dict per role.
        residuals = payload.get("residual_std", payload) if isinstance(payload, dict) else {}
        cleaned: dict[str, float] = {}
        for role, value in residuals.items():
            if isinstance(value, dict):
                value = value.get("woba" if role == "batter" else "era")
            if isinstance(value, (int, float)) and float(value) > 0:
                cleaned[role] = float(value)
        self.residual_std = cleaned
        if isinstance(payload, dict):
            self.backend = payload.get("backend")
            self.validation_players = {
                role: [int(pid) for pid in ids]
                for role, ids in (payload.get("validation_players") or {}).items()
            }


def prepare_targets(df: pd.DataFrame, role: str) -> np.ndarray:
    """Return the 1-D label vector for ``role``.

    Raises rather than substituting a constant: a constant target silently trains a model
    that predicts the same number for everyone, which is exactly the failure this
    refactor exists to prevent.
    """
    column = TARGET_COLUMN[role]
    if column not in df.columns:
        raise KeyError(
            f"Label column '{column}' missing for role={role}. "
            "Ground-truth labels are attached in pipeline.attach_training_labels."
        )
    values = pd.to_numeric(df[column], errors="coerce")
    if values.isna().any():
        raise ValueError(
            f"{int(values.isna().sum())} unlabeled rows for role={role}; "
            "drop them before calling prepare_targets."
        )
    return values.to_numpy(dtype=float)


def _split_by_player(labeled: pd.DataFrame, test_size: float, role: str):
    """Player-level split so no player appears in both train and validation."""
    logger = get_logger()
    unique_players = np.unique(labeled["player_id"].values)

    if len(unique_players) < 2:
        logger.warning(
            f"single_player_split role={role} players={len(unique_players)}; "
            "validating on training rows, residual_std will be optimistic"
        )
        return labeled, labeled

    train_players, val_players = train_test_split(
        unique_players, test_size=test_size, random_state=42
    )
    train = labeled[labeled["player_id"].isin(train_players)]
    val = labeled[labeled["player_id"].isin(val_players)]
    if train.empty or val.empty:
        logger.warning(f"degenerate_split role={role}; falling back to full-frame validation")
        return labeled, labeled
    return train, val


def train_models_from_frames(
    pitcher_df: pd.DataFrame,
    batter_df: pd.DataFrame,
    test_size: float = 0.2,
) -> ScoutSyncTranslationModel:
    """Train one model per role using a player-level train/validation split."""
    translator = ScoutSyncTranslationModel()
    logger = get_logger()

    for role, frame, features in [
        ("pitcher", pitcher_df, PITCHER_FEATURES),
        ("batter", batter_df, BATTER_FEATURES),
    ]:
        target = TARGET_COLUMN[role]
        if frame.empty or target not in frame.columns:
            logger.warning(f"skipping_train role={role} reason=no_labeled_frame")
            continue

        labeled = frame.dropna(subset=list(features) + [target], how="any")
        if len(labeled) < MIN_TRAIN_ROWS:
            logger.warning(f"skipping_train role={role} insufficient_rows={len(labeled)}")
            continue

        train, val = _split_by_player(labeled, test_size, role)
        held_out = sorted({int(pid) for pid in val["player_id"]} - {int(pid) for pid in train["player_id"]})
        translator.validation_players[role] = held_out
        if not held_out:
            logger.warning(
                f"no_held_out_players role={role}; backtest for this role will be in-sample"
            )
        translator.train_and_explain(
            train[features],
            prepare_targets(train, role),
            val[features],
            prepare_targets(val, role),
            role,
        )

    if translator.pitcher_model is None and translator.batter_model is None:
        raise RuntimeError(
            "No model could be trained. Check that seeding produced labeled players: "
            "python main.py seed"
        )

    translator.save()
    return translator
