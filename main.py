#!/usr/bin/env python3
"""ScoutSync ML monolith CLI entrypoint."""

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

if os.getenv("USE_SQLITE", "").lower() in ("1", "true", "yes"):
    os.environ["USE_SQLITE"] = "true"  # session.py checks lower() == "true"

from db.models import MlbProjection
from src.config import get_settings
from src.db.session import get_db_session, init_db
from src.ingestion.pybaseball_loader import seed_mlb_statcast
from src.ingestion.synthetic_seed import seed_synthetic_data
from src.logging_config import setup_logging
from src.ml.model import ScoutSyncTranslationModel
from src.pipeline import get_player_breakdown, run_train_pipeline
from src.validation.backtest import run_historical_validation


def cmd_init_db(_: argparse.Namespace) -> None:
    init_db()
    print("Database initialized.")


def cmd_seed(args: argparse.Namespace) -> None:
    session = get_db_session()
    try:
        mlb_result = seed_mlb_statcast(session, validation_year=args.year)
        syn_result = seed_synthetic_data(session, num_players=args.players)
        print(json.dumps({"mlb": mlb_result, "synthetic": syn_result}, indent=2))
    finally:
        session.close()


def cmd_train(_: argparse.Namespace) -> None:
    session = get_db_session()
    try:
        model = run_train_pipeline(session)
        print("Training complete. Models saved.")
        print(
            json.dumps(
                {
                    "backend": model.backend,
                    "pitcher_model": model.pitcher_model is not None,
                    "batter_model": model.batter_model is not None,
                    "residual_std": model.residual_std,
                },
                indent=2,
            )
        )
    finally:
        session.close()


def cmd_backtest(args: argparse.Namespace) -> None:
    model = ScoutSyncTranslationModel()
    model.load()
    session = get_db_session()
    try:
        run_historical_validation(model, session, validation_year=args.year)
    finally:
        session.close()


def cmd_project(args: argparse.Namespace) -> None:
    session = get_db_session()
    try:
        proj = (
            session.query(MlbProjection)
            .filter_by(player_id=args.player_id)
            .order_by(MlbProjection.calculation_date.desc())
            .first()
        )
        if proj is None:
            print(f"No projection for player {args.player_id}. Run: python main.py train")
        else:
            print(
                json.dumps(
                    {
                        "player_id": args.player_id,
                        "proj_wOBA": float(proj.proj_wOBA) if proj.proj_wOBA is not None else None,
                        "proj_wOBA_90ci": [
                            float(proj.proj_wOBA_lower_90) if proj.proj_wOBA_lower_90 is not None else None,
                            float(proj.proj_wOBA_upper_90) if proj.proj_wOBA_upper_90 is not None else None,
                        ],
                        "proj_ERA": float(proj.proj_ERA) if proj.proj_ERA is not None else None,
                        "proj_ERA_90ci": [
                            float(proj.proj_ERA_lower_90) if proj.proj_ERA_lower_90 is not None else None,
                            float(proj.proj_ERA_upper_90) if proj.proj_ERA_upper_90 is not None else None,
                        ],
                        "shap": proj.shap_explainability_json,
                    },
                    indent=2,
                    default=float,
                )
            )
        breakdown = get_player_breakdown(session, args.player_id)
        print(json.dumps({"breakdown": breakdown}, indent=2, default=float))
    finally:
        session.close()


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run(
        "src.ingestion.fastapi_app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


def main() -> None:
    setup_logging()
    parser = argparse.ArgumentParser(description="ScoutSync ML — Cross-League Baseball Translation")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="Apply DDL and create tables").set_defaults(func=cmd_init_db)

    seed_p = sub.add_parser("seed", help="Load pybaseball + synthetic amateur data")
    seed_p.add_argument("--year", type=int, default=get_settings().validation_year)
    seed_p.add_argument(
        "--players",
        type=int,
        default=120,
        help="Synthetic amateur cohort size; split evenly between pitchers and batters",
    )
    seed_p.set_defaults(func=cmd_seed)

    sub.add_parser("train", help="Normalize, feature-build, train XGBoost, persist projections").set_defaults(
        func=cmd_train
    )

    bt = sub.add_parser("backtest", help="Historical out-of-sample validation")
    bt.add_argument("--year", type=int, default=get_settings().validation_year)
    bt.set_defaults(func=cmd_backtest)

    pr = sub.add_parser("project", help="Show projection and breakdown for a player")
    pr.add_argument("--player-id", type=int, required=True)
    pr.set_defaults(func=cmd_project)

    srv = sub.add_parser("serve", help="Start FastAPI server")
    srv.add_argument("--host", default="0.0.0.0")
    srv.add_argument("--port", type=int, default=8000)
    srv.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
