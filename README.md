# ScoutSync ML

Cross-league baseball performance translation platform. Ingests non-MLB tracking data (NCAA TrackMan-style, synthetic amateur feeds) and translates metrics into projected MLB baselines using environmental normalization, league-quality adjustment, and XGBoost multi-output models with SHAP explainability.

## Prerequisites

- Python 3.11+
- Docker (for PostgreSQL)
- ~500MB disk for pybaseball cache on first run (optional; falls back to synthetic MLB data offline)

## Quick Start

```bash
cd ~/Projects/scoutsync-ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
docker compose up -d   # or: USE_SQLITE=true for local SQLite dev without Docker

python main.py init-db
python main.py seed
python main.py train
python main.py backtest --year 2024
python main.py project --player-id 1
python main.py serve
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `init-db` | Apply `db/schema.sql` and ORM tables |
| `seed` | Load pybaseball Statcast sample + synthetic NCAA/CCL tracking |
| `train` | Normalize → features → XGBoost → persist `mlb_projections` |
| `backtest --year 2024` | Out-of-sample RMSE/MAE vs rookie outcomes |
| `project --player-id N` | Print projection + distribution breakdown |
| `serve` | FastAPI on port 8000 |

## Streamlit Dashboard

Run the interactive UI without starting FastAPI:

```bash
USE_SQLITE=true streamlit run dashboard.py
```

Use the sidebar to search/select a player and switch between batter (wOBA) and pitcher (ERA) projections. The main panel shows profile info, 90% confidence metric cards, an outcome density curve, and a SHAP contribution chart.

## API Endpoints

- `GET /health` — database connectivity
- `POST /upload/trackman` — multipart CSV upload
- `POST /upload/json` — JSON array of tracking records
- `GET /players/{id}/projection` — latest MLB projection + SHAP JSON
- `GET /players/{id}/breakdown` — raw vs adjusted distributions for charts

### Example

```bash
curl http://localhost:8000/health
curl -F "file=@data/samples/trackman_sample.csv" http://localhost:8000/upload/trackman
curl http://localhost:8000/players/1/projection
```

## Architecture

1. **Ingestion** — FastAPI uploads, pybaseball Statcast, synthetic amateur seed
2. **Normalization** — altitude/temperature/humidity physics + league tier Z-scores
3. **ML Core** — `MultiOutputRegressor` + XGBoost; SHAP `TreeExplainer`
4. **Validation** — historical rookie backtest with RMSE/MAE logging

## Testing

```bash
python -m pytest tests/ -v
USE_SQLITE=true python main.py init-db && USE_SQLITE=true python main.py seed && USE_SQLITE=true python main.py train && USE_SQLITE=true python main.py backtest
```

## XGBoost on macOS

If `libomp` is missing, install with `brew install libomp`. The trainer automatically falls back to scikit-learn `GradientBoostingRegressor` when XGBoost cannot load.

## Data Notes

- First `seed` may download Statcast via pybaseball (cached under `data/pybaseball_cache/`).
- Without network, seed uses synthetic MLB Statcast and rookie outcome fallbacks.
- Amateur NCAA/CCL data is always synthetic for demo purposes.

## Project Layout

See plan: `db/` (schema + ORM), `src/` (ingestion, normalization, features, ml, validation), `main.py` (CLI), `tests/`.
