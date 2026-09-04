# ScoutSync ML ⚾

> A cross-league baseball translation pipeline: normalize amateur tracking data for park
> and competition effects, project MLB-baseline outcomes with confidence intervals, and
> explain every projection with signed SHAP attributions.

[![CI](https://github.com/AnishAwasthi/scoutsync-ml/actions/workflows/ci.yml/badge.svg)](https://github.com/AnishAwasthi/scoutsync-ml/actions/workflows/ci.yml)
[![Live Demo](https://img.shields.io/badge/Demo-Streamlit_Cloud-FF4B4B?logo=streamlit)](https://scoutsync-ml-sumemxdakhf3zpuagqcyae.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Design doc:** [docs/scoutsync-ml-design.md](docs/scoutsync-ml-design.md)

---

**The amateur data in this project is synthetic, and the model is validated against a
simulation — not against real major-league outcomes.**

Translating amateur performance to MLB requires knowing what happened to each amateur
player *after* they were drafted. No public dataset provides that player-level linkage.
Rather than invent one, this project generates a synthetic cohort from a known process
and measures whether the pipeline can recover it:

- Each synthetic player is drawn from a latent **talent** parameter.
- That talent drives **both** their pitch-level tracking metrics **and** their stored
  ground-truth outcome (`player_ground_truth`).
- Park bias is injected as the exact inverse of the environmental adjustment, so a
  correct normalization recovers the latent value.
- The backtest scores **held-out players** the model never trained on.

So the reported skill is a real, honest measurement of a real pipeline — of *signal
recovery on generated data*. It is **not** evidence that this predicts real prospects.
Statcast rows are genuine and are used only to set MLB league baselines.

If you're evaluating this as an engineering project, that framing is the point: the
interesting work is the pipeline, the normalization physics, the explainability layer,
and the test suite that keeps them honest.

---

## The problem

Evaluating talent across leagues is confounded by environment and competition. A curveball
at 5,000 feet breaks less than the same pitch at sea level. A fly ball carries farther. A
.400 wOBA against weak competition is not a .400 wOBA against strong competition. Raw
tracking numbers are not comparable across contexts.

ScoutSync normalizes those effects out before modeling:

1. **Environmental** — moist-air density from the barometric formula corrects the two
   metrics that air density actually drives (see below).
2. **Competition** — `M_norm = ((M_adj − μ_league) / σ_league) × γ_tier`, with γ falling
   from 1.00 (MLB) to 0.45 (Cape Cod).
3. **Projection** — one gradient-boosted regressor per role predicts wOBA (batters) or
   ERA (pitchers), with a 90% interval from held-out residual spread.
4. **Explanation** — signed per-player SHAP attributions showing which factors moved
   that specific projection, and in which direction.

### Which metrics are actually park-sensitive

Getting this right matters more than adjusting everything. Air density affects the ball
*in flight*, so it moves some metrics a lot, some barely, and some not at all:

| Metric | Air-density effect | What the pipeline does |
|---|---|---|
| **Break / movement** | Magnus force scales with density. Coors ≈ 82% of sea-level density → an 18″ break becomes ~14.8″ | Scale **up** in thin air, by `ρ_ref / ρ_park` |
| **Batted-ball carry** | Reduced drag → ~5% farther at Coors | Scale **down** in thin air |
| **Release speed** | None. `release_speed` is measured out of the hand; drag acts *after* release. The ball arrives ~1 mph faster at Coors — a plate-speed effect on a release-speed number | **Nothing** |
| **Exit velocity** | None. Measured off the bat; the collision does not involve the air | **Nothing** |
| **Spin rate** | None. Imparted by the hand | **Nothing** |

The model's density function reproduces the published Coors figure — 0.83 against a
reported 0.82 — using the ISA barometric formula plus a proper vapour-pressure humidity
term. Physics reference: Alan Nathan,
[Baseball At High Altitude](https://baseball.physics.illinois.edu/Denver.html).

This is not a cosmetic detail. Park-adjusted carry is the **single most important batter
feature** by SHAP importance, and correcting the physics raised held-out wOBA skill from
+25.5% to +38.2%.

## Quickstart

Requires Python 3.11+.

```bash
git clone https://github.com/AnishAwasthi/scoutsync-ml.git
cd scoutsync-ml
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Run the whole pipeline against a local SQLite file:

```bash
export USE_SQLITE=true
python main.py init-db      # create schema + reference leagues
python main.py seed         # MLB baselines + 120-player synthetic cohort
python main.py train        # normalize -> features -> fit -> persist projections
python main.py backtest     # score held-out players
```

Then open the dashboard:

```bash
USE_SQLITE=true streamlit run dashboard.py
```

Inspect a single player's projection, interval, and SHAP breakdown:

```bash
USE_SQLITE=true python main.py project --player-id 25
```

### PostgreSQL instead of SQLite

```bash
docker compose up -d
export DATABASE_URL=postgresql://scoutsync:scoutsync@localhost:5432/scoutsync
python main.py init-db
```

`init-db` applies [`db/schema.sql`](db/schema.sql) on PostgreSQL and creates the same
schema from ORM metadata on SQLite.

## Sample output

From a clean `seed → train → backtest` on the default 120-player cohort:

```
--- SIMULATION RECOVERY METRICS — SEASON 2024 ---
Synthetic cohort with known ground truth; not real MLB predictive accuracy.

 wOBA: n=12   (held-out)  RMSE=0.0164  MAE=0.0137  baseline(mean)=0.0266  skill=+38.2%
  ERA: n=12   (held-out)  RMSE=0.2817  MAE=0.2268  baseline(mean)=0.8804  skill=+68.0%
```

`skill` is the reduction in RMSE against a predict-the-cohort-mean baseline, which is the
number worth reading — raw RMSE looks good even for a model that has learned nothing.
Seeding is deterministic, so these numbers reproduce exactly.

## Architecture

```
main.py                    CLI: init-db | seed | train | backtest | project | serve
dashboard.py               Streamlit UI (self-bootstraps: seeds and trains on cold start)
db/
  schema.sql               PostgreSQL DDL
  models.py                SQLAlchemy ORM — single source of truth for the schema
src/
  ingestion/
    pybaseball_loader.py   Statcast -> MLB tier-1 baselines (cached, offline fallback)
    synthetic_seed.py      Latent-talent cohort generator + ground truth
    lite_seed.py           Network-free seeding for hosted demo and CI
    fastapi_app.py         REST endpoints
  normalization/
    environment.py         Barometric moist-air density; break and carry corrections
    league_context.py      μ/σ baselines and tier-weighted z-scores
  features/builder.py      Pitch/batted-ball rows -> player-season feature matrices
  ml/
    model.py               One regressor per role; player-level train/val split
    shap_engine.py         Signed SHAP attributions
  validation/backtest.py   Held-out scoring with a mean baseline
  dashboard/               Data access and Plotly figures
```

**Design notes worth calling out:**

- **Adjust only what physics says to adjust.** Break and carry are corrected; release
  speed, exit velocity, and spin rate are explicitly passed through, with
  `adjust_release_speed` kept as a documented identity so a reader finds the reasoning
  where they would look for the adjustment.
- **One estimator per role.** Batters predict wOBA, pitchers predict ERA. An earlier
  design used a multi-output regressor whose extra target columns were constants; it
  produced a pitcher model that returned 4.50 for every player with a zero-width
  interval. See `tests/test_model.py::test_regression_pitcher_predictions_are_not_constant`.
- **Player-level splitting.** Held-out player ids are persisted with the model artifact
  so the backtest is genuinely out-of-sample rather than mostly in-sample fit.
- **Labels raise, never default.** `prepare_targets` refuses missing labels instead of
  filling a constant, because a constant target silently trains a useless model.
- **Signed SHAP.** Contributions keep direction. Aggregating to `mean(|SHAP|)` for a
  per-player chart throws away the only thing that makes it an explanation.
- **Graceful backend fallback.** XGBoost when available, scikit-learn gradient boosting
  otherwise — logged at WARNING and surfaced in the UI, never silently.

## Testing

```bash
pip install -r requirements-dev.txt
USE_SQLITE=true pytest
ruff check .
```

81 tests covering normalization physics, the model layer, SHAP payload semantics, the REST
API, and the full seed → train → project pipeline against a throwaway SQLite database.

The physics tests pin direction *and* magnitude against published values, not just "the
number changed" — an earlier version's density function ignored altitude entirely and
still passed a looser test. A large share of the rest are named `test_regression_*` and
pin specific bugs that shipped earlier: constant pitcher predictions, zero-width
intervals, unsigned SHAP, `UnboundLocalError` on single-player frames, projections
accumulating across retrains.
[`tests/test_no_fabricated_output.py`](tests/test_no_fabricated_output.py) statically
asserts the dashboard never constructs a projection from numeric literals, so the
hosted demo cannot regress to displaying invented numbers.

## Limitations

- **No real amateur→MLB validation.** Discussed above; the central limitation.
- **Small cohort.** The default 120-player seed leaves ~12 held-out players per role, so
  backtest metrics carry wide uncertainty. Seed more players for a tighter estimate.
- **Linearized carry model.** The carry sensitivity (`0.28`) is derived from published
  Coors figures and linearized around the reference density. Real carry depends on launch
  angle and exit velocity too — a 100 mph ball at 33° gains far more than the average.
- **Park factors are unused.** `park_factor_hr` / `park_factor_obp` are stored but do
  not yet feed the adjustment; only altitude, temperature, and humidity do.
- **No wind or batted-ball spin.** Both matter for carry and neither is modeled.
- **Tier coefficients are assumed.** The γ values encode an assumed ordering of league
  difficulty rather than a measured one.

## License

[MIT](LICENSE)
