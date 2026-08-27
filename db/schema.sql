-- ScoutSync ML PostgreSQL Schema

-- 1. Leagues & Levels Reference
CREATE TABLE IF NOT EXISTS leagues (
    league_id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    abbreviation VARCHAR(20) NOT NULL UNIQUE,
    competition_tier INT NOT NULL CHECK (competition_tier BETWEEN 1 AND 5),
    base_run_environment NUMERIC(4,2) DEFAULT 4.50
);

-- 2. Environmental & Stadium Context
CREATE TABLE IF NOT EXISTS stadium_environments (
    stadium_id SERIAL PRIMARY KEY,
    league_id INT REFERENCES leagues(league_id),
    stadium_name VARCHAR(150) NOT NULL,
    altitude INT DEFAULT 0 CHECK (altitude >= -500 AND altitude <= 12000),
    park_factor_hr NUMERIC(4,2) DEFAULT 1.00 CHECK (park_factor_hr BETWEEN 0.50 AND 1.50),
    park_factor_obp NUMERIC(4,2) DEFAULT 1.00 CHECK (park_factor_obp BETWEEN 0.50 AND 1.50),
    temperature_mean NUMERIC(4,1),
    humidity_mean NUMERIC(4,1) CHECK (humidity_mean IS NULL OR (humidity_mean BETWEEN 0 AND 100))
);

-- 3. Core Player Manifest
CREATE TABLE IF NOT EXISTS players (
    player_id SERIAL PRIMARY KEY,
    first_name VARCHAR(50) NOT NULL,
    last_name VARCHAR(50) NOT NULL,
    birth_date DATE NOT NULL,
    throws VARCHAR(1) CHECK (throws IN ('L', 'R')),
    bats VARCHAR(1) CHECK (bats IN ('L', 'R', 'S')),
    primary_position VARCHAR(3)
);

-- 4. Granular In-Game Tracking Data
CREATE TABLE IF NOT EXISTS raw_tracking_data (
    tracking_id BIGSERIAL PRIMARY KEY,
    player_id INT REFERENCES players(player_id),
    stadium_id INT REFERENCES stadium_environments(stadium_id),
    game_date DATE NOT NULL,
    context_year INT NOT NULL,

    pitch_type VARCHAR(3),
    release_speed NUMERIC(4,1) CHECK (release_speed IS NULL OR (release_speed BETWEEN 40 AND 110)),
    spin_rate INT CHECK (spin_rate IS NULL OR (spin_rate BETWEEN 500 AND 3500)),
    vertical_break NUMERIC(4,1),
    horizontal_break NUMERIC(4,1),
    vertical_approach_angle NUMERIC(4,2),
    extension NUMERIC(3,2) CHECK (extension IS NULL OR (extension BETWEEN 3 AND 8)),
    plate_x NUMERIC(4,2),
    plate_z NUMERIC(4,2),

    exit_velocity NUMERIC(4,1) CHECK (exit_velocity IS NULL OR (exit_velocity BETWEEN 30 AND 120)),
    launch_angle NUMERIC(4,1) CHECK (launch_angle IS NULL OR (launch_angle BETWEEN -90 AND 90)),
    hit_distance INT CHECK (hit_distance IS NULL OR (hit_distance BETWEEN 0 AND 550)),
    is_strikeout BOOLEAN,
    is_walk BOOLEAN,
    is_chase BOOLEAN
);

CREATE INDEX IF NOT EXISTS idx_raw_tracking_player ON raw_tracking_data(player_id);
CREATE INDEX IF NOT EXISTS idx_raw_tracking_game_date ON raw_tracking_data(game_date);

-- 5. Outbound Projected Translations
CREATE TABLE IF NOT EXISTS mlb_projections (
    projection_id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(player_id),
    calculation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    target_season INT NOT NULL,

    proj_wOBA NUMERIC(4,3),
    proj_wOBA_lower_90 NUMERIC(4,3),
    proj_wOBA_upper_90 NUMERIC(4,3),

    proj_ERA NUMERIC(4,2),
    proj_ERA_lower_90 NUMERIC(4,2),
    proj_ERA_upper_90 NUMERIC(4,2),

    shap_explainability_json JSONB
);

CREATE INDEX IF NOT EXISTS idx_mlb_projections_player_season ON mlb_projections(player_id, target_season);

-- 6. Backtest audit (ORM helper)
CREATE TABLE IF NOT EXISTS backtest_runs (
    run_id SERIAL PRIMARY KEY,
    validation_year INT NOT NULL,
    rmse_woba NUMERIC(8,5),
    mae_woba NUMERIC(8,5),
    rmse_era NUMERIC(8,5),
    mae_era NUMERIC(8,5),
    run_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 7. Simulation ground truth (synthetic cohort only)
-- Each synthetic amateur player is generated from a latent "talent" parameter that
-- drives BOTH their tracking metrics and their true MLB outcome. Storing that outcome
-- here lets the backtest measure whether the pipeline recovers a known signal.
-- No real player has a row in this table.
CREATE TABLE IF NOT EXISTS player_ground_truth (
    player_id INT PRIMARY KEY REFERENCES players(player_id),
    role VARCHAR(10) NOT NULL CHECK (role IN ('batter', 'pitcher')),
    latent_talent NUMERIC(6,4) NOT NULL,
    true_wOBA NUMERIC(4,3),
    true_ERA NUMERIC(4,2),
    context_year INT NOT NULL
);

-- Reference league seed data
INSERT INTO leagues (name, abbreviation, competition_tier, base_run_environment)
VALUES
    ('Major League Baseball', 'MLB', 1, 4.60),
    ('Nippon Professional Baseball', 'NPB', 2, 4.20),
    ('Korea Baseball Organization', 'KBO', 3, 4.80),
    ('NCAA Division I', 'NCAA', 4, 5.50),
    ('Cape Cod Baseball League', 'CCL', 5, 4.90)
ON CONFLICT (abbreviation) DO NOTHING;
