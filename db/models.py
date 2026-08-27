"""SQLAlchemy ORM models mirroring db/schema.sql."""

from datetime import UTC, date, datetime

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class League(Base):
    __tablename__ = "leagues"

    league_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(100), nullable=False)
    abbreviation: Mapped[str] = mapped_column(String(20), nullable=False, unique=True)
    competition_tier: Mapped[int] = mapped_column(Integer, nullable=False)
    base_run_environment: Mapped[float] = mapped_column(Numeric(4, 2), default=4.50)

    stadiums: Mapped[list["StadiumEnvironment"]] = relationship(back_populates="league")


class StadiumEnvironment(Base):
    __tablename__ = "stadium_environments"

    stadium_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    league_id: Mapped[int | None] = mapped_column(ForeignKey("leagues.league_id"))
    stadium_name: Mapped[str] = mapped_column(String(150), nullable=False)
    altitude: Mapped[int] = mapped_column(Integer, default=0)
    park_factor_hr: Mapped[float] = mapped_column(Numeric(4, 2), default=1.00)
    park_factor_obp: Mapped[float] = mapped_column(Numeric(4, 2), default=1.00)
    temperature_mean: Mapped[float | None] = mapped_column(Numeric(4, 1))
    humidity_mean: Mapped[float | None] = mapped_column(Numeric(4, 1))

    league: Mapped["League | None"] = relationship(back_populates="stadiums")
    tracking_rows: Mapped[list["RawTrackingData"]] = relationship(back_populates="stadium")


class Player(Base):
    __tablename__ = "players"

    player_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    first_name: Mapped[str] = mapped_column(String(50), nullable=False)
    last_name: Mapped[str] = mapped_column(String(50), nullable=False)
    birth_date: Mapped[date] = mapped_column(Date, nullable=False)
    throws: Mapped[str | None] = mapped_column(String(1))
    bats: Mapped[str | None] = mapped_column(String(1))
    primary_position: Mapped[str | None] = mapped_column(String(3))

    tracking_rows: Mapped[list["RawTrackingData"]] = relationship(back_populates="player")
    projections: Mapped[list["MlbProjection"]] = relationship(back_populates="player")


class RawTrackingData(Base):
    __tablename__ = "raw_tracking_data"

    tracking_id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.player_id"))
    stadium_id: Mapped[int | None] = mapped_column(ForeignKey("stadium_environments.stadium_id"))
    game_date: Mapped[date] = mapped_column(Date, nullable=False)
    context_year: Mapped[int] = mapped_column(Integer, nullable=False)

    pitch_type: Mapped[str | None] = mapped_column(String(3))
    release_speed: Mapped[float | None] = mapped_column(Numeric(4, 1))
    spin_rate: Mapped[int | None] = mapped_column(Integer)
    vertical_break: Mapped[float | None] = mapped_column(Numeric(4, 1))
    horizontal_break: Mapped[float | None] = mapped_column(Numeric(4, 1))
    vertical_approach_angle: Mapped[float | None] = mapped_column(Numeric(4, 2))
    extension: Mapped[float | None] = mapped_column(Numeric(3, 2))
    plate_x: Mapped[float | None] = mapped_column(Numeric(4, 2))
    plate_z: Mapped[float | None] = mapped_column(Numeric(4, 2))

    exit_velocity: Mapped[float | None] = mapped_column(Numeric(4, 1))
    launch_angle: Mapped[float | None] = mapped_column(Numeric(4, 1))
    hit_distance: Mapped[int | None] = mapped_column(Integer)
    is_strikeout: Mapped[bool | None] = mapped_column(Boolean)
    is_walk: Mapped[bool | None] = mapped_column(Boolean)
    is_chase: Mapped[bool | None] = mapped_column(Boolean)

    player: Mapped["Player | None"] = relationship(back_populates="tracking_rows")
    stadium: Mapped["StadiumEnvironment | None"] = relationship(back_populates="tracking_rows")


class MlbProjection(Base):
    __tablename__ = "mlb_projections"

    projection_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    player_id: Mapped[int | None] = mapped_column(ForeignKey("players.player_id"))
    calculation_date: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))
    target_season: Mapped[int] = mapped_column(Integer, nullable=False)

    proj_wOBA: Mapped[float | None] = mapped_column(Numeric(4, 3))
    proj_wOBA_lower_90: Mapped[float | None] = mapped_column(Numeric(4, 3))
    proj_wOBA_upper_90: Mapped[float | None] = mapped_column(Numeric(4, 3))

    proj_ERA: Mapped[float | None] = mapped_column(Numeric(4, 2))
    proj_ERA_lower_90: Mapped[float | None] = mapped_column(Numeric(4, 2))
    proj_ERA_upper_90: Mapped[float | None] = mapped_column(Numeric(4, 2))

    shap_explainability_json: Mapped[dict | None] = mapped_column(JSON)

    player: Mapped["Player | None"] = relationship(back_populates="projections")


class BacktestRun(Base):
    __tablename__ = "backtest_runs"

    run_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    validation_year: Mapped[int] = mapped_column(Integer, nullable=False)
    rmse_woba: Mapped[float | None] = mapped_column(Numeric(8, 5))
    mae_woba: Mapped[float | None] = mapped_column(Numeric(8, 5))
    rmse_era: Mapped[float | None] = mapped_column(Numeric(8, 5))
    mae_era: Mapped[float | None] = mapped_column(Numeric(8, 5))
    run_at: Mapped[datetime] = mapped_column(DateTime, default=lambda: datetime.now(UTC))


class PlayerGroundTruth(Base):
    """
    Known outcome for a synthetic player, used only by the simulation backtest.

    The synthetic seeder draws a latent ``talent`` per player and uses it to generate
    both that player's tracking metrics and the ``true_*`` values stored here. The
    backtest therefore measures whether the pipeline recovers a signal it was given,
    not whether it predicts real major-league performance. Real players never get a row.
    """

    __tablename__ = "player_ground_truth"

    player_id: Mapped[int] = mapped_column(ForeignKey("players.player_id"), primary_key=True)
    role: Mapped[str] = mapped_column(String(10), nullable=False)
    latent_talent: Mapped[float] = mapped_column(Numeric(6, 4), nullable=False)
    true_wOBA: Mapped[float | None] = mapped_column(Numeric(4, 3))
    true_ERA: Mapped[float | None] = mapped_column(Numeric(4, 2))
    context_year: Mapped[int] = mapped_column(Integer, nullable=False)
