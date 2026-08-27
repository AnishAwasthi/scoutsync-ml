"""Engine, session, and schema bootstrap for ScoutSync.

There is exactly one schema (``db.models.Base``). SQLite and PostgreSQL differ only
in how the tables get created: PostgreSQL runs ``db/schema.sql``, SQLite uses ORM
metadata (the DDL file uses SERIAL/JSONB, which SQLite does not understand).
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from db.models import Base, League, MlbProjection, Player  # re-exported for callers

# Project root: src/db/session.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

REFERENCE_LEAGUES = [
    ("Major League Baseball", "MLB", 1, 4.60),
    ("Nippon Professional Baseball", "NPB", 2, 4.20),
    ("Korea Baseball Organization", "KBO", 3, 4.80),
    ("NCAA Division I", "NCAA", 4, 5.50),
    ("Cape Cod Baseball League", "CCL", 5, 4.90),
]

_engine = None
_SessionLocal = None
_sqlite_path: Path | None = None


def _resolve_sqlite_path() -> Path:
    """Absolute DB path, overridable so tests and tooling can use a scratch file."""
    global _sqlite_path
    if _sqlite_path is None:
        override = os.getenv("SCOUTSYNC_SQLITE_PATH")
        _sqlite_path = (Path(override) if override else PROJECT_ROOT / "scoutsync.db").resolve()
        _sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return _sqlite_path


def _sqlite_database_url() -> str:
    return f"sqlite:///{_resolve_sqlite_path()}"


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    if use_sqlite():
        _engine = create_engine(
            _sqlite_database_url(),
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=StaticPool,
        )
    else:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://scoutsync:scoutsync@localhost:5432/scoutsync",
        )
        _engine = create_engine(db_url, pool_pre_ping=True)

    return _engine


def reset_engine() -> None:
    """Drop cached engine/session/path so a new DATABASE_URL or path takes effect."""
    global _engine, _SessionLocal, _sqlite_path
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    _sqlite_path = None


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def seed_reference_leagues() -> int:
    """Insert the five reference leagues if absent. Idempotent."""
    session = get_session_factory()()
    try:
        existing = {abbr for (abbr,) in session.query(League.abbreviation).all()}
        added = 0
        for name, abbr, tier, run_env in REFERENCE_LEAGUES:
            if abbr in existing:
                continue
            session.add(
                League(
                    name=name,
                    abbreviation=abbr,
                    competition_tier=tier,
                    base_run_environment=run_env,
                )
            )
            added += 1
        session.commit()
        return added
    finally:
        session.close()


def init_db() -> None:
    """Create the full schema and seed reference leagues. Safe to call repeatedly."""
    engine = get_engine()

    if engine.dialect.name == "postgresql":
        ddl = (PROJECT_ROOT / "db" / "schema.sql").read_text(encoding="utf-8")
        with engine.begin() as conn:
            conn.execute(text(ddl))
    else:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        seed_reference_leagues()


class SmartSessionWrapper:
    """Session that also works as a context manager and as a FastAPI dependency."""

    def __init__(self):
        self.db = get_session_factory()()

    def __enter__(self):
        return self.db

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()

    def __getattr__(self, name):
        return getattr(self.db, name)

    def __iter__(self):
        try:
            yield self.db
        finally:
            self.db.close()


def get_db_session():
    return SmartSessionWrapper()


class LazySessionLocal:
    def __call__(self):
        return get_session_factory()()

    def __getattr__(self, name):
        return getattr(get_session_factory(), name)


SessionLocal = LazySessionLocal()


def use_sqlite() -> bool:
    """True when SQLite mode is active (env, Streamlit secrets, or cloud auto-detect)."""
    val = os.getenv("USE_SQLITE", "false").lower().strip()
    if val in ("true", "1", "yes"):
        return True
    try:
        import streamlit as st

        if "USE_SQLITE" in st.secrets:
            if str(st.secrets["USE_SQLITE"]).lower().strip() in ("true", "1", "yes"):
                return True
    except Exception:
        pass
    if os.getenv("HOME") == "/home/adminuser" or "STREAMLIT_SERVER_PORT" in os.environ:
        return True
    return False


def sqlite_db_path() -> Path:
    return _resolve_sqlite_path()


def check_db_connection() -> bool:
    """Verify the database engine can execute a simple query."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def has_tracking_schema() -> bool:
    """True when the pitch-level tracking table exists."""
    try:
        return inspect(get_engine()).has_table("raw_tracking_data")
    except Exception:
        return False


__all__ = [
    "Base",
    "League",
    "MlbProjection",
    "Player",
    "SessionLocal",
    "check_db_connection",
    "get_db_session",
    "get_engine",
    "get_session_factory",
    "has_tracking_schema",
    "init_db",
    "reset_engine",
    "seed_reference_leagues",
    "sqlite_db_path",
    "use_sqlite",
]
