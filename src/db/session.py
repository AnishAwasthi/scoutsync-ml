import os
from pathlib import Path

from sqlalchemy import Column, Date, Float, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Project root: src/db/session.py -> parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[2]

Base = declarative_base()

# Explicitly define self-contained models to guarantee flawless table initialization on the cloud
class Player(Base):
    __tablename__ = "players"
    player_id = Column(Integer, primary_key=True, autoincrement=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    birth_date = Column(Date, nullable=True)
    throws = Column(String, nullable=True)
    bats = Column(String, nullable=True)
    primary_position = Column(String, nullable=True)

class MlbProjection(Base):
    __tablename__ = "mlb_projections"
    id = Column(Integer, primary_key=True, autoincrement=True)
    player_id = Column(Integer, nullable=False)
    target_season = Column(Integer, nullable=False)
    proj_wOBA = Column(Float, nullable=True)
    proj_wOBA_lower_90 = Column(Float, nullable=True)
    proj_wOBA_upper_90 = Column(Float, nullable=True)
    proj_ERA = Column(Float, nullable=True)
    proj_ERA_lower_90 = Column(Float, nullable=True)
    proj_ERA_upper_90 = Column(Float, nullable=True)
    shap_explainability_json = Column(Text, nullable=True)

# Global internal caches—left empty until explicitly invoked
_engine = None
_SessionLocal = None
_sqlite_path: Path | None = None


def _resolve_sqlite_path() -> Path:
    """Absolute DB path in the app directory (writable on Streamlit Cloud)."""
    global _sqlite_path
    if _sqlite_path is None:
        _sqlite_path = (PROJECT_ROOT / "scoutsync.db").resolve()
        _sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    return _sqlite_path


def _sqlite_database_url() -> str:
    return f"sqlite:///{_resolve_sqlite_path()}"


def get_engine():
    global _engine
    if _engine is not None:
        return _engine

    if use_sqlite():
        db_url = _sqlite_database_url()
        _engine = create_engine(
            db_url,
            connect_args={"check_same_thread": False, "timeout": 30},
            poolclass=StaticPool,
        )
    else:
        db_url = os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:postgres@localhost:5432/scoutsync",
        )
        _engine = create_engine(db_url, pool_pre_ping=True)

    return _engine

def get_session_factory():
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

def _tables_present(engine) -> bool:
    inspector = inspect(engine)
    return inspector.has_table("players") and inspector.has_table("mlb_projections")


def init_db() -> None:
    """Create dashboard tables if missing; safe to call on every Streamlit rerun."""
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        if _tables_present(engine):
            return
        try:
            Base.metadata.create_all(bind=engine, checkfirst=True)
        except OperationalError as exc:
            msg = str(exc).lower()
            if "already exists" in msg:
                return
            # Stale/corrupt file from an old schema — reset and retry once
            if _resolve_sqlite_path().exists():
                _resolve_sqlite_path().unlink(missing_ok=True)
            Base.metadata.create_all(bind=engine, checkfirst=True)
        return

    try:
        with engine.begin() as conn:
            conn.execute(text("SELECT 1"))
    except Exception:
        Base.metadata.create_all(bind=engine, checkfirst=True)

class SmartSessionWrapper:
    def __init__(self):
        factory = get_session_factory()
        self.db = factory()
        
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
            secret_val = str(st.secrets["USE_SQLITE"]).lower().strip()
            if secret_val in ("true", "1", "yes"):
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