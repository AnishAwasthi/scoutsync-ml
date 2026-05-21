import os
from pathlib import Path

from sqlalchemy import Column, Date, Float, Integer, String, Text, create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Explicitly define self-contained models to guarantee flawless table initialization on the cloud
class Player(Base):
    __tablename__ = "players"
    player_id = Column(Integer, primary_key=True)
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

def get_engine():
    global _engine
    if _engine is not None:
        return _engine
        
    use_sqlite = os.getenv("USE_SQLITE", "false").lower().strip() in ("true", "1", "yes")
    try:
        import streamlit as st
        if "USE_SQLITE" in st.secrets:
            if str(st.secrets["USE_SQLITE"]).lower().strip() in ("true", "1", "yes"):
                use_sqlite = True
    except Exception:
        pass

    if os.getenv("HOME") == "/home/adminuser" or "STREAMLIT_SERVER_PORT" in os.environ:
        use_sqlite = True

    if use_sqlite:
        DATABASE_URL = "sqlite:///scoutsync.db"
        _engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
    else:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/scoutsync")
        _engine = create_engine(DATABASE_URL)
        
    return _engine

def get_session_factory():
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

def init_db():
    engine = get_engine()
    if "sqlite" in str(engine.url) or engine.url.drivername == "sqlite":
        Base.metadata.create_all(bind=engine)
    else:
        try:
            with engine.begin() as conn:
                pass
        except Exception:
            Base.metadata.create_all(bind=engine)

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
    return "sqlite" in str(get_engine().url)


def sqlite_db_path() -> Path:
    return Path("scoutsync.db").resolve()


def check_db_connection() -> bool:
    """Verify the database engine can execute a simple query."""
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False