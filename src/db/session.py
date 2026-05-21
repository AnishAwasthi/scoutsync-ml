import os
import sys
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

# Global internal caches—left empty until explicitly invoked
_engine = None
_SessionLocal = None

def get_engine():
    """Lazy-loads the database engine only when active queries are run"""
    global _engine
    if _engine is not None:
        return _engine
        
    # Evaluate SQLite activation flags safely
    use_sqlite = os.getenv("USE_SQLITE", "false").lower().strip() in ("true", "1", "yes")
    
    try:
        import streamlit as st
        if "USE_SQLITE" in st.secrets:
            if str(st.secrets["USE_SQLITE"]).lower().strip() in ("true", "1", "yes"):
                use_sqlite = True
    except Exception:
        pass

    # Cloud Fail-Safe: Force SQLite if running on Streamlit Cloud containers
    if os.getenv("HOME") == "/home/adminuser" or "STREAMLIT_SERVER_PORT" in os.environ:
        use_sqlite = True

    if use_sqlite:
        DATABASE_URL = "sqlite:///scoutsync.db"
        _engine = create_engine(
            DATABASE_URL, 
            connect_args={"check_same_thread": False}
        )
    else:
        DATABASE_URL = os.getenv(
            "DATABASE_URL", 
            "postgresql://postgres:postgres@localhost:5432/scoutsync"
        )
        _engine = create_engine(DATABASE_URL)
        
    return _engine

def get_session_factory():
    """Lazy-loads the sessionmaker bound to the active engine"""
    global _SessionLocal
    if _SessionLocal is not None:
        return _SessionLocal
    _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal

def init_db():
    """Initializes schema components safely using the lazy engine"""
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
    """Universal Session proxy that instantiates sessions on demand"""
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
    """Universal function mapping expected by dashboard.py"""
    return SmartSessionWrapper()

class LazySessionLocal:
    """Compatibility layer that mimics the traditional SessionLocal callable object"""
    def __call__(self):
        return get_session_factory()()
    def __getattr__(self, name):
        return getattr(get_session_factory(), name)

# Mapped globally so dashboard.py script bindings remain intact
SessionLocal = LazySessionLocal()