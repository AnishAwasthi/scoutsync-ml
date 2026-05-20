import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Comprehensive evaluation of SQLite activation flags
USE_SQLITE_ENV = os.getenv("USE_SQLITE", "false").lower().strip()
IS_SQLITE = USE_SQLITE_ENV in ("true", "1", "yes")

# Check Streamlit Cloud Secrets dictionary context directly
if not IS_SQLITE:
    try:
        import streamlit as st
        if "USE_SQLITE" in st.secrets:
            val = str(st.secrets["USE_SQLITE"]).lower().strip()
            if val in ("true", "1", "yes"):
                IS_SQLITE = True
    except Exception:
        pass

# 2. Bind the appropriate engine matching the resolved state
if IS_SQLITE:
    DATABASE_URL = "sqlite:///scoutsync.db"
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/scoutsync"
    )
    
    # Cloud Fail-Safe: Hot-swap to SQLite if running on a cloud server without an external DB
    if "localhost" in DATABASE_URL or "127.0.0.1" in DATABASE_URL:
        if os.getenv("HOME") == "/home/adminuser" or "STREAMLIT_SERVER_PORT" in os.environ:
            DATABASE_URL = "sqlite:///scoutsync.db"
            IS_SQLITE = True
            engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
        else:
            engine = create_engine(DATABASE_URL)
    else:
        engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Initializes schema components safely based on the active database driver"""
    if "sqlite" in str(engine.url) or engine.url.drivername == "sqlite":
        Base.metadata.create_all(bind=engine)
    else:
        try:
            with engine.begin() as conn:
                pass
        except Exception:
            Base.metadata.create_all(bind=engine)

# 3. Smart Session Provider to handle all call variants (Direct, Context, or Generator)
class SmartSessionWrapper:
    def __init__(self):
        self.db = SessionLocal()
        
    def __enter__(self):
        return self.db
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.db.close()
        
    def __getattr__(self, name):
        # Maps methods (.query, .add, .commit) directly if called as a normal object
        return getattr(self.db, name)
        
    def __iter__(self):
        # Maps generator/yield syntax if used as a FastAPI style dependency
        try:
            yield self.db
        finally:
            self.db.close()

def get_db_session():
    """Universal factory function expected by dashboard.py and backend pipelines"""
    return SmartSessionWrapper()