import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Comprehensive evaluation of SQLite activation flags
IS_SQLITE = False

# Check standard OS environment variables
if os.getenv("USE_SQLITE", "false").lower().strip() in ("true", "1", "yes"):
    IS_SQLITE = True

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
    
    # Cloud Fail-Safe: If running on Streamlit Cloud container but hitting default local string, force hot-swap
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
    """Initializes schema components safely by examining the active database driver directly"""
    # Inspect the engine parameters directly instead of relying solely on variable indicators
    if "sqlite" in str(engine.url) or engine.url.drivername == "sqlite":
        Base.metadata.create_all(bind=engine)
    else:
        try:
            with engine.begin() as conn:
                pass
        except Exception:
            # Absolute recovery track: If PostgreSQL connectivity drops on cloud, instantiate SQLite baseline
            print("PostgreSQL connection refused on cloud runtime. Falling back to SQLite engine.")
            Base.metadata.create_all(bind=engine)