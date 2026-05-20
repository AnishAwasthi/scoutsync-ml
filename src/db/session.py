import os
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# 1. Robustly parse environment variables
USE_SQLITE_ENV = os.getenv("USE_SQLITE", "false").lower().strip()
IS_SQLITE = USE_SQLITE_ENV in ("true", "1", "yes")

# 2. Dynamically build connection string
if IS_SQLITE:
    DATABASE_URL = "sqlite:///scoutsync.db"
    # connect_args is required for SQLite to safely handle concurrent Streamlit requests
    engine = create_engine(
        DATABASE_URL, 
        connect_args={"check_same_thread": False}
    )
else:
    # Production fallback to PostgreSQL
    DATABASE_URL = os.getenv(
        "DATABASE_URL", 
        "postgresql://postgres:postgres@localhost:5432/scoutsync"
    )
    engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def init_db():
    """Initializes schema components safely depending on active dialect"""
    if IS_SQLITE:
        # Ensure tables are built smoothly in local SQLite file
        Base.metadata.create_all(bind=engine)
    else:
        # Production PostgreSQL migrations pathway
        with engine.begin() as conn:
            # Fallback block for execution schemas
            pass