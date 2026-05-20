"""Database engine and session management."""

import os
from pathlib import Path

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from db.models import Base
from src.config import PROJECT_ROOT, get_settings

_engine = None
_SessionLocal = None

SQLITE_DATABASE_URL = "sqlite:///scoutsync.db"


def _sqlite_enabled() -> bool:
    """Resolve SQLite mode from env, Streamlit secrets, or cloud auto-fallback."""
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


def use_sqlite() -> bool:
    """Public check aligned with dashboard bootstrap (`lower() == 'true'` or cloud)."""
    if os.getenv("USE_SQLITE", "false").lower() == "true":
        return True
    return _sqlite_enabled()


def sqlite_db_path() -> Path:
    return Path("scoutsync.db").resolve()


def resolve_database_url() -> str:
    if use_sqlite() or _sqlite_enabled():
        return SQLITE_DATABASE_URL
    return os.getenv(
        "DATABASE_URL",
        get_settings().database_url,
    )


def get_engine():
    global _engine
    if _engine is None:
        url = resolve_database_url()
        if url.startswith("sqlite"):
            _engine = create_engine(
                url,
                connect_args={"check_same_thread": False},
            )
        else:
            _engine = create_engine(url, pool_pre_ping=True)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def get_db_session() -> Session:
    return get_session_factory()()


def seed_reference_leagues() -> None:
    from db.models import League

    session = get_session_factory()()
    try:
        if session.query(League).count() == 0:
            for name, abbr, tier, env in [
                ("Major League Baseball", "MLB", 1, 4.60),
                ("Nippon Professional Baseball", "NPB", 2, 4.20),
                ("Korea Baseball Organization", "KBO", 3, 4.80),
                ("NCAA Division I", "NCAA", 4, 5.50),
                ("Cape Cod Baseball League", "CCL", 5, 4.90),
            ]:
                session.add(
                    League(
                        name=name,
                        abbreviation=abbr,
                        competition_tier=tier,
                        base_run_environment=env,
                    )
                )
            session.commit()
    finally:
        session.close()


def init_db() -> None:
    """Create ORM tables; PostgreSQL also applies schema.sql."""
    engine = get_engine()
    url = str(engine.url)

    if url.startswith("sqlite") or engine.dialect.name == "sqlite":
        Base.metadata.create_all(bind=engine)
        seed_reference_leagues()
        _get_logger().info("database_initialized backend=sqlite")
        return

    schema_path = PROJECT_ROOT / "db" / "schema.sql"
    with open(schema_path, encoding="utf-8") as f:
        sql = f.read()
    with engine.begin() as conn:
        for statement in _split_sql_statements(sql):
            stmt = statement.strip()
            if stmt:
                conn.execute(text(stmt))
        Base.metadata.create_all(bind=conn)
    _get_logger().info("database_initialized backend=postgresql")


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    for line in sql.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        current.append(line)
        if stripped.endswith(";"):
            statements.append("\n".join(current))
            current = []
    if current:
        joined = "\n".join(current).strip()
        if joined:
            statements.append(joined)
    return statements


def check_db_connection() -> bool:
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _get_logger():
    from src.logging_config import get_logger

    return get_logger()
