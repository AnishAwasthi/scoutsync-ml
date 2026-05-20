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


def use_sqlite() -> bool:
    """True when USE_SQLITE env is set to 'true' (case-insensitive)."""
    return os.getenv("USE_SQLITE", "false").lower() == "true"


def sqlite_db_path() -> Path:
    """Resolved filesystem path for scoutsync.db (relative to process cwd)."""
    return Path("scoutsync.db").resolve()


def resolve_database_url() -> str:
    """Return SQLite URL or PostgreSQL URL from settings; never mixes both."""
    if use_sqlite():
        return SQLITE_DATABASE_URL
    return get_settings().database_url


def get_engine():
    global _engine
    if _engine is None:
        url = resolve_database_url()
        kwargs: dict = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            kwargs["pool_pre_ping"] = False
        else:
            kwargs["pool_pre_ping"] = True
        _engine = create_engine(url, **kwargs)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), autoflush=False, autocommit=False)
    return _SessionLocal


def get_db_session() -> Session:
    return get_session_factory()()


def init_db() -> None:
    """Apply schema.sql (PostgreSQL) or ORM create_all (SQLite)."""
    engine = get_engine()
    if use_sqlite():
        Base.metadata.create_all(bind=engine)
        _seed_reference_leagues_sqlite()
        get_logger().info("database_initialized backend=sqlite url=scoutsync.db")
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
    get_logger().info("database_initialized schema=db/schema.sql backend=postgresql")


def _seed_reference_leagues_sqlite() -> None:
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


def _split_sql_statements(sql: str) -> list[str]:
    """Split SQL file on semicolons outside of simple comment blocks."""
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


def get_logger():
    from src.logging_config import get_logger as _get

    return _get()
