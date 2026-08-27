"""Shared fixtures. Every test runs against a throwaway SQLite file, never a real DB."""

import os
from pathlib import Path

import pytest

# Must be set before any src.* import so config and session pick SQLite up.
os.environ["USE_SQLITE"] = "true"


@pytest.fixture
def isolated_model_dir(tmp_path, monkeypatch) -> Path:
    """Redirect model persistence so tests never overwrite data/models."""
    from src.config import get_settings

    model_dir = tmp_path / "models"
    model_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(get_settings(), "model_dir", str(model_dir), raising=False)
    return model_dir


@pytest.fixture
def scratch_db(tmp_path, monkeypatch):
    """A fresh, fully-migrated SQLite database scoped to one test."""
    from src.db import session as session_module

    monkeypatch.setenv("SCOUTSYNC_SQLITE_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("USE_SQLITE", "true")
    session_module.reset_engine()
    session_module.init_db()

    db = session_module.get_session_factory()()
    try:
        yield db
    finally:
        # The engine may already be disposed if the test swapped databases.
        try:
            db.close()
        except Exception:
            pass
        session_module.reset_engine()


@pytest.fixture
def seeded_db(scratch_db, isolated_model_dir):
    """A scratch database holding a small synthetic cohort with ground truth."""
    from src.ingestion.synthetic_seed import seed_synthetic_data

    seed_synthetic_data(scratch_db, num_players=40)
    return scratch_db


@pytest.fixture
def fresh_database_factory(tmp_path):
    """Build additional independent databases inside one test."""
    from src.db import session as session_module

    created = []

    def _make(name: str):
        os.environ["SCOUTSYNC_SQLITE_PATH"] = str(tmp_path / f"{name}.db")
        session_module.reset_engine()
        session_module.init_db()
        db = session_module.get_session_factory()()
        created.append(db)
        return db

    yield _make

    for db in created:
        try:
            db.close()
        except Exception:
            pass
    session_module.reset_engine()
