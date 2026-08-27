"""FastAPI surface tests. Requires httpx (installed alongside pytest)."""

import pytest

pytest.importorskip("httpx", reason="httpx is required by fastapi.testclient")

from fastapi.testclient import TestClient  # noqa: E402

from src.ingestion.fastapi_app import app  # noqa: E402


@pytest.fixture
def client(scratch_db):
    """A client bound to a throwaway database (the app reads the module-level engine)."""
    return TestClient(app)


def test_health_reports_database_connectivity(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] is True


def test_missing_projection_returns_404(client):
    assert client.get("/players/99999/projection").status_code == 404


def test_projection_endpoint_returns_a_trained_projection(seeded_db, isolated_model_dir):
    from db.models import MlbProjection
    from src.pipeline import run_train_pipeline

    run_train_pipeline(seeded_db)
    player_id = seeded_db.query(MlbProjection).first().player_id

    response = TestClient(app).get(f"/players/{player_id}/projection")
    assert response.status_code == 200

    body = response.json()
    assert body["player_id"] == player_id
    # Exactly one metric is populated, matching the player's role.
    assert (body["proj_wOBA"] is None) != (body["proj_ERA"] is None)


def test_breakdown_endpoint_reports_raw_and_adjusted(seeded_db, isolated_model_dir):
    from src.pipeline import run_train_pipeline

    run_train_pipeline(seeded_db)
    response = TestClient(app).get("/players/1/breakdown")
    assert response.status_code == 200

    body = response.json()
    assert "raw_distribution" in body
    assert "adjusted_distribution" in body
