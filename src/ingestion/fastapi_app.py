"""FastAPI ingestion and projection read API."""

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from db.models import MlbProjection, RawTrackingData
from src.db.session import check_db_connection, get_db_session
from src.ingestion.parsers import (
    dataframe_to_tracking_records,
    parse_json_tracking,
    parse_trackman_csv,
)
from src.pipeline import get_player_breakdown

app = FastAPI(title="ScoutSync ML", version="1.0.0")


class TrackingRecord(BaseModel):
    player_id: int
    stadium_id: int | None = None
    game_date: str
    context_year: int
    release_speed: float | None = None
    exit_velocity: float | None = None
    pitch_type: str | None = None


@app.get("/health")
def health():
    ok = check_db_connection()
    return {"status": "ok" if ok else "degraded", "database": ok}


@app.post("/upload/trackman")
async def upload_trackman(file: UploadFile = File(...)):
    content = await file.read()
    df = parse_trackman_csv(content)
    records = dataframe_to_tracking_records(df)
    session = get_db_session()
    try:
        for rec in records:
            session.add(RawTrackingData(**rec))
        session.commit()
        return {"inserted": len(records)}
    finally:
        session.close()


@app.post("/upload/json")
async def upload_json(records: list[TrackingRecord]):
    df = parse_json_tracking([r.model_dump() for r in records])
    rows = dataframe_to_tracking_records(df)
    session = get_db_session()
    try:
        for rec in rows:
            session.add(RawTrackingData(**rec))
        session.commit()
        return {"inserted": len(rows)}
    finally:
        session.close()


@app.get("/players/{player_id}/projection")
def get_projection(player_id: int):
    session = get_db_session()
    try:
        proj = (
            session.query(MlbProjection)
            .filter_by(player_id=player_id)
            .order_by(MlbProjection.calculation_date.desc())
            .first()
        )
        if not proj:
            raise HTTPException(status_code=404, detail="Projection not found")
        return {
            "player_id": player_id,
            "target_season": proj.target_season,
            "proj_wOBA": float(proj.proj_wOBA) if proj.proj_wOBA is not None else None,
            "proj_wOBA_lower_90": float(proj.proj_wOBA_lower_90) if proj.proj_wOBA_lower_90 else None,
            "proj_wOBA_upper_90": float(proj.proj_wOBA_upper_90) if proj.proj_wOBA_upper_90 else None,
            "proj_ERA": float(proj.proj_ERA) if proj.proj_ERA is not None else None,
            "proj_ERA_lower_90": float(proj.proj_ERA_lower_90) if proj.proj_ERA_lower_90 else None,
            "proj_ERA_upper_90": float(proj.proj_ERA_upper_90) if proj.proj_ERA_upper_90 else None,
            "shap_explainability_json": proj.shap_explainability_json,
        }
    finally:
        session.close()


@app.get("/players/{player_id}/breakdown")
def get_breakdown(player_id: int):
    session = get_db_session()
    try:
        data = get_player_breakdown(session, player_id)
        if data.get("error"):
            raise HTTPException(status_code=404, detail=data["error"])
        return data
    finally:
        session.close()
