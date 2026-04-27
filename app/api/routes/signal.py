"""
POST /signal
Receives implicit user feedback signals from the client.

Signals:
  skip     → user pressed skip within 5 seconds of audio starting
  complete → user listened to the full clip without skipping
  replay   → user pressed replay button

These signals update the user's preference weight vector in PostgreSQL.
No explicit rating UI — all implicit, which is appropriate for a driving context.
"""
import structlog
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.poi import SignalRequest
from app.services.module4_personalization.preference import apply_signal

log = structlog.get_logger()
router = APIRouter()


@router.post("/signal")
async def record_signal(
    req: SignalRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Apply a user feedback signal to their preference vector.

    Example request body:
    {
        "user_id": "user_abc123",
        "poi_id": "osm_12345678",
        "category": "historical",
        "action": "skip"
    }
    """
    try:
        updated_prefs = await apply_signal(
            user_id=req.user_id,
            category=req.category.value,
            action=req.action,
            db=db,
        )
        log.info(
            "signal_applied",
            user_id=req.user_id,
            category=req.category.value,
            action=req.action,
        )
        return {
            "status": "ok",
            "user_id": req.user_id,
            "updated_weights": {
                "historical": round(updated_prefs.weights[0], 3),
                "cultural":   round(updated_prefs.weights[1], 3),
                "commercial": round(updated_prefs.weights[2], 3),
                "nature":     round(updated_prefs.weights[3], 3),
                "food":       round(updated_prefs.weights[4], 3),
            },
            "dominant_category": updated_prefs.dominant_category(),
        }
    except Exception as e:
        log.error("signal_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
