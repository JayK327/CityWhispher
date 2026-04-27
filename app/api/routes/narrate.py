"""
POST /narrate
The main E2E pipeline endpoint. This is where all 5 modules connect.

Request flow:
  1. [Module 3] Check Redis cache for POI data at this location
  2. [Module 1] If cache miss: fetch Overpass + Wikipedia in parallel
  3. [Module 4] Load user preference weights from DB (or cold-start defaults)
  4. [Module 3] Select best POI based on confidence + user preferences
  5. [Module 2] Render Jinja2 prompt → call LLM → parse structured output
  6.            Synthesize TTS audio
  7.            Return NarrationResponse with script + audio_url + latency breakdown
"""
import time
import structlog
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.database import get_db
from app.models.poi import NarrationRequest, NarrationResponse, Tone
from app.services.module3_cache.lookahead import fetch_and_enrich_pois, select_best_poi
from app.services.module4_personalization.preference import get_or_create_preferences
from app.services.module2_llm.generator import generate_narration, generate_regional_fallback
from app.tts.synthesizer import synthesize

log = structlog.get_logger()
router = APIRouter()


@router.post("/narrate", response_model=NarrationResponse)
async def narrate(
    req: NarrationRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Full pipeline: GPS coordinates → enriched POI → LLM script → audio MP3.
    """
    pipeline_start = time.perf_counter()
    timings = {}

    # ── Step 1: Load user preferences ────────────────────────────────────────
    t = time.perf_counter()
    user_weights = None
    if req.user_id:
        prefs = await get_or_create_preferences(req.user_id, db)
        user_weights = prefs.weights
        tone = Tone(prefs.tone) if prefs.tone in Tone._value2member_map_ else req.tone
    else:
        tone = req.tone
    timings["prefs_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # ── Step 2: Fetch + enrich POIs (Module 1 + 3) ───────────────────────────
    t = time.perf_counter()
    records, fetch_timings = await fetch_and_enrich_pois(req.lat, req.lon)
    timings.update(fetch_timings)
    timings["fetch_total_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # ── Step 3: Select best POI (Module 3 + 4) ───────────────────────────────
    best_poi = select_best_poi(records, user_weights)

    # ── Step 4: Generate narration (Module 2) ────────────────────────────────
    t = time.perf_counter()
    fallback_used = False

    if best_poi:
        try:
            script_result, prompt_tokens = await generate_narration(best_poi, tone)
            timings["prompt_tokens"] = prompt_tokens
            poi_name = best_poi.name
            category = best_poi.category.value
        except Exception as e:
            log.warning("generation_failed", error=str(e), fallback=True)
            script_result = await generate_regional_fallback(req.lat, req.lon, tone)
            poi_name = "Nearby area"
            category = "unknown"
            fallback_used = True
    else:
        # No POI met the confidence threshold — use regional fallback
        script_result = await generate_regional_fallback(req.lat, req.lon, tone)
        poi_name = "Nearby area"
        category = "unknown"
        fallback_used = True

    timings["llm_ms"] = round((time.perf_counter() - t) * 1000, 1)

    # ── Step 5: TTS synthesis ─────────────────────────────────────────────────
    t = time.perf_counter()
    audio_path = await synthesize(script_result.script)
    timings["tts_ms"] = round((time.perf_counter() - t) * 1000, 1)

    timings["total_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 1)

    log.info(
        "narration_complete",
        poi=poi_name,
        fallback=fallback_used,
        total_ms=timings["total_ms"],
    )

    return NarrationResponse(
        poi_name=poi_name,
        category=category,
        script=script_result.script,
        word_count=script_result.word_count,
        confidence=script_result.confidence.value,
        audio_url=audio_path,
        latency_ms=timings,
        fallback_used=fallback_used,
    )
