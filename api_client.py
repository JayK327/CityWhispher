"""
api_client.py  (project root)

Bridge between the Streamlit UI and the FastAPI backend.
Imports pipeline logic DIRECTLY from app/ — no code duplication.

Two modes:
  live_api    → FastAPI is running → calls POST /narrate, POST /signal
  standalone  → FastAPI not running → imports from app/ and runs locally
"""

import httpx, asyncio, os, time, json, re, hashlib
from dataclasses import dataclass, field
from typing import Optional

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000")
TIMEOUT = 30.0

CATEGORY_COLORS = {
    "historical": "#818CF8", "cultural": "#34D399",
    "commercial": "#F87171", "nature":   "#6EE7B7",
    "food":       "#FBBF24", "unknown":  "#6B7280",
}


@dataclass
class POIResult:
    poi_name: str;       category: str;    confidence: str
    confidence_score: float;               description: str
    address: str;        opening_hours: str; source_count: int
    source_url: str;     poi_id: str;      lat: float; lon: float


@dataclass
class NarrationResult:
    poi: POIResult;  script: str;     word_count: int
    confidence: str; audio_url: Optional[str]
    latency_ms: dict = field(default_factory=dict)
    fallback_used: bool = False
    prompt_tokens: int = 0
    mode: str = "standalone"          # "live_api" | "standalone" | "error"
    error: Optional[str] = None


# ── Health check ──────────────────────────────────────────────────────────────
async def check_backend_health() -> dict:
    url = os.environ.get("API_BASE_URL", API_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            r = await client.get(f"{url}/health")
            if r.status_code == 200:
                d = r.json()
                return {"reachable": True, "redis": d.get("redis","unknown"), "url": url}
    except Exception:
        pass
    return {"reachable": False, "url": url}


# ── Narration ─────────────────────────────────────────────────────────────────
async def narrate(lat, lon, tone, user_id, weights) -> NarrationResult:
    url = os.environ.get("API_BASE_URL", API_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(f"{url}/narrate",
                json={"lat": lat, "lon": lon, "tone": tone, "user_id": user_id})
            resp.raise_for_status()
            d = resp.json()
            poi = POIResult(
                poi_name=d["poi_name"], category=d.get("category","unknown"),
                confidence=d["confidence"],
                confidence_score=d.get("confidence_score", _lvl(d["confidence"])),
                description=d.get("description",""), address=d.get("address",""),
                opening_hours=d.get("opening_hours",""), source_count=d.get("source_count",1),
                source_url=d.get("source_url",""), poi_id=d.get("poi_id",""),
                lat=lat, lon=lon,
            )
            return NarrationResult(
                poi=poi, script=d["script"], word_count=d["word_count"],
                confidence=d["confidence"], audio_url=d.get("audio_url"),
                latency_ms=d.get("latency_ms",{}), fallback_used=d.get("fallback_used",False),
                prompt_tokens=d.get("latency_ms",{}).get("prompt_tokens",0), mode="live_api",
            )
    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.RemoteProtocolError):
        return await _run_standalone(lat, lon, tone, weights)
    except Exception as e:
        return await _run_standalone(lat, lon, tone, weights)


# ── Signal ────────────────────────────────────────────────────────────────────
async def send_signal(user_id, poi_id, category, action, local_weights) -> list:
    url = os.environ.get("API_BASE_URL", API_BASE_URL)
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.post(f"{url}/signal",
                json={"user_id":user_id,"poi_id":poi_id,"category":category,"action":action})
            resp.raise_for_status()
            w = resp.json().get("updated_weights", {})
            return [w.get("historical",local_weights[0]), w.get("cultural",local_weights[1]),
                    w.get("commercial",local_weights[2]), w.get("nature",local_weights[3]),
                    w.get("food",local_weights[4])]
    except Exception:
        # Fall back to local update using app/ logic directly
        from app.services.module4_personalization.preference import update_preference_weights
        return update_preference_weights(local_weights, category, action)


# ── Standalone — runs pipeline directly from app/ ─────────────────────────────
async def _run_standalone(lat, lon, tone, weights) -> NarrationResult:
    """
    Runs the full pipeline locally by importing directly from app/.
    No duplication — same code the FastAPI backend uses.
    """
    import time
    from app.services.module1_ingestion.overpass  import fetch_overpass_pois
    from app.services.module1_ingestion.wikipedia import fetch_wikipedia_summary
    from app.services.module1_ingestion.normalizer import normalize_batch
    from app.services.module3_cache.lookahead     import select_best_poi
    from app.services.module2_llm.generator       import generate_narration, generate_regional_fallback
    from app.models.poi                           import Tone
    from app.tts.synthesizer                      import synthesize
    from app.config                               import settings

    timings = {}
    try:
        tone_enum = Tone(tone)
    except Exception:
        tone_enum = Tone.informative

    # Fetch
    t = time.perf_counter()
    osm_records = await fetch_overpass_pois(lat, lon, settings.poi_search_radius_m)
    timings["overpass_ms"] = round((time.perf_counter()-t)*1000)

    if not osm_records:
        script = await generate_regional_fallback(lat, lon, tone_enum)
        audio  = await synthesize(script.script)
        dummy  = _dummy_poi(lat, lon)
        return NarrationResult(poi=dummy, script=script.script, word_count=script.word_count,
            confidence="low", audio_url=audio, latency_ms=timings, fallback_used=True, mode="standalone")

    # Wikipedia
    t = time.perf_counter()
    wiki_results = await asyncio.gather(
        *[fetch_wikipedia_summary(r.name) for r in osm_records], return_exceptions=True)
    timings["wikipedia_ms"] = round((time.perf_counter()-t)*1000)
    wiki_map = {osm.name: (w if isinstance(w, object) and not isinstance(w, Exception) else None)
                for osm, w in zip(osm_records, wiki_results)}

    # Normalize + score
    t = time.perf_counter()
    records = normalize_batch(osm_records, wiki_map)
    timings["normalize_ms"] = round((time.perf_counter()-t)*1000)

    # Select
    best = select_best_poi(records, weights, settings.confidence_threshold)
    if not best:
        script = await generate_regional_fallback(lat, lon, tone_enum)
        audio  = await synthesize(script.script)
        dummy  = _dummy_poi(lat, lon)
        return NarrationResult(poi=dummy, script=script.script, word_count=script.word_count,
            confidence="low", audio_url=audio, latency_ms=timings, fallback_used=True, mode="standalone")

    # Generate
    t = time.perf_counter()
    script_result, tokens = await generate_narration(best, tone_enum)
    timings["llm_ms"] = round((time.perf_counter()-t)*1000)
    timings["prompt_tokens"] = tokens

    # TTS
    t = time.perf_counter()
    audio = await synthesize(script_result.script)
    timings["tts_ms"] = round((time.perf_counter()-t)*1000)
    timings["total_ms"] = sum(v for k,v in timings.items() if k.endswith("_ms"))

    poi = POIResult(
        poi_name=best.name, category=best.category.value,
        confidence=script_result.confidence.value,
        confidence_score=best.confidence_score,
        description=best.description or "", address=best.address or "",
        opening_hours=best.opening_hours or "", source_count=best.source_count,
        source_url=best.source_url or "", poi_id=best.poi_id, lat=lat, lon=lon,
    )
    return NarrationResult(poi=poi, script=script_result.script,
        word_count=script_result.word_count, confidence=script_result.confidence.value,
        audio_url=audio, latency_ms=timings, fallback_used=False,
        prompt_tokens=tokens, mode="standalone")


def _lvl(level): return {"high":0.88,"medium":0.60,"low":0.30}.get(level,0.50)

def _dummy_poi(lat, lon): return POIResult(
    poi_name="Nearby area", category="unknown", confidence="low",
    confidence_score=0.0, description="", address="", opening_hours="",
    source_count=0, source_url="", poi_id="fallback", lat=lat, lon=lon)
