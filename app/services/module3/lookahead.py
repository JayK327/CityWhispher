"""
MODULE 3 — Lookahead Cache + Async Pipeline
This is the core latency optimization module.

Key concepts:
  1. Redis POI cache — check before calling external APIs
  2. asyncio.gather() — fetch Overpass + Wikipedia in parallel, not sequentially
  3. Latency profiling — time.perf_counter() around each stage, log a breakdown
  4. GPS grid rounding — round to 3 decimal places (~111m) to maximize cache hits

WHY THIS MATTERS:
  Sequential: Overpass (1200ms) + Wikipedia (800ms) = 2000ms
  Parallel:   max(Overpass, Wikipedia) = ~1200ms
  Cached:     Redis lookup = ~5ms
"""
import asyncio
import json
import time
import structlog

from app.models.poi import POIRecord
from app.services.module1_ingestion.overpass import fetch_overpass_pois
from app.services.module1_ingestion.wikipedia import fetch_wikipedia_summary
from app.services.module1_ingestion.normalizer import normalize_batch
from app.services.module3_cache.redis_client import get_redis
from app.config import settings

log = structlog.get_logger()

POI_CACHE_TTL = 600  # 10 minutes


def _cache_key(lat: float, lon: float) -> str:
    """
    Round coordinates to 3 decimal places (~111m grid) for cache key.
    Two locations within 111m share the same cache entry — acceptable for POIs.
    """
    return f"poi:{round(lat, 3)}:{round(lon, 3)}"


async def fetch_and_enrich_pois(
    lat: float,
    lon: float,
    radius_m: int = 500,
) -> tuple[list[POIRecord], dict]:
    """
    Fetch + enrich POIs for a location, with caching.

    Returns:
        (list of POIRecords sorted by confidence, latency_breakdown dict)

    Latency breakdown keys:
        cache_check_ms, overpass_ms, wikipedia_ms, normalize_ms, total_ms
    """
    timings = {}
    redis = await get_redis()

    # ── Step 1: Check cache ──────────────────────────────────────────────────
    t0 = time.perf_counter()
    cache_key = _cache_key(lat, lon)
    cached = await redis.get(cache_key)
    timings["cache_check_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    if cached:
        records = [POIRecord.model_validate(r) for r in json.loads(cached)]
        timings["source"] = "cache"
        log.info("poi_cache_hit", key=cache_key, count=len(records))
        return records, timings

    # ── Step 2: Parallel fetch from Overpass + (optionally) Wikipedia ────────
    t1 = time.perf_counter()
    osm_records = await fetch_overpass_pois(lat, lon, radius_m)
    timings["overpass_ms"] = round((time.perf_counter() - t1) * 1000, 1)

    if not osm_records:
        timings["source"] = "api_empty"
        return [], timings

    # Fetch Wikipedia summaries in parallel for all POI names
    t2 = time.perf_counter()
    wiki_results = await asyncio.gather(
        *[fetch_wikipedia_summary(r.name) for r in osm_records],
        return_exceptions=True,  # don't fail the whole batch if one Wikipedia lookup fails
    )
    timings["wikipedia_ms"] = round((time.perf_counter() - t2) * 1000, 1)

    # Build name → wiki summary map (skip exceptions)
    wiki_map = {}
    for osm_rec, wiki_result in zip(osm_records, wiki_results):
        if isinstance(wiki_result, Exception):
            wiki_map[osm_rec.name] = None
        else:
            wiki_map[osm_rec.name] = wiki_result

    # ── Step 3: Normalize + score ────────────────────────────────────────────
    t3 = time.perf_counter()
    records = normalize_batch(osm_records, wiki_map)
    timings["normalize_ms"] = round((time.perf_counter() - t3) * 1000, 1)

    # ── Step 4: Store in Redis cache ─────────────────────────────────────────
    if records:
        serialized = json.dumps([r.model_dump() for r in records])
        await redis.set(cache_key, serialized, ex=POI_CACHE_TTL)

    timings["source"] = "api"
    log.info(
        "poi_fetch_complete",
        lat=lat, lon=lon,
        count=len(records),
        timings=timings,
    )
    return records, timings


def select_best_poi(
    records: list[POIRecord],
    user_weights: list[float] | None = None,
) -> POIRecord | None:
    """
    Select the best POI to narrate based on:
      1. Confidence score (minimum threshold)
      2. User category preference weights (if available)

    Returns None if no record meets the confidence threshold.
    """
    from app.config import settings
    from app.models.user import CATEGORY_INDEX

    eligible = [r for r in records if r.confidence_score >= settings.confidence_threshold]
    if not eligible:
        return None

    if not user_weights:
        # No preferences — return highest confidence
        return eligible[0]

    # Score each eligible POI: confidence * user_category_weight
    def preference_score(poi: POIRecord) -> float:
        cat_idx = CATEGORY_INDEX.get(poi.category.value, -1)
        pref = user_weights[cat_idx] if cat_idx >= 0 else 0.1
        return poi.confidence_score * pref

    return max(eligible, key=preference_score)
