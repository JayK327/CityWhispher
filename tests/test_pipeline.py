"""
Unit tests for Module 1 + Module 3 — no LLM calls needed.

Run with: pytest tests/test_pipeline.py -v
These tests are fast and don't require an OpenAI key.
"""
import pytest
from app.services.module1_ingestion.confidence import compute_confidence
from app.services.module1_ingestion.normalizer import normalize_overpass_poi
from app.services.module3_cache.lookahead import select_best_poi, _cache_key
from app.models.poi import OverpassRawPOI, POIRecord, ContentCategory


# ─── Confidence Scorer Tests ──────────────────────────────────────────────────

def test_confidence_high_with_all_fields():
    score = compute_confidence(
        name="Eiffel Tower",
        description="A wrought iron lattice tower built by Gustave Eiffel in 1889 for the World's Fair in Paris.",
        address="Champ de Mars, Paris",
        opening_hours="09:00-23:45",
        source_url="https://en.wikipedia.org/wiki/Eiffel_Tower",
        source_count=2,
        category="historical",
    )
    assert score >= 0.75, f"Expected high confidence, got {score}"


def test_confidence_low_with_minimal_fields():
    score = compute_confidence(
        name="Unknown Cafe",
        description=None,
        address=None,
        opening_hours=None,
        source_url=None,
        source_count=1,
        category="unknown",
    )
    assert score < 0.45, f"Expected low confidence, got {score}"


def test_confidence_medium_single_source_with_description():
    score = compute_confidence(
        name="Old Bridge",
        description="A historic stone bridge dating back several centuries.",
        address=None,
        opening_hours=None,
        source_url=None,
        source_count=1,
        category="historical",
    )
    assert 0.45 <= score < 0.75, f"Expected medium confidence, got {score}"


def test_confidence_bonus_for_two_sources():
    score_one = compute_confidence("Place", None, None, None, None, 1, "cultural")
    score_two = compute_confidence("Place", None, None, None, None, 2, "cultural")
    assert score_two > score_one, "Two sources should score higher than one"


# ─── Normalizer Tests ─────────────────────────────────────────────────────────

def test_normalize_osm_only():
    raw = OverpassRawPOI(
        osm_id=123456,
        name="Test Museum",
        lat=51.5,
        lon=-0.1,
        tags={"tourism": "museum", "opening_hours": "10:00-18:00"},
    )
    record = normalize_overpass_poi(raw, wiki=None)
    assert record.poi_id == "osm_123456"
    assert record.name == "Test Museum"
    assert record.category.value == "cultural"
    assert record.source_count == 1
    assert record.opening_hours == "10:00-18:00"


def test_normalize_with_wikipedia_boosts_confidence():
    raw = OverpassRawPOI(
        osm_id=999,
        name="Famous Castle",
        lat=48.0,
        lon=2.0,
        tags={"historic": "castle"},
    )
    from app.models.poi import WikipediaSummary
    wiki = WikipediaSummary(
        title="Famous Castle",
        extract="A medieval castle built in the 12th century by local lords. It served as a fortress and administrative center for the region.",
        page_url="https://en.wikipedia.org/wiki/Famous_Castle",
    )

    record_no_wiki   = normalize_overpass_poi(raw, wiki=None)
    record_with_wiki = normalize_overpass_poi(raw, wiki=wiki)

    assert record_with_wiki.confidence_score > record_no_wiki.confidence_score
    assert record_with_wiki.source_count == 2
    assert record_with_wiki.description == wiki.extract


# ─── Cache Key Tests ──────────────────────────────────────────────────────────

def test_cache_key_rounding():
    """Locations within ~111m should share a cache key."""
    key1 = _cache_key(51.5001, -0.1001)
    key2 = _cache_key(51.5002, -0.1002)
    assert key1 == key2, "Nearby locations should share a cache key"


def test_cache_key_different_locations():
    """Locations far apart should have different cache keys."""
    key1 = _cache_key(51.500, -0.100)
    key2 = _cache_key(48.858, 2.294)
    assert key1 != key2


# ─── POI Selection Tests ──────────────────────────────────────────────────────

def _make_poi(poi_id: str, category: str, confidence: float) -> POIRecord:
    return POIRecord(
        poi_id=poi_id, name=f"POI {poi_id}",
        lat=51.5, lon=-0.1,
        category=ContentCategory(category),
        confidence_score=confidence,
    )


def test_select_best_poi_by_confidence():
    records = [
        _make_poi("a", "historical", 0.8),
        _make_poi("b", "cultural",   0.6),
        _make_poi("c", "food",       0.3),   # below threshold
    ]
    best = select_best_poi(records)
    assert best is not None
    assert best.poi_id == "a"


def test_select_best_poi_with_user_preference():
    """User who prefers food should get the food POI even if it's lower confidence."""
    records = [
        _make_poi("hist", "historical", 0.8),
        _make_poi("food", "food",       0.55),
    ]
    # User strongly prefers food: [historical=0.1, cultural=0.1, commercial=0.1, nature=0.1, food=0.9]
    food_lover_weights = [0.1, 0.1, 0.1, 0.1, 0.9]
    best = select_best_poi(records, user_weights=food_lover_weights)
    assert best is not None
    assert best.poi_id == "food"


def test_select_best_poi_none_above_threshold():
    records = [
        _make_poi("a", "historical", 0.2),
        _make_poi("b", "cultural",   0.3),
    ]
    best = select_best_poi(records)
    assert best is None
