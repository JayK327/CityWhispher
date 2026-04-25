"""
POI normalizer — takes raw OSM + Wikipedia and merges into a single shape.

This is our ETL transform step. Two sources, different schemas → one canonical model.
"""
import structlog
from app.models.poi import OverpassRawPOI, WikipediaSummary, POIRecord
from app.services.module1_ingestion.overpass import infer_category_from_tags
from app.services.module1_ingestion.confidence import compute_confidence

log = structlog.get_logger()


def normalize_overpass_poi(
    raw: OverpassRawPOI,
    wiki: WikipediaSummary | None = None,
) -> POIRecord:
    """
    Merge OSM raw record + optional Wikipedia into a POIRecord.

    Strategy:
    - name: from OSM (already filtered)
    - description: Wikipedia (preferred) → OSM tags
    - category: inferred from OSM
    - source_count: 1 (OSM only) or 2 (OSM + wiki)
    - confidence: computed from all above
    """
    tags = raw.tags

    description = None
    source_url = None

    if wiki:
        description = wiki.extract
        source_url = wiki.page_url

    if not description:
        description = tags.get("description") or tags.get("note")

    category_str = infer_category_from_tags(tags)
    source_count = 2 if wiki else 1

    # Try to piece together an address from scattered OSM tags
    address_parts = [
        tags.get("addr:housenumber", ""),
        tags.get("addr:street", ""),
        tags.get("addr:city", ""),
    ]
    address = " ".join(p for p in address_parts if p).strip() or None

    confidence = compute_confidence(
        name=raw.name,
        description=description,
        address=address,
        opening_hours=tags.get("opening_hours"),
        source_url=source_url,
        source_count=source_count,
        category=category_str,
    )

    record = POIRecord(
        poi_id=f"osm_{raw.osm_id}",
        name=raw.name,
        lat=raw.lat,
        lon=raw.lon,
        category=category_str,
        description=description,
        source_url=source_url,
        address=address,
        opening_hours=tags.get("opening_hours"),
        source_count=source_count,
        confidence_score=confidence,
    )

    log.debug(
        "poi_normalized",
        poi_id=record.poi_id,
        name=record.name,
        confidence=confidence,
        has_wiki=wiki is not None,
    )
    return record


def normalize_batch(
    osm_records: list[OverpassRawPOI],
    wiki_map: dict[str, WikipediaSummary | None],
) -> list[POIRecord]:
    """Normalize batch of OSM records + wiki lookups. Sort by confidence."""
    normalized = []
    for raw in osm_records:
        wiki = wiki_map.get(raw.name)
        record = normalize_overpass_poi(raw, wiki)
        normalized.append(record)

    normalized.sort(key=lambda r: r.confidence_score, reverse=True)
    return normalized
