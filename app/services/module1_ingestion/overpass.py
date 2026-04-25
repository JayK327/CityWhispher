"""
Fetches POIs from OpenStreetMap via the free Overpass API.
"""
import httpx
import structlog
from app.models.poi import OverpassRawPOI

log = structlog.get_logger()

OVERPASS_URL = "https://overpass-api.de/api/interpreter"

# Tags we care about
TAG_CATEGORY_MAP = {
    "historic": "historical",
    "tourism": "cultural",
    "amenity": "commercial",
    "natural": "nature",
    "leisure": "nature",
    "shop": "commercial",
    "cuisine": "food",
    "restaurant": "food",
    "cafe": "food",
    "museum": "cultural",
    "monument": "historical",
    "ruins": "historical",
    "viewpoint": "nature",
    "park": "nature",
}


def build_overpass_query(lat: float, lon: float, radius_m: int = 500) -> str:
    """Build Overpass query for POIs within radius_m meters."""
    return f"""
    [out:json][timeout:15];
    (
      node["tourism"](around:{radius_m},{lat},{lon});
      node["historic"](around:{radius_m},{lat},{lon});
      node["amenity"~"restaurant|cafe|museum|theatre|library"](around:{radius_m},{lat},{lon});
      node["leisure"~"park|nature_reserve"](around:{radius_m},{lat},{lon});
      node["natural"~"peak|waterfall|spring"](around:{radius_m},{lat},{lon});
    );
    out body;
    """


async def fetch_overpass_pois(
    lat: float,
    lon: float,
    radius_m: int = 500,
) -> list[OverpassRawPOI]:
    """Fetch POIs from Overpass. Returns empty list on error — never raises."""
    query = build_overpass_query(lat, lon, radius_m)
    results: list[OverpassRawPOI] = []

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                OVERPASS_URL,
                data={"data": query},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            resp.raise_for_status()
            data = resp.json()

        for element in data.get("elements", []):
            tags = element.get("tags", {})
            name = tags.get("name") or tags.get("name:en")
            if not name:
                continue
            results.append(
                OverpassRawPOI(
                    osm_id=element["id"],
                    name=name,
                    lat=element["lat"],
                    lon=element["lon"],
                    tags=tags,
                )
            )

        log.info("overpass_fetch_done", count=len(results), lat=lat, lon=lon)
        return results

    except Exception as e:
        log.warning("overpass_fetch_failed", error=str(e), lat=lat, lon=lon)
        return []


def infer_category_from_tags(tags: dict) -> str:
    """Map OSM tags to categories. Returns 'unknown' if we can't figure it out."""
    for key in ["historic", "tourism", "leisure", "natural", "amenity", "shop"]:
        val = tags.get(key, "")
        if not val:
            continue
        mapped = TAG_CATEGORY_MAP.get(val) or TAG_CATEGORY_MAP.get(key)
        if mapped:
            return mapped
    return "unknown"
