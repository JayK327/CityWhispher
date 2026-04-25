"""
MODULE 1 — Wikipedia REST API fetcher
Enriches POIs with factual descriptions from Wikipedia.

Key concepts:
  - Wikipedia's free REST API (no key needed)
  - Graceful degradation — if Wikipedia has no article, we continue without it
  - This is the "second source" that boosts confidence score
"""
import httpx
import structlog
from app.models.poi import WikipediaSummary

log = structlog.get_logger()

WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary"


async def fetch_wikipedia_summary(name: str) -> WikipediaSummary | None:
    """
    Fetch the Wikipedia page summary for a given POI name.
    Returns None if no article found — caller handles the fallback.
    """
    # Clean up the name for URL encoding
    search_title = name.replace(" ", "_")

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                f"{WIKI_API}/{search_title}",
                headers={"User-Agent": "CityWhisper/1.0 (educational project)"},
                follow_redirects=True,
            )

            if resp.status_code == 404:
                # Try a slightly different format — remove "The" prefix etc.
                log.debug("wikipedia_not_found", name=name)
                return None

            resp.raise_for_status()
            data = resp.json()

            # Wikipedia returns disambiguation pages — skip those
            if data.get("type") == "disambiguation":
                return None

            extract = data.get("extract", "").strip()
            if not extract or len(extract) < 50:
                return None

            return WikipediaSummary(
                title=data.get("title", name),
                extract=extract[:800],  # cap at 800 chars to limit prompt tokens
                page_url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                thumbnail=data.get("thumbnail", {}).get("source"),
            )

    except httpx.HTTPStatusError as e:
        log.debug("wikipedia_http_error", status=e.response.status_code, name=name)
        return None
    except Exception as e:
        log.warning("wikipedia_fetch_failed", error=str(e), name=name)
        return None


async def search_wikipedia(query: str) -> WikipediaSummary | None:
    """
    Fallback: Use Wikipedia's search API when direct title lookup fails.
    Tries to find the best matching article for a POI name.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                "https://en.wikipedia.org/api/rest_v1/page/related",
                params={"q": query, "limit": 1},
                headers={"User-Agent": "CityWhisper/1.0"},
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            pages = data.get("pages", [])
            if pages:
                return await fetch_wikipedia_summary(pages[0]["title"])
    except Exception:
        pass
    return None
