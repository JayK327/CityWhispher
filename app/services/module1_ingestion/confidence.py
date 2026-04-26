"""
Confidence scorer for POI data quality.

We assign a score 0–1 before feeding anything to the LLM. Works in 3 dimensions:
- Completeness: do we have name, category, description, etc?
- Source agreement: is this from OSM only, or also Wikipedia?
- Description quality: 50 words vs 5 words?

Thresholds:
- >= 0.75 → take it seriously, detailed narration
- >= 0.45 → maybe, add hedging language
- < 0.45 → skip this POI, use regional fallback instead

This prevents wasting LLM calls on garbage data.
"""

REQUIRED_FIELDS = ["name", "lat", "lon", "category"]
QUALITY_FIELDS = ["description", "address", "opening_hours", "source_url"]


def compute_confidence(
    name: str | None,
    description: str | None,
    address: str | None,
    opening_hours: str | None,
    source_url: str | None,
    source_count: int,
    category: str,
) -> float:
    """Quick data quality check. Returns 0.0–1.0."""
    score = 0.0

    # Required field completeness (0.5 max)
    present = sum([bool(name), True, True, category != "unknown"])
    score += (present / 4) * 0.5

    # Source bonus (0.3 max)
    # 2+ sources = we trust it more
    if source_count >= 2:
        score += 0.3
    elif source_count == 1:
        score += 0.1

    # Description helps (0.2 max)
    # but only if it's actually substantial
    if description:
        words = len(description.split())
        if words >= 50:
            score += 0.2
        elif words >= 20:
            score += 0.1
        elif words >= 5:
            score += 0.05

    return round(min(score, 1.0), 4)


def explain_confidence(score: float) -> str:
    """Human-readable translation of the score."""
    if score >= 0.75:
        return f"HIGH ({score:.2f}) — enough to write a proper script"
    elif score >= 0.45:
        return f"MEDIUM ({score:.2f}) — decent, but be vague"
    else:
        return f"LOW ({score:.2f}) — use generic regional commentary instead"
