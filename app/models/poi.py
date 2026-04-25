from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ─── Enums ───────────────────────────────────────────────────────────────────

class ContentCategory(str, Enum):
    historical = "historical"
    cultural   = "cultural"
    commercial = "commercial"
    nature     = "nature"
    food       = "food"
    unknown    = "unknown"


class ConfidenceLevel(str, Enum):
    high   = "high"    # score >= 0.75
    medium = "medium"  # score >= 0.45
    low    = "low"     # score <  0.45


class Tone(str, Enum):
    informative = "informative"
    casual      = "casual"
    family      = "family"


# ─── Source records (raw, before normalization) ───────────────────────────────

class OverpassRawPOI(BaseModel):
    osm_id:   int
    name:     Optional[str] = None
    lat:      float
    lon:      float
    tags:     dict = Field(default_factory=dict)


class WikipediaSummary(BaseModel):
    title:       str
    extract:     str
    page_url:    str
    thumbnail:   Optional[str] = None


# ─── Canonical POI record (after normalization) ───────────────────────────────

class POIRecord(BaseModel):
    """
    Unified canonical model for a Point of Interest.
    This is what gets stored in PostgreSQL and passed to the LLM.
    """
    poi_id:         str                   # f"osm_{osm_id}"
    name:           str
    lat:            float
    lon:            float
    category:       ContentCategory
    description:    Optional[str] = None  # from Wikipedia or OSM tags
    source_url:     Optional[str] = None
    address:        Optional[str] = None
    opening_hours:  Optional[str] = None
    source_count:   int = 1               # how many sources contributed
    confidence_score: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.low

    @model_validator(mode="after")
    def set_confidence_level(self) -> "POIRecord":
        if self.confidence_score >= 0.75:
            self.confidence_level = ConfidenceLevel.high
        elif self.confidence_score >= 0.45:
            self.confidence_level = ConfidenceLevel.medium
        else:
            self.confidence_level = ConfidenceLevel.low
        return self

    def to_prompt_context(self) -> dict:
        """Returns only the fields relevant for the LLM — keeps token count low."""
        return {
            "name":          self.name,
            "category":      self.category.value,
            "description":   self.description or "",
            "address":       self.address or "",
            "opening_hours": self.opening_hours or "",
            "source_url":    self.source_url or "",
        }


# ─── LLM output ──────────────────────────────────────────────────────────────

class NarrationScript(BaseModel):
    """Structured output enforced on the LLM response."""
    script:      str
    word_count:  int
    confidence:  ConfidenceLevel

    @model_validator(mode="after")
    def recount_words(self) -> "NarrationScript":
        # Always recount — don't trust the model's self-report
        self.word_count = len(self.script.split())
        return self


# ─── API request / response ───────────────────────────────────────────────────

class NarrationRequest(BaseModel):
    lat:     float = Field(..., ge=-90,  le=90)
    lon:     float = Field(..., ge=-180, le=180)
    user_id: Optional[str] = None
    tone:    Tone = Tone.informative


class NarrationResponse(BaseModel):
    poi_name:       str
    category:       str
    script:         str
    word_count:     int
    confidence:     str
    audio_url:      Optional[str] = None
    latency_ms:     dict           = Field(default_factory=dict)
    fallback_used:  bool = False


class SignalRequest(BaseModel):
    user_id:  str
    poi_id:   str
    category: ContentCategory
    action:   str = Field(..., pattern="^(skip|replay|complete)$")
