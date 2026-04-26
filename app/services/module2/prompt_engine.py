"""
Prompt templating. Uses Jinja2 so non-coders can tweak prompts without touching Python.
"""
from pathlib import Path
import jinja2
import tiktoken
import structlog

from app.models.poi import POIRecord, Tone

log = structlog.get_logger()

PROMPTS_DIR = Path(__file__).parent.parent.parent.parent / "prompts"

_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(PROMPTS_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

_tokenizer = tiktoken.get_encoding("cl100k_base")


def render_narrator_prompt(
    poi: POIRecord,
    tone: Tone = Tone.informative,
    word_min: int = 55,
    word_max: int = 80,
) -> tuple[str, int]:
    """Render narrator.j2 template. Returns (prompt, token_count)."""
    template = _jinja_env.get_template("narrator.j2")

    rendered = template.render(
        poi_facts=poi.to_prompt_context(),
        tone=tone.value,
        word_min=word_min,
        word_max=word_max,
    )

    token_count = len(_tokenizer.encode(rendered))

    log.debug(
        "prompt_rendered",
        poi_id=poi.poi_id,
        tone=tone.value,
        tokens=token_count,
    )

    return rendered, token_count


def render_regional_fallback_prompt(
    lat: float,
    lon: float,
    region_name: str = "this area",
    tone: Tone = Tone.informative,
) -> str:
    """Generate fallback prompt when no POI data. Keep user from hearing silence."""
    # Old version (commented out):
    # return f"""You are an in-car travel narrator.
    # The listener is passing through {region_name} (coordinates: {lat:.3f}, {lon:.3f}).
    # No specific point of interest data is available for this exact location.
    #
    # Generate a warm, general 50-60 word commentary about traveling through this type of area.
    # Mention the act of discovery, the journey itself, and invite curiosity.
    # Do NOT make up specific facts about {region_name}.
    # Tone: {tone.value}
    #
    # Respond with JSON only: {{"script": "...", "word_count": N, "confidence": "low"}}"""

    # New version with stricter word count requirement:
    return f"""You are an in-car travel narrator.
The listener is passing through {region_name} (coordinates: {lat:.3f}, {lon:.3f}).
No specific point of interest data is available for this exact location.

Generate a warm, detailed 50-60 word commentary about traveling through this type of area.
IMPORTANT: Your script MUST be between 50-60 words — write enough content to fill this length.
Mention the act of discovery, the journey itself, and invite curiosity.
Do NOT make up specific facts about {region_name}.
Tone: {tone.value}

Respond with JSON only: {{"script": "...", "word_count": N, "confidence": "low"}}"""
