"""
LLM call handler. Send prompt to Groq, parse JSON response, trim to word count.
"""
import json
import re
import structlog
from groq import AsyncGroq
import nltk

from app.config import settings
from app.models.poi import NarrationScript, POIRecord, Tone, ConfidenceLevel
from app.services.module2_llm.prompt_engine import (
    render_narrator_prompt,
    render_regional_fallback_prompt,
)

log = structlog.get_logger()

# Grab NLTK tokenizer for sentence splitting
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

# Groq client (switched from OpenAI)
# _client = AsyncOpenAI(api_key=settings.openai_api_key)
_client = AsyncGroq(api_key=settings.groq_api_key)


def trim_to_word_limit(text: str, max_words: int = 80) -> str:
    """
    Chop text at sentence boundaries to respect word count.

    Why not just truncate at word N?
    Because mid-sentence TTS audio sounds terrible.
    """
    sentences = nltk.sent_tokenize(text)
    result = []
    total = 0
    for sent in sentences:
        words = len(sent.split())
        if total + words > max_words and result:
            break
        result.append(sent)
        total += words
    return " ".join(result)


def clean_json_response(raw: str) -> str:
    """LLMs sometimes wrap JSON in markdown. Strip that."""
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


async def generate_narration(
    poi: POIRecord,
    tone: Tone = Tone.informative,
) -> tuple[NarrationScript, int]:
    """
    Generate narration for a POI.

    Returns: (NarrationScript, prompt_token_count)
    """
    prompt, token_count = render_narrator_prompt(
        poi=poi,
        tone=tone,
        word_min=settings.target_word_min,
        word_max=settings.target_word_max,
    )

    response = await _client.chat.completions.create(
        model=settings.llm_model,
        max_tokens=settings.llm_max_tokens,
        temperature=0.7,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Generate the travel commentary now."},
        ],
    )

    raw_content = response.choices[0].message.content or "{}"
    cleaned = clean_json_response(raw_content)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        log.error("llm_json_parse_failed", error=str(e), raw=raw_content[:200])
        raise ValueError(f"LLM returned invalid JSON: {e}")

    script = data.get("script", "").strip()

    # Trim if over limit
    if len(script.split()) > settings.target_word_max:
        script = trim_to_word_limit(script, settings.target_word_max)

    narration = NarrationScript(
        script=script,
        word_count=len(script.split()),
        confidence=data.get("confidence", "medium"),
    )

    log.info(
        "narration_generated",
        poi_id=poi.poi_id,
        word_count=narration.word_count,
        confidence=narration.confidence,
        prompt_tokens=token_count,
    )

    return narration, token_count


async def generate_regional_fallback(
    lat: float,
    lon: float,
    tone: Tone = Tone.informative,
) -> NarrationScript:
    """Generate generic commentary when there's no good POI data."""
    prompt = render_regional_fallback_prompt(lat, lon, tone=tone)

    try:
        response = await _client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=200,
            temperature=0.8,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": "Generate the regional commentary."},
            ],
        )
        raw = clean_json_response(response.choices[0].message.content or "{}")
        data = json.loads(raw)
        script = data.get("script", "Enjoy the journey ahead.")
    except Exception as e:
        log.warning("fallback_generation_failed", error=str(e))
        script = "Enjoy the journey — every road has its own story waiting to be discovered."

    return NarrationScript(
        script=script,
        word_count=len(script.split()),
        confidence=ConfidenceLevel.low,
    )
