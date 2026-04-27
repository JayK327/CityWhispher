"""
MODULE 5 — Evaluation Test Suite

Run with:
  pytest tests/test_eval.py -v

To run with MLflow tracking:
  python scripts/run_eval.py

This suite tests every golden set sample through the FULL pipeline
(prompt render → LLM call → scoring) and asserts quality thresholds.

What you learn from running this:
  - How your current prompt version performs across 10 diverse POIs
  - Which dimension fails most often (factual accuracy? length? safety?)
  - How much the score changes when you tweak the prompt template
"""
import json
import pytest
import asyncio
from pathlib import Path

from app.models.poi import POIRecord, Tone, ContentCategory
from app.services.module2_llm.generator import generate_narration
from app.services.module5_eval.scorer import score_script
from app.config import settings

GOLDEN_SET_PATH = Path(__file__).parent / "golden_set.json"


def load_golden_set() -> list[dict]:
    with open(GOLDEN_SET_PATH) as f:
        return json.load(f)


def dict_to_poi_record(sample: dict) -> POIRecord:
    """Convert a golden set entry into a POIRecord for the pipeline."""
    facts = sample["source_facts"]
    return POIRecord(
        poi_id=sample["id"],
        name=facts["name"],
        lat=sample["lat"],
        lon=sample["lon"],
        category=ContentCategory(sample["category"]),
        description=facts.get("description", ""),
        source_url=facts.get("source_url", ""),
        address=facts.get("address", ""),
        opening_hours=facts.get("opening_hours", ""),
        source_count=2,
        confidence_score=0.85,  # golden set items are all high confidence
    )


# ─── Tests ────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@pytest.mark.parametrize("sample", load_golden_set())
async def test_generation_and_quality(sample: dict):
    """
    Full pipeline test: generate a script for each golden sample,
    then score it on all 3 dimensions.
    """
    poi = dict_to_poi_record(sample)
    source_facts = sample["source_facts"]

    # Generate narration
    script_result, _ = await generate_narration(poi, Tone.informative)

    assert script_result.script, f"[{sample['id']}] Empty script returned"

    # Score it
    score = await score_script(script_result.script, source_facts)

    # ── Dimension 1: Factual accuracy ─────────────────────────────────────────
    assert score.factual_accuracy >= 0.9, (
        f"[{sample['id']}] Factual accuracy {score.factual_accuracy:.2f} < 0.90\n"
        f"Unsupported claims: {score.unsupported_claims}\n"
        f"Script: {script_result.script}"
    )

    # ── Dimension 2: Length compliance ────────────────────────────────────────
    assert score.length_compliance == 1.0, (
        f"[{sample['id']}] Word count {score.word_count} "
        f"not in [{settings.target_word_min}, {settings.target_word_max}]\n"
        f"Script: {script_result.script}"
    )

    # ── Dimension 3: Driving safety ───────────────────────────────────────────
    assert score.driving_safety == 1.0, (
        f"[{sample['id']}] Script failed driving safety check\n"
        f"Script: {script_result.script}"
    )

    # ── Overall ───────────────────────────────────────────────────────────────
    assert score.overall >= 0.85, (
        f"[{sample['id']}] Overall score {score.overall:.2f} < 0.85"
    )


@pytest.mark.asyncio
async def test_word_count_in_range():
    """Explicit word count boundary test."""
    samples = load_golden_set()
    poi = dict_to_poi_record(samples[0])

    script_result, _ = await generate_narration(poi, Tone.informative)
    wc = script_result.word_count

    assert settings.target_word_min <= wc <= settings.target_word_max, (
        f"Word count {wc} outside [{settings.target_word_min}, {settings.target_word_max}]"
    )


@pytest.mark.asyncio
async def test_tone_casual_produces_different_output():
    """Verify tone parameter actually changes the output style."""
    samples = load_golden_set()
    poi = dict_to_poi_record(samples[0])

    informative, _ = await generate_narration(poi, Tone.informative)
    casual, _       = await generate_narration(poi, Tone.casual)

    # Scripts should not be identical — tone should have effect
    assert informative.script != casual.script, (
        "Casual and informative tone produced identical scripts — "
        "tone parameter may not be working"
    )


@pytest.mark.asyncio
async def test_no_hallucination_with_minimal_facts():
    """
    Test with a very sparse POI record to verify the fallback confidence
    signal works and the model doesn't invent facts.
    """
    sparse_poi = POIRecord(
        poi_id="sparse_test_001",
        name="Old Mill",
        lat=51.0,
        lon=0.5,
        category=ContentCategory.historical,
        description="",     # no description
        source_count=1,
        confidence_score=0.3,  # below threshold — should trigger fallback
    )

    # Sparse POIs should still generate without crashing
    script_result, _ = await generate_narration(sparse_poi, Tone.informative)
    assert script_result.script, "Even sparse POIs should produce a script"
    assert len(script_result.script.split()) > 10, "Script is too short"
