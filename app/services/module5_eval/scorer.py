"""
MODULE 5 — 3-Dimension Scorer
Evaluates every generated script on three dimensions:

  Dimension 1 — Factual accuracy   (0.0 or 1.0)
    Uses LLM-as-judge to detect hallucinated claims.
    Weight: 0.5 (most important)

  Dimension 2 — Length compliance  (0.0 or 1.0)
    Is word count within the 55–80 target range?
    Weight: 0.3

  Dimension 3 — Driving safety     (0.0 or 1.0)
    Does the script avoid long sentences (>25 words)?
    Does it avoid complex numbers, directions?
    Weight: 0.2

Final score = weighted average of all three dimensions.
Threshold for CI pass: >= 0.85 overall, >= 0.90 on factual accuracy.
"""
import re
import structlog
from dataclasses import dataclass
import nltk

from app.config import settings
from app.services.module5_eval.judge import judge_factual_accuracy

log = structlog.get_logger()

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

# Scoring weights
WEIGHTS = {
    "factual_accuracy":  0.5,
    "length_compliance": 0.3,
    "driving_safety":    0.2,
}

# Patterns that are unsafe while driving
UNSAFE_PATTERNS = [
    r"\b\d{4,}\b",              # large numbers (year is ok, but 12,345 is not)
    r"\bturn (left|right)\b",   # directions
    r"\bfollowing \w+ steps\b", # enumerated instructions
]


@dataclass
class ScoreResult:
    factual_accuracy:  float
    length_compliance: float
    driving_safety:    float
    overall:           float
    word_count:        int
    unsupported_claims: list[str]
    passed:            bool

    def to_dict(self) -> dict:
        return {
            "factual_accuracy":   self.factual_accuracy,
            "length_compliance":  self.length_compliance,
            "driving_safety":     self.driving_safety,
            "overall":            self.overall,
            "word_count":         self.word_count,
            "unsupported_claims": self.unsupported_claims,
            "passed":             self.passed,
        }


def score_length_compliance(script: str) -> float:
    """Check word count is within target range."""
    wc = len(script.split())
    return 1.0 if settings.target_word_min <= wc <= settings.target_word_max else 0.0


def score_driving_safety(script: str) -> float:
    """
    Check the script is safe to hear while driving:
    - No sentences over 25 words
    - No complex numbers or turn-by-turn directions
    """
    sentences = nltk.sent_tokenize(script)

    # Check sentence length
    long_sentences = [s for s in sentences if len(s.split()) > 25]
    if long_sentences:
        log.debug("long_sentences_found", count=len(long_sentences))
        return 0.0

    # Check for unsafe patterns
    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, script, re.IGNORECASE):
            log.debug("unsafe_pattern_found", pattern=pattern)
            return 0.0

    return 1.0


async def score_script(
    script: str,
    source_facts: dict,
) -> ScoreResult:
    """
    Score a generated script on all 3 dimensions.

    Args:
        script:       The generated narration text
        source_facts: The POI facts dict that was passed to the LLM

    Returns:
        ScoreResult with individual and overall scores
    """
    # Dimension 1: Factual accuracy (async — calls LLM judge)
    unsupported, factual_score = await judge_factual_accuracy(source_facts, script)

    # Dimension 2: Length compliance (sync)
    length_score = score_length_compliance(script)

    # Dimension 3: Driving safety (sync)
    safety_score = score_driving_safety(script)

    # Weighted average
    overall = (
        factual_score  * WEIGHTS["factual_accuracy"]  +
        length_score   * WEIGHTS["length_compliance"] +
        safety_score   * WEIGHTS["driving_safety"]
    )

    # Pass criteria: overall >= 0.85 AND factual accuracy >= 0.9
    passed = overall >= 0.85 and factual_score >= 0.9

    result = ScoreResult(
        factual_accuracy=factual_score,
        length_compliance=length_score,
        driving_safety=safety_score,
        overall=round(overall, 4),
        word_count=len(script.split()),
        unsupported_claims=unsupported,
        passed=passed,
    )

    log.info("score_result", **result.to_dict())
    return result
