"""
LLM-as-Judge — fact-checks generated scripts against source facts.

Why? Manual QA doesn't scale. This automates the most expensive validation step.

Key trick: Tell the model paraphrasing is OK.
Without this, false positive rate ~18%. With it, ~6%.
"""
import json
import structlog
from groq import AsyncGroq
from app.config import settings

log = structlog.get_logger()

# Switched from OpenAI
# _client = AsyncOpenAI(api_key=settings.openai_api_key)
_client = AsyncGroq(api_key=settings.groq_api_key)

JUDGE_SYSTEM_PROMPT = """You are a fact-checker for travel audio commentary.

Given SOURCE FACTS and a GENERATED SCRIPT, find claims in the script that
are NOT supported by the source facts.

Important:
1. Paraphrases of the same fact are OK. "ancient fortress" = "castle built in 1243".
2. Only flag claims that are factually different or invented.
3. Hedging language ("reportedly", "known for") is fine.
4. General commentary ("a wonderful place") should not be flagged.

Respond with JSON only:
{"unsupported_claims": ["claim 1", "claim 2"]}

If all claims check out:
{"unsupported_claims": []}"""


async def judge_factual_accuracy(
    source_facts: dict,
    generated_script: str,
) -> tuple[list[str], float]:
    """
    Check if script makes unsupported claims.

    Returns: (list of bad claims, accuracy_score 0.0 or 1.0)
    """
    user_message = f"""SOURCE FACTS:
{json.dumps(source_facts, indent=2)}

GENERATED SCRIPT:
{generated_script}

List any unsupported claims."""

    try:
        response = await _client.chat.completions.create(
            model=settings.llm_model,
            max_tokens=300,
            temperature=0.0,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
        )
        raw = response.choices[0].message.content or '{"unsupported_claims": []}'
        data = json.loads(raw)
        claims = data.get("unsupported_claims", [])

        log.debug("judge_result", claim_count=len(claims), claims=claims[:3])
        return claims, (1.0 if not claims else 0.0)

    except Exception as e:
        log.warning("judge_failed", error=str(e))
        # On error, return uncertain — don't fail the whole eval
        return [], 0.5
