"""
MODULE 4 — Personalization: Preference Vector
Per-user category weight vector with exponential decay.

Data model:
  weights: list[float] of length 5
  indices: historical=0, cultural=1, commercial=2, nature=3, food=4

Update rule (exponential decay):
  new_weight = old_weight * DECAY + signal_delta
  Clipped to [0.05, 1.0] — floor prevents a category from being silenced forever

Why exponential decay (not simple average)?
  - Recent behavior should matter more than old behavior
  - A user who loved history last month but skips it now should shift quickly
  - The 0.9 decay means a signal from 10 rides ago contributes only 0.9^10 ≈ 35%

Cold start:
  - New user → default weights [0.4, 0.3, 0.1, 0.1, 0.1]
  - Biased toward historical + cultural (broadest appeal for most travelers)
  - After 3+ signals, the vector personalizes to real preferences
"""
import numpy as np
import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.schemas import UserPreferenceTable
from app.models.user import UserPreferences, CATEGORY_INDEX, DEFAULT_WEIGHTS

log = structlog.get_logger()

# ── Signal strengths ──────────────────────────────────────────────────────────
DECAY = 0.9
SIGNALS = {
    "skip":     -0.15,   # strong negative — don't want this category
    "complete":  0.05,   # weak positive — listened to the end
    "replay":    0.25,   # strong positive — actively wanted more
}

# ── Route type → default weight presets (cold start context-aware defaults) ───
ROUTE_DEFAULTS = {
    "highway":      [0.5, 0.2, 0.05, 0.2, 0.05],  # historical + nature
    "city_center":  [0.2, 0.2, 0.3,  0.1, 0.2],   # commercial + food
    "coastal":      [0.2, 0.2, 0.1,  0.4, 0.1],   # nature + cultural
    "default":      [0.4, 0.3, 0.1,  0.1, 0.1],   # general default
}


def update_preference_weights(
    weights: list[float],
    category: str,
    action: str,
) -> list[float]:
    """
    Apply one signal to the preference weight vector.

    Args:
        weights:  Current weight vector (length 5)
        category: Content category that was narrated (e.g. "historical")
        action:   User signal: "skip" | "complete" | "replay"

    Returns:
        Updated weight vector (new list, original unchanged)
    """
    vec = np.array(weights, dtype=float)
    idx = CATEGORY_INDEX.get(category, -1)

    if idx == -1:
        log.warning("unknown_category_signal", category=category)
        return weights  # unknown category — no update

    delta = SIGNALS.get(action, 0.0)
    vec[idx] = vec[idx] * DECAY + delta

    # Clip to [0.05, 1.0] — floor prevents zeroing out a category permanently
    vec = np.clip(vec, 0.05, 1.0)

    log.debug(
        "preference_updated",
        category=category,
        action=action,
        old=round(weights[idx], 3),
        new=round(float(vec[idx]), 3),
    )
    return vec.tolist()


def cold_start_weights(route_type: str = "default") -> list[float]:
    """Return default weights for a new user based on route context."""
    return ROUTE_DEFAULTS.get(route_type, ROUTE_DEFAULTS["default"]).copy()


# ── Database helpers ──────────────────────────────────────────────────────────

async def get_or_create_preferences(
    user_id: str,
    db: AsyncSession,
    route_type: str = "default",
) -> UserPreferences:
    """
    Fetch user preferences from DB. Creates a row with cold-start defaults
    if the user has never been seen before.
    """
    result = await db.execute(
        select(UserPreferenceTable).where(UserPreferenceTable.user_id == user_id)
    )
    row = result.scalar_one_or_none()

    if row is None:
        # New user — insert cold start defaults
        weights = cold_start_weights(route_type)
        row = UserPreferenceTable(user_id=user_id, weights=weights)
        db.add(row)
        await db.commit()
        log.info("new_user_preferences_created", user_id=user_id, route_type=route_type)

    return UserPreferences(
        user_id=row.user_id,
        weights=row.weights,
        tone=row.tone,
        updated_at=row.updated_at,
    )


async def apply_signal(
    user_id: str,
    category: str,
    action: str,
    db: AsyncSession,
) -> UserPreferences:
    """
    Apply a user signal to their stored preference vector.
    This is called by the POST /signal endpoint.
    """
    result = await db.execute(
        select(UserPreferenceTable).where(UserPreferenceTable.user_id == user_id)
    )
    row = result.scalar_one_or_none()

    if row is None:
        # User doesn't exist yet — create with defaults
        prefs = await get_or_create_preferences(user_id, db)
        result = await db.execute(
            select(UserPreferenceTable).where(UserPreferenceTable.user_id == user_id)
        )
        row = result.scalar_one()

    new_weights = update_preference_weights(row.weights, category, action)
    row.weights = new_weights
    await db.commit()
    await db.refresh(row)

    return UserPreferences(
        user_id=row.user_id,
        weights=row.weights,
        tone=row.tone,
    )
