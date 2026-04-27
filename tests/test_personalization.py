"""
Unit tests for Module 4 — Personalization
No DB, no LLM, no async. Pure logic tests.

Run with: pytest tests/test_personalization.py -v
"""
import pytest
import numpy as np
from app.services.module4_personalization.preference import (
    update_preference_weights,
    cold_start_weights,
    SIGNALS,
    DECAY,
)
from app.models.user import CATEGORY_INDEX, DEFAULT_WEIGHTS


# ─── Weight Update Tests ──────────────────────────────────────────────────────

def test_skip_decreases_weight():
    weights = DEFAULT_WEIGHTS.copy()
    idx = CATEGORY_INDEX["historical"]
    original = weights[idx]
    updated = update_preference_weights(weights, "historical", "skip")
    assert updated[idx] < original, "Skip should decrease historical weight"


def test_replay_increases_weight():
    weights = DEFAULT_WEIGHTS.copy()
    idx = CATEGORY_INDEX["food"]
    original = weights[idx]
    updated = update_preference_weights(weights, "food", "replay")
    assert updated[idx] > original, "Replay should increase food weight"


def test_complete_increases_weight_slightly():
    weights = DEFAULT_WEIGHTS.copy()
    idx = CATEGORY_INDEX["nature"]
    original = weights[idx]
    updated = update_preference_weights(weights, "nature", "complete")
    assert updated[idx] > original
    # Complete signal is smaller than replay
    replay_updated = update_preference_weights(weights, "nature", "replay")
    assert updated[idx] < replay_updated[idx]


def test_weights_never_go_below_floor():
    """Even after many skips, weight should not go below 0.05."""
    weights = DEFAULT_WEIGHTS.copy()
    for _ in range(50):   # 50 consecutive skips
        weights = update_preference_weights(weights, "commercial", "skip")
    assert weights[CATEGORY_INDEX["commercial"]] >= 0.05


def test_weights_never_exceed_ceiling():
    """Even after many replays, weight should not exceed 1.0."""
    weights = DEFAULT_WEIGHTS.copy()
    for _ in range(50):
        weights = update_preference_weights(weights, "food", "replay")
    assert weights[CATEGORY_INDEX["food"]] <= 1.0


def test_exponential_decay_formula():
    """Verify the exact decay formula is applied."""
    weights = [0.5, 0.5, 0.5, 0.5, 0.5]
    category = "historical"
    idx = CATEGORY_INDEX[category]
    old_weight = weights[idx]
    delta = SIGNALS["skip"]

    updated = update_preference_weights(weights, category, "skip")
    expected = np.clip(old_weight * DECAY + delta, 0.05, 1.0)
    assert abs(updated[idx] - expected) < 1e-6


def test_unknown_category_returns_unchanged():
    """Unknown category should not modify any weights."""
    weights = DEFAULT_WEIGHTS.copy()
    updated = update_preference_weights(weights, "alien_architecture", "replay")
    assert updated == weights


# ─── Cold Start Tests ─────────────────────────────────────────────────────────

def test_default_cold_start_biases_historical():
    weights = cold_start_weights("default")
    hist_idx = CATEGORY_INDEX["historical"]
    # Historical should have the highest default weight
    assert weights[hist_idx] == max(weights)


def test_city_cold_start_biases_commercial():
    weights = cold_start_weights("city_center")
    comm_idx = CATEGORY_INDEX["commercial"]
    assert weights[comm_idx] >= 0.3


def test_cold_start_weights_sum_to_reasonable_range():
    """Weights should not sum to more than 5.0 (max = all 1.0)."""
    for route_type in ["default", "highway", "city_center", "coastal"]:
        weights = cold_start_weights(route_type)
        assert len(weights) == 5
        assert all(0.0 <= w <= 1.0 for w in weights)


def test_multiple_signals_converge():
    """
    After consistent signals, the dominant category should reflect
    the user's actual preference, not the cold start bias.
    """
    weights = cold_start_weights("default")   # starts biased toward historical

    # Simulate 10 consecutive food replays and historical skips
    for _ in range(10):
        weights = update_preference_weights(weights, "food", "replay")
        weights = update_preference_weights(weights, "historical", "skip")

    food_idx = CATEGORY_INDEX["food"]
    hist_idx = CATEGORY_INDEX["historical"]
    assert weights[food_idx] > weights[hist_idx], (
        "After 10 food replays and historical skips, food should dominate"
    )
