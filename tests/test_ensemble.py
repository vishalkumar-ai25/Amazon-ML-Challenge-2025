"""Tests for ensemble and blending optimization module.

Verifies:
- Finding optimal blend weights via Nelder-Mead / SLSQP minimizing SMAPE
- Weights normalization (sum to 1, non-negative)
- Blended ensemble outperforms or matches individual models
- Output predictions stay positive
"""
import numpy as np
import pytest
from src.metrics import smape


def test_find_optimal_blend_weights_single_model():
    """Single model should receive 100% weight."""
    from src.ensemble import find_optimal_blend_weights
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    oof_preds = {"model_a": np.array([11.0, 19.0, 32.0, 38.0])}

    weights = find_optimal_blend_weights(oof_preds, y_true)
    assert "model_a" in weights
    assert weights["model_a"] == pytest.approx(1.0, abs=1e-3)


def test_find_optimal_blend_weights_two_models():
    """Ensemble of complementary models should achieve better or equal SMAPE."""
    from src.ensemble import find_optimal_blend_weights, apply_blend
    rng = np.random.RandomState(42)
    y_true = rng.uniform(10.0, 100.0, size=200)

    # Model A overpredicts, Model B underpredicts
    pred_a = y_true * 1.15 + rng.normal(0, 2, size=200)
    pred_b = y_true * 0.85 + rng.normal(0, 2, size=200)
    pred_a = np.maximum(pred_a, 1.0)
    pred_b = np.maximum(pred_b, 1.0)

    oof_preds = {"model_a": pred_a, "model_b": pred_b}

    smape_a = smape(y_true, pred_a)
    smape_b = smape(y_true, pred_b)

    weights = find_optimal_blend_weights(oof_preds, y_true)

    assert len(weights) == 2
    assert sum(weights.values()) == pytest.approx(1.0, abs=1e-3)
    assert all(w >= 0 for w in weights.values())

    blended = apply_blend(oof_preds, weights)
    blended_smape = smape(y_true, blended)

    # Blended SMAPE should strictly beat individual models in this complementary scenario
    assert blended_smape < min(smape_a, smape_b)
    assert (blended > 0).all()
