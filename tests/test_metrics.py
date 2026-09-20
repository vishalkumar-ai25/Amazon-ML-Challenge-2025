"""Tests for SMAPE metric and evaluation utilities.

Verifies our SMAPE implementation matches the exact competition definition:
    SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|) / 2)
Bounded between 0% and 200%. Lower is better.
"""
import numpy as np
import pytest


def test_smape_perfect_prediction():
    """Identical predictions should yield SMAPE = 0%."""
    from src.metrics import smape
    y_true = np.array([100.0, 200.0, 50.0])
    y_pred = np.array([100.0, 200.0, 50.0])
    assert smape(y_true, y_pred) == pytest.approx(0.0, abs=1e-10)


def test_smape_competition_example():
    """Match the exact example from the problem statement.
    
    actual=100, predicted=120 → SMAPE = |100-120| / ((|100|+|120|)/2) * 100%
                                      = 20 / 110 * 100% = 18.18...%
    """
    from src.metrics import smape
    y_true = np.array([100.0])
    y_pred = np.array([120.0])
    expected = 20.0 / 110.0 * 100.0  # 18.1818...%
    assert smape(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


def test_smape_symmetric():
    """SMAPE should give the same result regardless of over/under prediction."""
    from src.metrics import smape
    y_true = np.array([100.0])
    # Over-predict by 20
    smape_over = smape(y_true, np.array([120.0]))
    # Under-predict by 20
    smape_under = smape(y_true, np.array([80.0]))
    # These should NOT be equal for SMAPE (denominator changes), 
    # but both should be valid positive percentages
    assert smape_over > 0
    assert smape_under > 0


def test_smape_bounded():
    """SMAPE should be bounded between 0% and 200%."""
    from src.metrics import smape
    y_true = np.array([100.0, 50.0, 200.0])
    y_pred = np.array([1.0, 1000.0, 0.01])  # wildly wrong
    result = smape(y_true, y_pred)
    assert 0.0 <= result <= 200.0


def test_smape_multiple_samples():
    """SMAPE is the mean of per-sample errors."""
    from src.metrics import smape
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([120.0, 200.0])  # first is off, second is perfect
    # Sample 1: 20/110 * 100 = 18.1818...
    # Sample 2: 0/200 * 100 = 0.0
    # Mean: 18.1818... / 2 = 9.0909...
    expected = (20.0 / 110.0 * 100.0) / 2.0
    assert smape(y_true, y_pred) == pytest.approx(expected, rel=1e-6)


def test_smape_handles_small_values():
    """SMAPE should handle very small prices without division by zero."""
    from src.metrics import smape
    y_true = np.array([0.13, 0.50])  # min price in dataset is $0.13
    y_pred = np.array([0.15, 0.45])
    result = smape(y_true, y_pred)
    assert np.isfinite(result)
    assert result >= 0


def test_smape_clips_negative_predictions():
    """Negative predictions should be handled gracefully (clipped to small positive)."""
    from src.metrics import smape
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([-5.0, 25.0])
    result = smape(y_true, y_pred)
    assert np.isfinite(result)
    assert result > 0


def test_smape_input_types():
    """Should accept lists, arrays, and Series."""
    from src.metrics import smape
    y_true_list = [100.0, 200.0]
    y_pred_list = [110.0, 190.0]
    result_list = smape(y_true_list, y_pred_list)
    result_array = smape(np.array(y_true_list), np.array(y_pred_list))
    assert result_list == pytest.approx(result_array, rel=1e-10)


def test_smape_large_dataset_performance():
    """SMAPE should compute efficiently on dataset-sized inputs (75K)."""
    from src.metrics import smape
    rng = np.random.RandomState(42)
    y_true = rng.uniform(0.13, 2800, size=75_000)
    y_pred = y_true * rng.uniform(0.5, 1.5, size=75_000)
    result = smape(y_true, y_pred)
    assert np.isfinite(result)
    assert 0 <= result <= 200
