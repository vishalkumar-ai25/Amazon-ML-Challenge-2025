"""Tests for post-processing calibration functions."""
import numpy as np
import pytest

from src.metrics import smape
from src.postprocess import (
    apply_postprocessing,
    apply_power_law_calibration,
    calibrate_predictions_nested_cv,
    evaluate_power_law_nested_cv,
    optimize_clip_floor,
    optimize_global_multiplier,
    optimize_power_law_calibration,
)


class TestPostprocessingCalibration:

    def test_optimize_global_multiplier_improves_biased_predictions(self):
        np.random.seed(42)
        y_true = np.random.uniform(5.0, 100.0, size=1000)
        # Biased predictions (scaled by 1.15)
        y_pred = y_true * 1.15

        optimal_alpha, baseline_smape, calibrated_smape = optimize_global_multiplier(y_true, y_pred)

        # Multiplier should be roughly 1 / 1.15 ≈ 0.869
        assert optimal_alpha == pytest.approx(1.0 / 1.15, rel=1e-2)
        assert calibrated_smape < baseline_smape
        assert calibrated_smape == pytest.approx(0.0, abs=1e-3)

    def test_optimize_clip_floor_prevents_low_end_blowups(self):
        y_true = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
        # Prediction on first item is tiny (0.001) causing high SMAPE
        y_pred = np.array([0.001, 1.0, 2.0, 5.0, 10.0])

        optimal_floor, baseline, calibrated = optimize_clip_floor(y_true, y_pred, search_range=(0.1, 1.0))
        assert optimal_floor >= 0.1
        assert calibrated < baseline

    def test_calibrate_predictions_nested_cv_no_leakage(self):
        np.random.seed(42)
        y_true = np.random.uniform(2.0, 50.0, size=500)
        y_pred = y_true * 1.10 + np.random.normal(0, 1.0, size=500)
        y_pred = np.maximum(y_pred, 0.5)

        results = calibrate_predictions_nested_cv(y_true, y_pred, n_splits=5, seed=42)

        assert "mean_alpha" in results
        assert "mean_floor" in results
        assert "calibrated_cv_smape" in results
        assert results["calibrated_cv_smape"] <= results["baseline_smape"]
        assert results["mean_alpha"] < 1.0  # Correcting upward bias

    def test_apply_postprocessing_constraints(self):
        raw_preds = np.array([0.01, 5.5, 12.0, 45.0, 150.0])
        calibrated = apply_postprocessing(raw_preds, multiplier=0.98, clip_min=0.25)

        assert len(calibrated) == len(raw_preds)
        assert (calibrated > 0).all()
        assert (calibrated >= 0.25).all()
        assert not np.isnan(calibrated).any()
        assert calibrated[0] == 0.25

    def test_distribution_gap_and_alignment(self):
        from src.postprocess import analyze_distribution_gap, align_test_distribution
        train_p = np.array([5.0, 10.0, 15.0, 20.0, 50.0, 100.0, 250.0])
        # Suppose model outputs compressed test predictions
        test_p = np.array([12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0])
        gap = analyze_distribution_gap(train_p, test_p)
        assert "train" in gap and "test" in gap
        assert gap["train"]["mean"] > 0
        
        aligned = align_test_distribution(test_p, train_p, blend_weight=0.2)
        assert len(aligned) == len(test_p)
        assert (aligned > 0).all()
        # Rank preservation: aligned order should strictly match test_p order
        assert (np.diff(aligned) > 0).all()

    def test_power_law_monotonicity(self):
        """Verify that power-law calibration preserves rank order strictly (monotonicity)."""
        preds = np.linspace(0.1, 500.0, 1000)

        for a in [0.7, 0.9, 1.0, 1.15, 1.35]:
            for b in [-0.5, 0.0, 0.5]:
                calibrated = apply_power_law_calibration(preds, a=a, b=b, clip_min=0.05)
                diffs = np.diff(calibrated)
                # Calibrated prices must be strictly non-decreasing everywhere
                assert (diffs >= 0).all(), f"Monotonicity violated for a={a}, b={b}"
                # Above the clip_min floor, it must be strictly increasing
                above_floor = calibrated > 0.05
                if np.sum(above_floor) > 1:
                    assert (np.diff(calibrated[above_floor]) > 0).all()

    def test_power_law_numerical_stability(self):
        """Verify stability across extreme price ranges ($0.001 to $3,000+)."""
        extreme_preds = np.array([1e-5, 0.01, 0.5, 5.0, 25.0, 150.0, 1500.0, 3000.0])
        calibrated = apply_power_law_calibration(extreme_preds, a=1.35, b=0.2, clip_min=0.05)

        assert not np.isnan(calibrated).any()
        assert not np.isinf(calibrated).any()
        assert (calibrated >= 0.05).all()
        assert (calibrated > 0).all()
        assert calibrated[0] == 0.05  # Floor clamping applied

    def test_power_law_convergence(self):
        """Verify Nelder-Mead converges and fixes tail-compression / regression to mean."""
        np.random.seed(42)
        # Synthetic true prices with heavy right skew
        y_true = np.exp(np.random.normal(np.log(14.0), 1.0, size=2000))
        y_true = np.clip(y_true, 0.5, 500.0)

        # Compressed predictions (regression to mean):
        # low true values are overpredicted, high true values are underpredicted
        y_pred = np.exp(0.70 * np.log(y_true) + 0.30 * np.log(14.0) + np.random.normal(0, 0.05, size=2000))

        opt_a, opt_b, baseline_smape, calibrated_smape = optimize_power_law_calibration(
            y_true, y_pred, clip_min=0.05
        )

        # Optimal 'a' should stretch variance back out (a > 1.0)
        assert opt_a > 1.05
        assert 0.7 <= opt_a <= 1.4
        assert -1.5 <= opt_b <= 1.5
        assert calibrated_smape < baseline_smape
        assert baseline_smape - calibrated_smape > 3.0  # Noticeable improvement

    def test_evaluate_power_law_nested_cv(self):
        """Verify nested CV executes leak-free and reports stable parameters."""
        np.random.seed(42)
        y_true = np.exp(np.random.normal(np.log(20.0), 0.8, size=1000))
        y_pred = np.exp(0.85 * np.log(y_true) + 0.15 * np.log(20.0))

        results = evaluate_power_law_nested_cv(y_true, y_pred, n_splits=5, seed=42)

        assert "mean_a" in results
        assert "std_a" in results
        assert "mean_b" in results
        assert "std_b" in results
        assert "calibrated_cv_smape" in results
        assert "baseline_smape" in results
        assert results["calibrated_cv_smape"] <= results["baseline_smape"]
        assert results["std_a"] < 0.1  # Parameters are stable across folds

    def test_power_law_rank_safety_negative_exponent(self):
        """Verify that negative power exponents are clamped to prevent rank inversion."""
        raw_preds = np.array([2.0, 20.0, 200.0])
        # Attempting to invert ranking with negative exponent
        calibrated = apply_power_law_calibration(raw_preds, a=-1.0, b=0.0)

        # Must maintain positive ranking
        assert calibrated[0] < calibrated[1] < calibrated[2]

    def test_apply_postprocessing_with_power_law(self):
        """Verify apply_postprocessing seamless dispatch to power law."""
        raw_preds = np.array([1.0, 10.0, 100.0])
        cal = apply_postprocessing(raw_preds, power_a=1.1, power_b=-0.05, clip_min=0.1)

        assert len(cal) == 3
        assert (cal > 0).all()
        assert cal[0] < cal[1] < cal[2]
