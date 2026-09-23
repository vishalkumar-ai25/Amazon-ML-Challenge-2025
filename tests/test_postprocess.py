"""Tests for post-processing calibration functions."""
import numpy as np
import pytest

from src.metrics import smape
from src.postprocess import (
    apply_asymmetric_piecewise_calibration,
    apply_postprocessing,
    apply_power_law_calibration,
    calibrate_predictions_nested_cv,
    evaluate_asymmetric_piecewise_nested_cv,
    evaluate_power_law_nested_cv,
    optimize_asymmetric_piecewise_calibration,
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

    def test_asymmetric_piecewise_c0_continuity(self):
        """Verify C0 continuity at boundary thresholds T_low and T_high."""
        t_low, t_high = 5.0, 40.0
        eps = 1e-6
        preds = np.array([t_low - eps, t_low, t_low + eps, t_high - eps, t_high, t_high + eps])
        cal = apply_asymmetric_piecewise_calibration(preds, beta_low=0.6, beta_high=0.4, t_low=t_low, t_high=t_high)

        # Values across t_low boundary must match within 1e-4
        assert abs(cal[0] - cal[1]) < 1e-4
        assert abs(cal[1] - cal[2]) < 1e-4
        # Values across t_high boundary must match within 1e-4
        assert abs(cal[3] - cal[4]) < 1e-4
        assert abs(cal[4] - cal[5]) < 1e-4

    def test_asymmetric_piecewise_c1_continuity(self):
        """Verify C1 derivative continuity at boundary thresholds."""
        t_low, t_high = 5.0, 40.0
        h = 1e-5
        # Finite difference numerical derivatives on left and right of boundaries
        # For t_low:
        f_l_minus = apply_asymmetric_piecewise_calibration(np.array([t_low - h]), beta_low=0.8, beta_high=0.5, t_low=t_low, t_high=t_high)[0]
        f_mid_low = apply_asymmetric_piecewise_calibration(np.array([t_low]), beta_low=0.8, beta_high=0.5, t_low=t_low, t_high=t_high)[0]
        f_l_plus = apply_asymmetric_piecewise_calibration(np.array([t_low + h]), beta_low=0.8, beta_high=0.5, t_low=t_low, t_high=t_high)[0]

        d_left = (f_mid_low - f_l_minus) / h
        d_right = (f_l_plus - f_mid_low) / h
        assert abs(d_left - d_right) < 1e-3, f"Slope discontinuity at t_low: d_left={d_left}, d_right={d_right}"

        # For t_high:
        f_h_minus = apply_asymmetric_piecewise_calibration(np.array([t_high - h]), beta_low=0.8, beta_high=0.5, t_low=t_low, t_high=t_high)[0]
        f_mid_high = apply_asymmetric_piecewise_calibration(np.array([t_high]), beta_low=0.8, beta_high=0.5, t_low=t_low, t_high=t_high)[0]
        f_h_plus = apply_asymmetric_piecewise_calibration(np.array([t_high + h]), beta_low=0.8, beta_high=0.5, t_low=t_low, t_high=t_high)[0]

        d_left_h = (f_mid_high - f_h_minus) / h
        d_right_h = (f_h_plus - f_mid_high) / h
        assert abs(d_left_h - d_right_h) < 1e-3, f"Slope discontinuity at t_high: d_left={d_left_h}, d_right={d_right_h}"

    def test_asymmetric_piecewise_strict_monotonicity(self):
        """Verify strict monotonicity across the entire range (zero rank inversions)."""
        np.random.seed(42)
        sorted_preds = np.sort(np.random.uniform(0.1, 500.0, size=2000))

        for b_low in [0.0, 0.2, 0.5, 1.0]:
            for b_high in [0.0, 0.2, 0.5, 1.0]:
                cal = apply_asymmetric_piecewise_calibration(
                    sorted_preds, beta_low=b_low, beta_high=b_high, t_low=5.0, t_high=40.0
                )
                diffs = np.diff(cal)
                assert (diffs >= 0).all(), f"Monotonicity violated for beta_low={b_low}, beta_high={b_high}"

    def test_asymmetric_piecewise_zero_beta_identity(self):
        """Verify that beta_low=0 and beta_high=0 yields the exact identity transformation."""
        preds = np.array([0.5, 2.5, 5.0, 15.0, 40.0, 100.0, 500.0])
        cal = apply_asymmetric_piecewise_calibration(preds, beta_low=0.0, beta_high=0.0)
        np.testing.assert_allclose(cal, preds, rtol=1e-5)

    def test_asymmetric_piecewise_directionality(self):
        """Verify downward contraction for low prices and upward expansion for high prices."""
        t_low, t_high = 5.0, 40.0
        preds = np.array([2.0, 10.0, 80.0])
        cal = apply_asymmetric_piecewise_calibration(preds, beta_low=0.5, beta_high=0.5, t_low=t_low, t_high=t_high)

        # y < t_low: downward contraction
        assert cal[0] < preds[0]
        # t_low <= y <= t_high: unchanged
        assert cal[1] == pytest.approx(preds[1], rel=1e-5)
        # y > t_high: upward expansion
        assert cal[2] > preds[2]

    def test_optimize_asymmetric_piecewise_calibration_on_skewed_data(self):
        """Verify optimization detects and fixes tail compression."""
        np.random.seed(42)
        # Ground truth prices
        y_true = np.exp(np.random.normal(np.log(15.0), 1.2, size=1500))
        # Compressed predictions: low end dragged up, high end dragged down
        y_pred = y_true.copy()
        low_idx = y_true < 5.0
        y_pred[low_idx] = y_true[low_idx] * 1.5  # overpredicted
        high_idx = y_true > 40.0
        y_pred[high_idx] = y_true[high_idx] * 0.7  # underpredicted

        b_l, b_h, _, _, base_sm, cal_sm = optimize_asymmetric_piecewise_calibration(
            y_true, y_pred, t_low=5.0, t_high=40.0
        )

        assert b_l > 0.05, f"Expected positive beta_low to fix overprediction, got {b_l}"
        assert b_h > 0.05, f"Expected positive beta_high to fix underprediction, got {b_h}"
        assert cal_sm < base_sm
        assert base_sm - cal_sm > 1.0  # Noticeable improvement

    def test_evaluate_asymmetric_piecewise_nested_cv(self):
        """Verify nested CV executes leak-free without error and parameters stay bounded."""
        np.random.seed(42)
        y_true = np.exp(np.random.normal(np.log(15.0), 1.0, size=800))
        y_pred = y_true * 1.05

        res = evaluate_asymmetric_piecewise_nested_cv(y_true, y_pred, n_splits=5, seed=42)
        assert "mean_beta_low" in res
        assert "mean_beta_high" in res
        assert "calibrated_cv_smape" in res
        assert res["calibrated_cv_smape"] <= res["baseline_smape"]
        assert 0.0 <= res["mean_beta_low"] <= 1.5
        assert 0.0 <= res["mean_beta_high"] <= 1.5

    def test_apply_postprocessing_with_asymmetric_dispatch(self):
        """Verify apply_postprocessing correctly cascades power-law and asymmetric calibration."""
        preds = np.array([2.0, 15.0, 100.0])
        cal = apply_postprocessing(
            preds, power_a=1.05, power_b=0.0, beta_low=0.4, beta_high=0.3, clip_min=0.1
        )
        assert len(cal) == 3
        assert (cal > 0).all()
        assert cal[0] < cal[1] < cal[2]

