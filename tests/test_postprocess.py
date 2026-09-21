"""Tests for post-processing calibration functions."""
import numpy as np
import pytest

from src.metrics import smape
from src.postprocess import (
    apply_postprocessing,
    calibrate_predictions_nested_cv,
    optimize_clip_floor,
    optimize_global_multiplier,
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
