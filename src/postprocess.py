"""Post-processing calibration for SMAPE optimization.

Provides:
- Global scalar multiplier calibration (corrects log-symmetric distortion under SMAPE).
- Lower-bound floor calibration (clip_min) to avoid catastrophic relative errors on cheap items.
- Nested cross-validation for leak-free calibration parameter selection.
"""
from __future__ import annotations

import argparse
import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.model_selection import KFold

from src.metrics import smape


def optimize_global_multiplier(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    search_range: Tuple[float, float] = (0.85, 1.15),
) -> Tuple[float, float, float]:
    """Find the single scalar multiplier alpha that minimizes SMAPE(y_true, alpha * y_pred).

    Args:
        y_true: Ground truth positive prices.
        y_pred: Predicted positive prices.
        search_range: Bounds for the scalar search.

    Returns:
        Tuple of (optimal_alpha, baseline_smape, calibrated_smape).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    baseline_smape = float(smape(y_true, y_pred))

    def objective(alpha: float) -> float:
        scaled = np.maximum(y_pred * alpha, 1e-4)
        return float(smape(y_true, scaled))

    res = minimize_scalar(objective, bounds=search_range, method="bounded")
    optimal_alpha = float(res.x)
    calibrated_smape = float(res.fun)

    return optimal_alpha, baseline_smape, calibrated_smape


def optimize_clip_floor(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    search_range: Tuple[float, float] = (0.05, 1.50),
) -> Tuple[float, float, float]:
    """Find the optimal lower-bound floor (clip_min) that minimizes SMAPE.

    Args:
        y_true: Ground truth positive prices.
        y_pred: Predicted positive prices.
        search_range: Bounds for the lower floor.

    Returns:
        Tuple of (optimal_floor, baseline_smape, calibrated_smape).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    baseline_smape = float(smape(y_true, y_pred))

    def objective(floor: float) -> float:
        clamped = np.maximum(y_pred, floor)
        return float(smape(y_true, clamped))

    res = minimize_scalar(objective, bounds=search_range, method="bounded")
    optimal_floor = float(res.x)
    calibrated_smape = float(res.fun)

    return optimal_floor, baseline_smape, calibrated_smape


def calibrate_predictions_nested_cv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
) -> Dict[str, float]:
    """Evaluate stability of multiplier and floor using nested K-Fold CV.

    Fits alpha and floor on (K-1) folds and evaluates on the validation fold
    to guarantee that post-processing does not overfit OOF noise.

    Args:
        y_true: Ground truth positive prices.
        y_pred: Raw out-of-fold predictions.
        n_splits: Number of cross-validation folds.
        seed: Random state seed.

    Returns:
        Dict with mean_alpha, mean_floor, baseline_smape, and calibrated_cv_smape.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    alphas = []
    floors = []
    val_calibrated_preds = np.zeros_like(y_pred)

    for train_idx, val_idx in kf.split(y_pred):
        y_tr, p_tr = y_true[train_idx], y_pred[train_idx]
        y_va, p_va = y_true[val_idx], y_pred[val_idx]

        # 1. Fit multiplier on train fold
        alpha, _, _ = optimize_global_multiplier(y_tr, p_tr)
        alphas.append(alpha)

        # 2. Fit floor on train fold after scaling
        p_tr_scaled = p_tr * alpha
        floor, _, _ = optimize_clip_floor(y_tr, p_tr_scaled)
        floors.append(floor)

        # 3. Apply to validation fold
        val_calibrated_preds[val_idx] = np.maximum(p_va * alpha, floor)

    baseline_score = float(smape(y_true, y_pred))
    calibrated_cv_score = float(smape(y_true, val_calibrated_preds))

    return {
        "mean_alpha": float(np.mean(alphas)),
        "std_alpha": float(np.std(alphas)),
        "mean_floor": float(np.mean(floors)),
        "std_floor": float(np.std(floors)),
        "baseline_smape": baseline_score,
        "calibrated_cv_smape": calibrated_cv_score,
        "smape_delta": baseline_score - calibrated_cv_score,
    }


def apply_postprocessing(
    predictions: np.ndarray,
    multiplier: float = 1.0,
    clip_min: float = 0.05,
) -> np.ndarray:
    """Apply calibrated multiplier and lower floor to test predictions.

    Args:
        predictions: Array of raw model predictions.
        multiplier: Optimal scaling factor alpha.
        clip_min: Optimal lower floor threshold.

    Returns:
        Calibrated positive price array.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    calibrated = np.maximum(preds * multiplier, clip_min)

    assert not np.isnan(calibrated).any(), "NaN found in calibrated predictions"
    assert (calibrated > 0).all(), "Non-positive price found in calibrated predictions"
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SMAPE predictions with post-processing.")
    parser.add_argument("--sub_file", type=str, default="dataset/test_out.csv", help="Path to submission CSV to calibrate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Explicit multiplier to apply")
    parser.add_argument("--clip_min", type=float, default=0.05, help="Explicit lower floor to apply")
    parser.add_argument("--output_file", type=str, default=None, help="Output path (defaults to overwriting sub_file)")
    args = parser.parse_args()

    if not os.path.exists(args.sub_file):
        raise FileNotFoundError(f"Submission file not found: {args.sub_file}")

    df = pd.read_csv(args.sub_file)
    print(f"Loaded submission: {args.sub_file} ({len(df)} rows)")

    raw_prices = df["price"].values
    calibrated_prices = apply_postprocessing(raw_prices, multiplier=args.alpha, clip_min=args.clip_min)

    df["price"] = calibrated_prices
    out_path = args.output_file or args.sub_file
    df.to_csv(out_path, index=False)
    print(f"Saved calibrated submission to {out_path} (alpha={args.alpha:.4f}, clip_min={args.clip_min:.4f})")


if __name__ == "__main__":
    main()
