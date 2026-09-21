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
from scipy.optimize import minimize, minimize_scalar
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


def apply_power_law_calibration(
    predictions: np.ndarray,
    a: float,
    b: float,
    clip_min: float = 0.05,
) -> np.ndarray:
    """Apply calibrated log-affine power-law transformation to predictions.

    Formula:
        y_cal = max(exp(a * ln(max(y_pred, 1e-4)) + b), clip_min)

    Args:
        predictions: Array of raw predicted prices (strictly positive).
        a: Power parameter (slope in log-space). Controls variance stretching/compression.
        b: Bias parameter (intercept in log-space). Controls overall level shift.
        clip_min: Minimum price threshold to prevent small zero-division errors.

    Returns:
        Array of calibrated positive prices.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    # Strictly enforce a >= 0.5 to prevent ranking inversion
    a_clamped = max(float(a), 0.5)
    b_val = float(b)

    log_preds = np.log(np.maximum(preds, 1e-4))
    calibrated = np.maximum(np.exp(a_clamped * log_preds + b_val), clip_min)

    assert not np.isnan(calibrated).any(), "NaN found in calibrated predictions"
    assert not np.isinf(calibrated).any(), "Inf found in calibrated predictions"
    assert (calibrated > 0).all(), "Non-positive price found in calibrated predictions"

    return calibrated


def optimize_power_law_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clip_min: float = 0.05,
    a_bounds: Tuple[float, float] = (0.7, 1.4),
    b_bounds: Tuple[float, float] = (-1.5, 1.5),
) -> Tuple[float, float, float, float]:
    """Find optimal log-affine power-law parameters (a*, b*) minimizing SMAPE.

    Mathematical formulation:
        ln(y_cal) = a * ln(y_pred) + b  ==>  y_cal = exp(b) * (y_pred)^a

    When a > 1, the transformation stretches the distribution:
    - Low predictions (< median) contract downward (fixing Decile 0 overprediction).
    - High predictions (> median) expand upward (fixing Decile 9 underprediction).
    - The bias term b keeps the median price calibrated.

    Args:
        y_true: Ground truth positive prices.
        y_pred: Predicted positive prices.
        clip_min: Lower floor threshold.
        a_bounds: Feasible interval for a [min_a, max_a].
        b_bounds: Feasible interval for b [min_b, max_b].

    Returns:
        Tuple of (optimal_a, optimal_b, baseline_smape, calibrated_smape).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    baseline_smape = float(smape(y_true, y_pred))
    log_safe_preds = np.log(np.maximum(y_pred, 1e-4))

    def objective(params: np.ndarray) -> float:
        a_val, b_val = float(params[0]), float(params[1])
        # Prevent price rank inversion
        a_eff = max(a_val, 0.5)
        # Vectorized power-law calculation
        cal_preds = np.maximum(np.exp(a_eff * log_safe_preds + b_val), clip_min)
        return float(smape(y_true, cal_preds))

    x0 = np.array([1.0, 0.0], dtype=np.float64)
    bounds = [a_bounds, b_bounds]

    res = minimize(
        objective,
        x0=x0,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 500, "xatol": 1e-4, "fatol": 1e-4},
    )

    a_opt = float(np.clip(res.x[0], a_bounds[0], a_bounds[1]))
    b_opt = float(np.clip(res.x[1], b_bounds[0], b_bounds[1]))

    final_calibrated = apply_power_law_calibration(y_pred, a=a_opt, b=b_opt, clip_min=clip_min)
    calibrated_smape = float(smape(y_true, final_calibrated))

    return a_opt, b_opt, baseline_smape, calibrated_smape


def evaluate_power_law_nested_cv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    clip_min: float = 0.05,
) -> Dict[str, float]:
    """Evaluate stability of power-law calibration using nested K-Fold CV.

    Fits (a, b) on (K-1) folds and evaluates on the validation fold
    to guarantee zero target leakage.

    Args:
        y_true: Ground truth positive prices.
        y_pred: Raw out-of-fold predictions.
        n_splits: Number of cross-validation folds.
        seed: Random state seed.
        clip_min: Lower floor threshold.

    Returns:
        Dict with mean_a, std_a, mean_b, std_b, baseline_smape, calibrated_cv_smape, smape_delta.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    a_list = []
    b_list = []
    val_calibrated_preds = np.zeros_like(y_pred)

    for train_idx, val_idx in kf.split(y_pred):
        y_tr, p_tr = y_true[train_idx], y_pred[train_idx]
        y_va, p_va = y_true[val_idx], y_pred[val_idx]

        a_opt, b_opt, _, _ = optimize_power_law_calibration(y_tr, p_tr, clip_min=clip_min)
        a_list.append(a_opt)
        b_list.append(b_opt)

        val_calibrated_preds[val_idx] = apply_power_law_calibration(
            p_va, a=a_opt, b=b_opt, clip_min=clip_min
        )

    baseline_score = float(smape(y_true, y_pred))
    calibrated_cv_score = float(smape(y_true, val_calibrated_preds))

    return {
        "mean_a": float(np.mean(a_list)),
        "std_a": float(np.std(a_list)),
        "mean_b": float(np.mean(b_list)),
        "std_b": float(np.std(b_list)),
        "baseline_smape": baseline_score,
        "calibrated_cv_smape": calibrated_cv_score,
        "smape_delta": baseline_score - calibrated_cv_score,
    }


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


def analyze_distribution_gap(
    train_prices: np.ndarray,
    test_preds: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """Compare summary statistics between training targets and test predictions.

    Helps diagnose distribution compression or shift between validation and test.
    """
    tr = np.asarray(train_prices, dtype=np.float64)
    te = np.asarray(test_preds, dtype=np.float64)

    quantiles = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
    return {
        "train": {
            "mean": float(np.mean(tr)),
            "std": float(np.std(tr)),
            "median": float(np.median(tr)),
            **{f"q{int(q*100):02d}": float(np.quantile(tr, q)) for q in quantiles},
        },
        "test": {
            "mean": float(np.mean(te)),
            "std": float(np.std(te)),
            "median": float(np.median(te)),
            **{f"q{int(q*100):02d}": float(np.quantile(te, q)) for q in quantiles},
        },
    }


def align_test_distribution(
    test_preds: np.ndarray,
    train_prices: np.ndarray,
    blend_weight: float = 0.15,
) -> np.ndarray:
    """Gently align test prediction quantiles with empirical train price quantiles.

    Preserves exact rank order of test predictions while adjusting for variance
    compression common in log-trained regression models.

    Args:
        test_preds: Array of model test predictions.
        train_prices: Array of ground-truth training prices.
        blend_weight: Weight given to quantile mapping (0.0 = original, 1.0 = pure quantile mapping).

    Returns:
        Distribution-aligned positive price array.
    """
    te = np.asarray(test_preds, dtype=np.float64)
    tr = np.asarray(train_prices, dtype=np.float64)

    # Rank-preserving percentile mapping
    percentiles = (np.argsort(np.argsort(te)) + 0.5) / len(te)
    mapped_prices = np.quantile(tr, percentiles)

    aligned = (1.0 - blend_weight) * te + blend_weight * mapped_prices
    return np.maximum(aligned, 0.05)


def apply_postprocessing(
    predictions: np.ndarray,
    multiplier: float = 1.0,
    clip_min: float = 0.05,
    power_a: Optional[float] = None,
    power_b: Optional[float] = None,
) -> np.ndarray:
    """Apply calibrated multiplier or power-law transformation and lower floor to predictions.

    Args:
        predictions: Array of raw model predictions.
        multiplier: Scaling factor alpha (used when power_a/power_b are None).
        clip_min: Optimal lower floor threshold.
        power_a: Optional power-law exponent.
        power_b: Optional power-law bias.

    Returns:
        Calibrated positive price array.
    """
    if power_a is not None and power_b is not None:
        return apply_power_law_calibration(predictions, a=power_a, b=power_b, clip_min=clip_min)

    preds = np.asarray(predictions, dtype=np.float64)
    calibrated = np.maximum(preds * multiplier, clip_min)

    assert not np.isnan(calibrated).any(), "NaN found in calibrated predictions"
    assert not np.isinf(calibrated).any(), "Inf found in calibrated predictions"
    assert (calibrated > 0).all(), "Non-positive price found in calibrated predictions"
    return calibrated


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate SMAPE predictions with post-processing.")
    parser.add_argument("--sub_file", type=str, default="dataset/test_out.csv", help="Path to submission CSV to calibrate")
    parser.add_argument("--alpha", type=float, default=1.0, help="Explicit multiplier to apply")
    parser.add_argument("--clip_min", type=float, default=0.05, help="Explicit lower floor to apply")
    parser.add_argument("--power_a", type=float, default=None, help="Explicit power-law exponent a")
    parser.add_argument("--power_b", type=float, default=None, help="Explicit power-law bias b")
    parser.add_argument("--output_file", type=str, default=None, help="Output path (defaults to overwriting sub_file)")
    args = parser.parse_args()

    if not os.path.exists(args.sub_file):
        raise FileNotFoundError(f"Submission file not found: {args.sub_file}")

    df = pd.read_csv(args.sub_file)
    print(f"Loaded submission: {args.sub_file} ({len(df)} rows)")

    raw_prices = df["price"].values
    if args.power_a is not None and args.power_b is not None:
        calibrated_prices = apply_power_law_calibration(
            raw_prices, a=args.power_a, b=args.power_b, clip_min=args.clip_min
        )
        print(f"Applying Power-Law Calibration (a={args.power_a:.4f}, b={args.power_b:.4f}, clip_min={args.clip_min:.4f})")
    else:
        calibrated_prices = apply_postprocessing(
            raw_prices, multiplier=args.alpha, clip_min=args.clip_min
        )
        print(f"Applying Multiplier Calibration (alpha={args.alpha:.4f}, clip_min={args.clip_min:.4f})")

    df["price"] = calibrated_prices
    out_path = args.output_file or args.sub_file
    df.to_csv(out_path, index=False)
    print(f"Saved calibrated submission to {out_path}")


if __name__ == "__main__":
    main()
