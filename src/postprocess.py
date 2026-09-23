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


def apply_asymmetric_piecewise_calibration(
    predictions: np.ndarray,
    beta_low: float,
    beta_high: float,
    t_low: float = 5.0,
    t_high: float = 40.0,
    clip_min: float = 0.05,
) -> np.ndarray:
    """Apply C1-continuous asymmetric piecewise calibration in log-space.

    Formula:
        For z = ln(y):
        z_cal = z - 0.5 * beta_low * (z - ln(t_low))^2   if y < t_low
        z_cal = z                                       if t_low <= y <= t_high
        z_cal = z + 0.5 * beta_high * (z - ln(t_high))^2 if y > t_high
        y_cal = max(exp(z_cal), clip_min)

    Mathematical Properties:
    - C0 continuity at t_low and t_high (matching exact middle values).
    - C1 continuity at t_low and t_high (matching exact middle slope 1.0).
    - Strict monotonicity everywhere for beta_low >= 0, beta_high >= 0 (zero ranking inversions).
    - Non-regression: When beta_low = beta_high = 0, y_cal = y identically.

    Args:
        predictions: Array of input predictions.
        beta_low: Downward contraction intensity for low-priced items (>= 0).
        beta_high: Upward expansion intensity for high-priced items (>= 0).
        t_low: Lower boundary threshold (default $5.00).
        t_high: Upper boundary threshold (default $40.00).
        clip_min: Minimum floor threshold.

    Returns:
        Calibrated positive price array.
    """
    preds = np.asarray(predictions, dtype=np.float64)
    b_low = max(float(beta_low), 0.0)
    b_high = max(float(beta_high), 0.0)

    if b_low == 0.0 and b_high == 0.0:
        return np.maximum(preds, clip_min)

    log_preds = np.log(np.maximum(preds, 1e-4))
    log_t_low = np.log(max(float(t_low), 1e-3))
    log_t_high = np.log(max(float(t_high), float(t_low) + 0.1))

    log_cal = log_preds.copy()

    # Low zone: z < log_t_low (pull down toward true low-end prices)
    low_mask = log_preds < log_t_low
    if np.any(low_mask):
        diff_low = log_preds[low_mask] - log_t_low
        log_cal[low_mask] = log_preds[low_mask] - 0.5 * b_low * (diff_low ** 2)

    # High zone: z > log_t_high (expand up toward true luxury prices)
    high_mask = log_preds > log_t_high
    if np.any(high_mask):
        diff_high = log_preds[high_mask] - log_t_high
        log_cal[high_mask] = log_preds[high_mask] + 0.5 * b_high * (diff_high ** 2)

    calibrated = np.maximum(np.exp(np.clip(log_cal, -3.0, 9.5)), clip_min)

    assert not np.isnan(calibrated).any(), "NaN found in asymmetric calibrated predictions"
    assert not np.isinf(calibrated).any(), "Inf found in asymmetric calibrated predictions"
    assert (calibrated > 0).all(), "Non-positive price found in asymmetric calibrated predictions"

    return calibrated


def optimize_asymmetric_piecewise_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    t_low: float = 5.0,
    t_high: float = 40.0,
    clip_min: float = 0.05,
    beta_bounds: Tuple[float, float] = (0.0, 1.5),
) -> Tuple[float, float, float, float, float, float]:
    """Find optimal (beta_low*, beta_high*) minimizing SMAPE.

    Args:
        y_true: Ground truth positive prices.
        y_pred: Predicted positive prices.
        t_low: Lower boundary threshold.
        t_high: Upper boundary threshold.
        clip_min: Lower floor threshold.
        beta_bounds: Feasible interval for each beta parameter.

    Returns:
        Tuple of (beta_low_opt, beta_high_opt, t_low, t_high, baseline_smape, calibrated_smape).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    baseline_smape = float(smape(y_true, y_pred))

    def objective(params: np.ndarray) -> float:
        b_l = max(float(params[0]), 0.0)
        b_h = max(float(params[1]), 0.0)
        cal = apply_asymmetric_piecewise_calibration(
            y_pred, beta_low=b_l, beta_high=b_h, t_low=t_low, t_high=t_high, clip_min=clip_min
        )
        return float(smape(y_true, cal))

    x0 = np.array([0.0, 0.0], dtype=np.float64)
    bounds = [beta_bounds, beta_bounds]

    res = minimize(
        objective,
        x0=x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 300, "ftol": 1e-5},
    )

    if res.fun >= baseline_smape:
        res_nm = minimize(
            objective,
            x0=np.array([0.1, 0.1], dtype=np.float64),
            method="Nelder-Mead",
            bounds=bounds,
            options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-4},
        )
        if res_nm.fun < res.fun:
            res = res_nm

    b_low_opt = float(np.clip(res.x[0], beta_bounds[0], beta_bounds[1]))
    b_high_opt = float(np.clip(res.x[1], beta_bounds[0], beta_bounds[1]))

    final_cal = apply_asymmetric_piecewise_calibration(
        y_pred, beta_low=b_low_opt, beta_high=b_high_opt, t_low=t_low, t_high=t_high, clip_min=clip_min
    )
    calibrated_smape = float(smape(y_true, final_cal))

    if calibrated_smape >= baseline_smape:
        return 0.0, 0.0, t_low, t_high, baseline_smape, baseline_smape

    return b_low_opt, b_high_opt, t_low, t_high, baseline_smape, calibrated_smape


def evaluate_asymmetric_piecewise_nested_cv(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_splits: int = 5,
    seed: int = 42,
    t_low: float = 5.0,
    t_high: float = 40.0,
    clip_min: float = 0.05,
) -> Dict[str, float]:
    """Evaluate stability of asymmetric piecewise calibration using nested K-Fold CV.

    Fits (beta_low, beta_high) on (K-1) folds and evaluates on the validation fold
    to guarantee zero target leakage.

    Returns:
        Dict with parameter means, standard deviations, and cross-validated SMAPE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    b_low_list = []
    b_high_list = []
    val_calibrated_preds = np.zeros_like(y_pred)

    for train_idx, val_idx in kf.split(y_pred):
        y_tr, p_tr = y_true[train_idx], y_pred[train_idx]
        y_va, p_va = y_true[val_idx], y_pred[val_idx]

        b_l, b_h, _, _, _, _ = optimize_asymmetric_piecewise_calibration(
            y_tr, p_tr, t_low=t_low, t_high=t_high, clip_min=clip_min
        )
        b_low_list.append(b_l)
        b_high_list.append(b_h)

        val_calibrated_preds[val_idx] = apply_asymmetric_piecewise_calibration(
            p_va, beta_low=b_l, beta_high=b_h, t_low=t_low, t_high=t_high, clip_min=clip_min
        )

    baseline_score = float(smape(y_true, y_pred))
    calibrated_cv_score = float(smape(y_true, val_calibrated_preds))

    return {
        "mean_beta_low": float(np.mean(b_low_list)),
        "std_beta_low": float(np.std(b_low_list)),
        "mean_beta_high": float(np.mean(b_high_list)),
        "std_beta_high": float(np.std(b_high_list)),
        "baseline_smape": baseline_score,
        "calibrated_cv_smape": calibrated_cv_score,
        "smape_delta": baseline_score - calibrated_cv_score,
        "t_low": float(t_low),
        "t_high": float(t_high),
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
    beta_low: Optional[float] = None,
    beta_high: Optional[float] = None,
    t_low: float = 5.0,
    t_high: float = 40.0,
) -> np.ndarray:
    """Apply calibrated multiplier, power-law, or asymmetric piecewise transformation to predictions.

    Args:
        predictions: Array of raw model predictions.
        multiplier: Scaling factor alpha (used when power_a/power_b are None).
        clip_min: Optimal lower floor threshold.
        power_a: Optional power-law exponent.
        power_b: Optional power-law bias.
        beta_low: Optional downward contraction parameter for low-end prices (< t_low).
        beta_high: Optional upward expansion parameter for high-end prices (> t_high).
        t_low: Lower boundary threshold (default $5.00).
        t_high: Upper boundary threshold (default $40.00).

    Returns:
        Calibrated positive price array.
    """
    if power_a is not None and power_b is not None:
        cal = apply_power_law_calibration(predictions, a=power_a, b=power_b, clip_min=clip_min)
    else:
        preds = np.asarray(predictions, dtype=np.float64)
        cal = np.maximum(preds * multiplier, clip_min)

    if beta_low is not None and beta_high is not None:
        cal = apply_asymmetric_piecewise_calibration(
            cal, beta_low=beta_low, beta_high=beta_high, t_low=t_low, t_high=t_high, clip_min=clip_min
        )

    assert not np.isnan(cal).any(), "NaN found in calibrated predictions"
    assert not np.isinf(cal).any(), "Inf found in calibrated predictions"
    assert (cal > 0).all(), "Non-positive price found in calibrated predictions"
    return cal


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
