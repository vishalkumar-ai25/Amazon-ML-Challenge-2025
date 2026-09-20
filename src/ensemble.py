"""Ensemble and blending optimizer for the Amazon ML Challenge 2025.

Optimizes convex combination weights across diverse models
(e.g., LightGBM, CatBoost, Ridge) to directly minimize SMAPE error.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from src.metrics import smape


def find_optimal_blend_weights(
    oof_predictions: dict[str, np.ndarray],
    y_true: np.ndarray,
    *,
    seed: int = 42,
) -> dict[str, float]:
    """Find convex combination weights that directly minimize SMAPE on OOF predictions.

    Args:
        oof_predictions: Mapping of model_name -> 1D array of OOF price predictions.
        y_true: 1D array of ground truth prices.
        seed: Random seed for optimizer initialization.

    Returns:
        Mapping of model_name -> normalized weight (sum to 1.0, non-negative).
    """
    model_names = list(oof_predictions.keys())
    n_models = len(model_names)

    if n_models == 1:
        return {model_names[0]: 1.0}

    # Stack predictions matrix: (n_samples, n_models)
    P = np.column_stack([oof_predictions[name] for name in model_names])

    def objective(weights: np.ndarray) -> float:
        # Softmax or normalized projection to enforce sum(w) = 1 and w >= 0
        w = np.maximum(weights, 0)
        s = np.sum(w)
        if s > 0:
            w = w / s
        else:
            w = np.ones(n_models) / n_models

        blended = np.dot(P, w)
        return smape(y_true, blended)

    # Initial equal weights
    initial_weights = np.ones(n_models) / n_models
    bounds = [(0.0, 1.0) for _ in range(n_models)]

    res = minimize(
        objective,
        initial_weights,
        method="Nelder-Mead",
        bounds=bounds,
        options={"maxiter": 500, "disp": False},
    )

    w_opt = np.maximum(res.x, 0)
    w_sum = np.sum(w_opt)
    if w_sum > 0:
        w_opt = w_opt / w_sum
    else:
        w_opt = np.ones(n_models) / n_models

    return {name: float(w_opt[i]) for i, name in enumerate(model_names)}


def apply_blend(
    predictions_dict: dict[str, np.ndarray],
    weights: dict[str, float],
    *,
    clip_min: float = 1e-5,
) -> np.ndarray:
    """Compute weighted linear combination of predictions.

    Args:
        predictions_dict: Mapping of model_name -> 1D array of predictions.
        weights: Mapping of model_name -> weight.
        clip_min: Minimum price clip floor.

    Returns:
        Blended 1D prediction array.
    """
    first_key = next(iter(predictions_dict.keys()))
    n_samples = len(predictions_dict[first_key])
    blended = np.zeros(n_samples, dtype=np.float64)

    total_weight = sum(weights.get(name, 0.0) for name in predictions_dict.keys())
    if total_weight <= 0:
        total_weight = 1.0

    for name, preds in predictions_dict.items():
        w = weights.get(name, 0.0) / total_weight
        blended += w * np.asarray(preds, dtype=np.float64)

    return np.maximum(blended, clip_min)
