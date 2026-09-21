"""Custom SMAPE Objective and Evaluation Metrics for GBDT Models (LightGBM & XGBoost).

Directly aligns tree split criteria and early stopping with the competition metric:
    SMAPE = 100 * mean( |pred - true| / ((|true| + |pred|) / 2) )

Operates in log-space:
    Model outputs z = ln(pred), so pred = exp(z).
    Gradients and Hessians are analytically computed with respect to z.
"""
from __future__ import annotations

from typing import Tuple, Union
import numpy as np


def compute_smape_grad_hess(
    z_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-4,
    min_hess: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute analytical gradient and hessian of SMAPE with respect to log-prediction z.

    Let y = true price (> 0), y_hat = exp(z) (> 0).
    L(z, y) = 2 * |y_hat - y| / (y_hat + y)

    First derivative with respect to z:
        dL/dz = (dL/dy_hat) * (dy_hat/dz)
              = sign(y_hat - y) * 4 * y * y_hat / (y_hat + y)^2

    We apply Charbonnier smoothing to |y_hat - y|:
        sqrt((y_hat - y)^2 + eps^2)
    yielding smooth gradient:
        g = ((y_hat - y) / sqrt((y_hat - y)^2 + eps^2)) * (4 * y * y_hat / ((y_hat + y)^2 + eps))

    Hessian approximation:
        h = max(4 * y * y_hat / ((y_hat + y)^2 + eps), min_hess)

    Args:
        z_pred: Log-predictions from the GBDT model (1D array).
        y_true: Ground-truth prices (1D array, strictly positive).
        eps: Small smoothing constant to avoid singularity and division by zero.
        min_hess: Floor on hessian values to ensure positive definiteness in tree splits.

    Returns:
        (grad, hess) tuple of float32 or float64 arrays matching z_pred.shape.
    """
    # Clamp log-predictions to stable bounds [exp(-3) ≈ 0.05, exp(9) ≈ 8103]
    z_clamped = np.clip(z_pred, -3.0, 9.0)
    y_hat = np.exp(z_clamped)
    y = np.maximum(y_true, eps)

    diff = y_hat - y
    denom = (y_hat + y) ** 2 + eps
    smooth_sign = diff / np.sqrt(diff ** 2 + eps ** 2)

    # 1st order gradient (bounded in [-1, 1])
    grad = smooth_sign * (4.0 * y * y_hat / denom)

    # 2nd order hessian (strictly positive, well-conditioned)
    hess = np.maximum(4.0 * y * y_hat / denom, min_hess)

    return grad, hess


# ---------------------------------------------------------------------------
# LightGBM Custom Objective & Evaluation Callbacks
# ---------------------------------------------------------------------------

def lgb_smape_objective(a, b) -> Tuple[np.ndarray, np.ndarray]:
    """LightGBM custom objective supporting both:
    1. sklearn API: (y_true, y_pred)
    2. lgb.train API: (y_pred, train_dataset)
    """
    if hasattr(b, "get_label"):
        preds = a
        labels = b.get_label()
    elif hasattr(a, "get_label"):
        preds = b
        labels = a.get_label()
    else:
        # sklearn LGBMRegressor passes (y_true, y_pred)
        labels = a
        preds = b

    if np.median(labels) < 15.0 and (labels > 0).all():
        y_true = np.exp(labels)
    else:
        y_true = labels

    grad, hess = compute_smape_grad_hess(preds, y_true)
    return grad, hess


def lgb_smape_eval(a, b) -> Tuple[str, float, bool]:
    """LightGBM custom evaluation metric supporting both sklearn and train API."""
    if hasattr(b, "get_label"):
        preds = a
        labels = b.get_label()
    elif hasattr(a, "get_label"):
        preds = b
        labels = a.get_label()
    else:
        labels = a
        preds = b

    if np.median(labels) < 15.0 and (labels > 0).all():
        y_true = np.exp(labels)
    else:
        y_true = labels

    y_pred = np.exp(np.clip(preds, -3.0, 9.0))
    y_true = np.maximum(y_true, 1e-5)
    y_pred = np.maximum(y_pred, 1e-5)

    num = np.abs(y_pred - y_true)
    den = (y_true + y_pred) / 2.0
    score = float(np.mean(num / den) * 100.0)

    return "smape", score, False


# ---------------------------------------------------------------------------
# XGBoost Custom Objective & Evaluation Callbacks
# ---------------------------------------------------------------------------

def xgb_smape_objective(preds: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
    """XGBoost custom objective for direct SMAPE minimization in log-space.

    Args:
        preds: Model output predictions z = ln(price).
        dtrain: xgboost.DMatrix containing labels.

    Returns:
        (grad, hess) arrays.
    """
    labels = dtrain.get_label()
    if np.median(labels) < 15.0 and (labels > 0).all():
        y_true = np.exp(labels)
    else:
        y_true = labels

    grad, hess = compute_smape_grad_hess(preds, y_true)
    return grad, hess


def xgb_smape_eval(preds: np.ndarray, dtrain) -> Tuple[str, float]:
    """XGBoost custom evaluation metric computing exact SMAPE (%).

    Args:
        preds: Model output predictions z = ln(price).
        dtrain: xgboost.DMatrix.

    Returns:
        ('smape', smape_score).
    """
    labels = dtrain.get_label()
    if np.median(labels) < 15.0 and (labels > 0).all():
        y_true = np.exp(labels)
    else:
        y_true = labels

    y_pred = np.exp(np.clip(preds, -3.0, 9.0))
    y_true = np.maximum(y_true, 1e-5)
    y_pred = np.maximum(y_pred, 1e-5)

    num = np.abs(y_pred - y_true)
    den = (y_true + y_pred) / 2.0
    score = float(np.mean(num / den) * 100.0)

    return "smape", score
