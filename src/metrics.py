"""Evaluation metrics for the Amazon ML Challenge 2025.

Competition metric: Symmetric Mean Absolute Percentage Error (SMAPE).
    SMAPE = (1/n) * Σ |predicted - actual| / ((|actual| + |predicted|) / 2) * 100

Bounded between 0% and 200%. Lower is better.
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

# Type alias for array-like inputs
ArrayLike = Union[np.ndarray, Sequence[float], "pd.Series"]

# Minimum price floor to avoid division-by-zero edge cases
_MIN_PRICE = 1e-5


def smape(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    clip_min: float = _MIN_PRICE,
) -> float:
    """Compute Symmetric Mean Absolute Percentage Error (SMAPE).

    Matches the exact competition definition:
        SMAPE = mean(|pred - actual| / ((|actual| + |pred|) / 2)) * 100

    Args:
        y_true: Ground truth prices.
        y_pred: Predicted prices.
        clip_min: Floor for both arrays to avoid division by zero.
            Defaults to 1e-5.

    Returns:
        SMAPE as a percentage (0.0 to 200.0). Lower is better.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"Shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )

    # Clip to small positive to handle zero/negative gracefully
    y_true = np.clip(y_true, clip_min, None)
    y_pred = np.clip(y_pred, clip_min, None)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0

    return float(np.mean(numerator / denominator) * 100.0)
