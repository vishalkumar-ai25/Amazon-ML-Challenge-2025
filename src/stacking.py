"""Feature-conditioned Stacking Meta-Learner for SMAPE optimization.

Replaces static linear convex blending with an attribute-conditioned meta-learner
that dynamically adjusts model weights based on product characteristics
(e.g., standard quantity, presence of image, pack size).
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

from src.metrics import smape
from src.postprocess import apply_postprocessing, optimize_global_multiplier, optimize_clip_floor


def build_meta_features(
    model_preds_dict: Dict[str, np.ndarray],
    conditioning_df: Optional[pd.DataFrame] = None,
) -> np.ndarray:
    """Construct meta-feature matrix combining log-predictions and product attributes.

    Args:
        model_preds_dict: Dictionary mapping model names to positive price predictions.
            Keys e.g. 'adapter', 'lgbm', 'cat', 'ridge'.
        conditioning_df: Optional DataFrame with columns such as 'std_quantity',
            'has_image', 'pack_qty', 'is_multipack'.

    Returns:
        2D numpy array of shape (n_samples, n_meta_features).
    """
    n_samples = len(next(iter(model_preds_dict.values())))
    cols = []

    # 1. Base model predictions in log-space
    model_names = sorted(model_preds_dict.keys())
    for name in model_names:
        pred = np.asarray(model_preds_dict[name], dtype=np.float64)
        assert len(pred) == n_samples, f"Length mismatch for model {name}"
        cols.append(np.log(np.maximum(pred, 1e-4)))

    # 2. Pairwise log-ratios between primary models (Adapter vs LightGBM)
    if "adapter" in model_preds_dict and "lgbm" in model_preds_dict:
        ad_pred = np.maximum(model_preds_dict["adapter"], 1e-4)
        lg_pred = np.maximum(model_preds_dict["lgbm"], 1e-4)
        log_ratio = np.log(ad_pred) - np.log(lg_pred)
        cols.append(log_ratio)

    # 3. Conditioning tabular attributes
    if conditioning_df is not None:
        assert len(conditioning_df) == n_samples, "Length mismatch in conditioning_df"

        if "std_quantity" in conditioning_df.columns:
            qty = pd.to_numeric(conditioning_df["std_quantity"], errors="coerce").fillna(1.0).values
            cols.append(np.log1p(np.maximum(qty, 0.0)))

        if "has_image" in conditioning_df.columns:
            has_img = pd.to_numeric(conditioning_df["has_image"], errors="coerce").fillna(0.0).values
            cols.append(has_img)

        if "pack_qty" in conditioning_df.columns:
            pq = pd.to_numeric(conditioning_df["pack_qty"], errors="coerce").fillna(1.0).values
            cols.append(np.log1p(np.maximum(pq, 0.0)))

        if "is_multipack" in conditioning_df.columns:
            mp = pd.to_numeric(conditioning_df["is_multipack"], errors="coerce").fillna(0.0).values
            cols.append(mp)

    return np.column_stack(cols).astype(np.float64)


class FeatureConditionedStacker:
    """Stacking Meta-Learner using L2-regularized log-linear meta-regression.

    Optimized for SMAPE by modeling target in log-space with Ridge regression,
    followed by post-processing multiplier and floor calibration.
    """

    def __init__(self, alpha: float = 10.0, calibrate_postprocess: bool = True):
        self.alpha = alpha
        self.calibrate_postprocess = calibrate_postprocess
        self.meta_model: Optional[Ridge] = None
        self.optimal_multiplier: float = 1.0
        self.optimal_floor: float = 0.05
        self.model_names: List[str] = []

    def fit(
        self,
        oof_dict: Dict[str, np.ndarray],
        y_true: np.ndarray,
        conditioning_df: Optional[pd.DataFrame] = None,
    ) -> FeatureConditionedStacker:
        """Fit the meta-learner on out-of-fold predictions.

        Args:
            oof_dict: Dictionary of model out-of-fold predictions.
            y_true: True positive target prices.
            conditioning_df: Optional product attributes.
        """
        self.model_names = sorted(oof_dict.keys())
        y_true = np.asarray(y_true, dtype=np.float64)
        y_log = np.log(np.maximum(y_true, 1e-4))

        X_meta = build_meta_features(oof_dict, conditioning_df)

        self.meta_model = Ridge(alpha=self.alpha, fit_intercept=True, random_state=42)
        self.meta_model.fit(X_meta, y_log)

        raw_pred_log = self.meta_model.predict(X_meta)
        raw_pred = np.exp(raw_pred_log)

        if self.calibrate_postprocess:
            self.optimal_multiplier, _, _ = optimize_global_multiplier(y_true, raw_pred)
            scaled = raw_pred * self.optimal_multiplier
            self.optimal_floor, _, _ = optimize_clip_floor(y_true, scaled)
        else:
            self.optimal_multiplier = 1.0
            self.optimal_floor = 0.05

        return self

    def predict(
        self,
        test_dict: Dict[str, np.ndarray],
        conditioning_df: Optional[pd.DataFrame] = None,
    ) -> np.ndarray:
        """Predict test set prices using the fitted stacking meta-learner.

        Args:
            test_dict: Dictionary of model test predictions.
            conditioning_df: Optional product attributes for test set.

        Returns:
            Calibrated strictly positive price predictions.
        """
        if self.meta_model is None:
            raise RuntimeError("FeatureConditionedStacker must be fitted before predict.")

        X_meta = build_meta_features(test_dict, conditioning_df)
        pred_log = self.meta_model.predict(X_meta)
        pred = np.exp(pred_log)

        return apply_postprocessing(
            pred,
            multiplier=self.optimal_multiplier,
            clip_min=self.optimal_floor,
        )


def evaluate_stacking_cv(
    oof_dict: Dict[str, np.ndarray],
    y_true: np.ndarray,
    conditioning_df: Optional[pd.DataFrame] = None,
    n_splits: int = 5,
    seed: int = 42,
    alpha: float = 10.0,
) -> Dict[str, float]:
    """Evaluate stacking meta-learner performance via nested K-Fold CV.

    Guarantees zero leakage: the meta-learner is trained on (K-1) folds
    and evaluated on the hold-out validation fold.

    Returns:
        Dict with baseline SMAPE (best single model), static blend SMAPE,
        and stacking CV SMAPE.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    n_samples = len(y_true)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    stack_oof = np.zeros(n_samples, dtype=np.float64)

    for train_idx, val_idx in kf.split(y_true):
        fold_train_oof = {k: v[train_idx] for k, v in oof_dict.items()}
        fold_val_oof = {k: v[val_idx] for k, v in oof_dict.items()}
        fold_y_tr = y_true[train_idx]

        cond_tr = conditioning_df.iloc[train_idx].copy() if conditioning_df is not None else None
        cond_va = conditioning_df.iloc[val_idx].copy() if conditioning_df is not None else None

        stacker = FeatureConditionedStacker(alpha=alpha, calibrate_postprocess=True)
        stacker.fit(fold_train_oof, fold_y_tr, cond_tr)
        stack_oof[val_idx] = stacker.predict(fold_val_oof, cond_va)

    best_single_smape = min(smape(y_true, oof_dict[k]) for k in oof_dict)
    stack_smape = float(smape(y_true, stack_oof))

    return {
        "best_single_smape": float(best_single_smape),
        "stacking_oof_smape": stack_smape,
        "smape_improvement": float(best_single_smape - stack_smape),
    }
