"""Extreme Price-Tier Binary Classifiers for Two-Stage Classification-Gated Regression.

Provides Out-Of-Fold (OOF) and test probability estimation for:
- Budget / Sample Units (P(price <= budget_threshold))
- Bulk / Luxury Units (P(price >= luxury_threshold))

These probability signals and bounded log-odds prevent regression models
from suffering from median shrinkage on Deciles 0 and 9.
"""
from __future__ import annotations

import time
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score, log_loss

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from sklearn.linear_model import LogisticRegression
    HAS_SKLEARN_LR = True
except ImportError:
    HAS_SKLEARN_LR = False


def _safe_logit(probs: np.ndarray, eps: float = 1e-4, clip_bounds: Tuple[float, float] = (-5.0, 5.0)) -> np.ndarray:
    """Compute numerical-stable bounded log-odds."""
    p = np.clip(probs, eps, 1.0 - eps)
    logits = np.log(p / (1.0 - p))
    return np.clip(logits, clip_bounds[0], clip_bounds[1])


def train_extreme_tier_classifiers_cv(
    X_train: sparse.csr_matrix | np.ndarray,
    y_train: np.ndarray,
    X_test: sparse.csr_matrix | np.ndarray,
    cv_splits: List[Tuple[np.ndarray, np.ndarray]],
    budget_threshold: float = 4.0,
    luxury_threshold: float = 50.0,
    n_estimators: int = 160,
    learning_rate: float = 0.08,
    num_leaves: int = 31,
    random_state: int = 42,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """Train leak-free 5-fold Out-Of-Fold binary classifiers for extreme price tiers.

    Args:
        X_train: Feature matrix for training samples (sparse CSR or dense).
        y_train: Continuous price ground-truth array.
        X_test: Feature matrix for test samples.
        cv_splits: Precomputed list of (train_idx, val_idx) tuples.
        budget_threshold: Price upper-bound for budget/sample classifier (e.g. $4.00).
        luxury_threshold: Price lower-bound for bulk/luxury classifier (e.g. $50.00).
        n_estimators: Maximum trees per fold.
        learning_rate: LightGBM learning rate.
        num_leaves: Tree complexity.
        random_state: Random seed.
        verbose: Whether to log fold progress.

    Returns:
        Tuple of:
            - oof_features: (N, 4) float32 array [p_budget, p_luxury, logit_budget, logit_luxury]
            - test_features: (M, 4) float32 array [p_budget, p_luxury, logit_budget, logit_luxury]
            - oof_prob_budget: (N,) float32
            - oof_prob_luxury: (N,) float32
            - metrics: Dict of AUC-ROC and LogLoss for both classifiers
    """
    t0 = time.time()
    n_train = len(y_train)
    n_test = X_test.shape[0]

    # Binary labels
    y_budget = (y_train <= budget_threshold).astype(np.int32)
    y_luxury = (y_train >= luxury_threshold).astype(np.int32)

    pct_budget = float(np.mean(y_budget) * 100.0)
    pct_luxury = float(np.mean(y_luxury) * 100.0)

    if verbose:
        print(f"\n[Extreme Tier Classifiers] Target Distribution:", flush=True)
        print(f"  Budget Tier (price <= ${budget_threshold:.2f}): {np.sum(y_budget):,} samples ({pct_budget:.1f}%)", flush=True)
        print(f"  Luxury Tier (price >= ${luxury_threshold:.2f}): {np.sum(y_luxury):,} samples ({pct_luxury:.1f}%)", flush=True)

    oof_p_budget = np.zeros(n_train, dtype=np.float32)
    oof_p_luxury = np.zeros(n_train, dtype=np.float32)
    test_p_budget = np.zeros(n_test, dtype=np.float32)
    test_p_luxury = np.zeros(n_test, dtype=np.float32)

    n_folds = len(cv_splits)

    warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        yb_tr, yb_va = y_budget[tr_idx], y_budget[va_idx]
        yl_tr, yl_va = y_luxury[tr_idx], y_luxury[va_idx]

        if HAS_LIGHTGBM:
            lgb_params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "learning_rate": learning_rate,
                "num_leaves": num_leaves,
                "n_estimators": n_estimators,
                "subsample": 0.85,
                "colsample_bytree": 0.85,
                "random_state": random_state + fold,
                "n_jobs": -1,
                "verbose": -1,
            }

            # 1. Budget Classifier
            clf_b = lgb.LGBMClassifier(**lgb_params)
            clf_b.fit(X_tr, yb_tr)
            p_va_b = clf_b.predict_proba(X_va)[:, 1]
            p_te_b = clf_b.predict_proba(X_test)[:, 1]

            # 2. Luxury Classifier
            clf_l = lgb.LGBMClassifier(**lgb_params)
            clf_l.fit(X_tr, yl_tr)
            p_va_l = clf_l.predict_proba(X_va)[:, 1]
            p_te_l = clf_l.predict_proba(X_test)[:, 1]

        elif HAS_SKLEARN_LR:
            # Fallback to fast LogisticRegression if LightGBM is unavailable
            clf_b = LogisticRegression(max_iter=200, random_state=random_state + fold)
            clf_b.fit(X_tr, yb_tr)
            p_va_b = clf_b.predict_proba(X_va)[:, 1]
            p_te_b = clf_b.predict_proba(X_test)[:, 1]

            clf_l = LogisticRegression(max_iter=200, random_state=random_state + fold)
            clf_l.fit(X_tr, yl_tr)
            p_va_l = clf_l.predict_proba(X_va)[:, 1]
            p_te_l = clf_l.predict_proba(X_test)[:, 1]

        else:
            # Naive prior fallback
            p_va_b = np.full(len(va_idx), np.mean(yb_tr), dtype=np.float32)
            p_te_b = np.full(n_test, np.mean(yb_tr), dtype=np.float32)
            p_va_l = np.full(len(va_idx), np.mean(yl_tr), dtype=np.float32)
            p_te_l = np.full(n_test, np.mean(yl_tr), dtype=np.float32)

        oof_p_budget[va_idx] = p_va_b.astype(np.float32)
        oof_p_luxury[va_idx] = p_va_l.astype(np.float32)
        test_p_budget += p_te_b.astype(np.float32) / n_folds
        test_p_luxury += p_te_l.astype(np.float32) / n_folds

    # Validation metrics
    try:
        auc_budget = float(roc_auc_score(y_budget, oof_p_budget))
        auc_luxury = float(roc_auc_score(y_luxury, oof_p_luxury))
        ll_budget = float(log_loss(y_budget, np.clip(oof_p_budget, 1e-6, 1.0 - 1e-6)))
        ll_luxury = float(log_loss(y_luxury, np.clip(oof_p_luxury, 1e-6, 1.0 - 1e-6)))
    except Exception:
        auc_budget, auc_luxury = 0.5, 0.5
        ll_budget, ll_luxury = 1.0, 1.0

    metrics = {
        "auc_budget": auc_budget,
        "auc_luxury": auc_luxury,
        "logloss_budget": ll_budget,
        "logloss_luxury": ll_luxury,
        "elapsed_seconds": float(time.time() - t0),
    }

    if verbose:
        print(f"  [Budget Classifier] OOF AUC: {auc_budget:.4f}, LogLoss: {ll_budget:.4f}", flush=True)
        print(f"  [Luxury Classifier] OOF AUC: {auc_luxury:.4f}, LogLoss: {ll_luxury:.4f}", flush=True)
        print(f"  Completed extreme tier classification in {metrics['elapsed_seconds']:.1f}s", flush=True)

    # Compute stable log-odds
    oof_logit_budget = _safe_logit(oof_p_budget)
    oof_logit_luxury = _safe_logit(oof_p_luxury)
    test_logit_budget = _safe_logit(test_p_budget)
    test_logit_luxury = _safe_logit(test_p_luxury)

    # Pack 4-feature matrices [p_budget, p_luxury, logit_budget, logit_luxury]
    oof_features = np.column_stack([oof_p_budget, oof_p_luxury, oof_logit_budget, oof_logit_luxury]).astype(np.float32)
    test_features = np.column_stack([test_p_budget, test_p_luxury, test_logit_budget, test_logit_luxury]).astype(np.float32)

    return oof_features, test_features, oof_p_budget, oof_p_luxury, metrics
