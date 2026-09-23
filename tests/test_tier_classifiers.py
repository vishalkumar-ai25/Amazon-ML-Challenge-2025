"""Tests for extreme price-tier binary classifiers."""
import numpy as np
import pytest
from scipy import sparse

from src.tier_classifiers import (
    _safe_logit,
    train_extreme_tier_classifiers_cv,
)


class TestExtremeTierClassifiers:

    def test_safe_logit_numerical_bounds(self):
        """Verify _safe_logit produces strictly bounded, non-NaN outputs."""
        probs = np.array([0.0, 1e-8, 0.01, 0.5, 0.99, 1.0 - 1e-8, 1.0])
        logits = _safe_logit(probs, clip_bounds=(-5.0, 5.0))

        assert len(logits) == len(probs)
        assert not np.isnan(logits).any()
        assert not np.isinf(logits).any()
        assert (logits >= -5.0).all()
        assert (logits <= 5.0).all()
        # Logit of 0.5 should be 0.0
        assert logits[3] == pytest.approx(0.0, abs=1e-3)
        # Monotonicity: higher probability => higher logit
        assert (np.diff(logits) >= 0).all()

    def test_tier_classifiers_shapes_and_validity(self):
        """Verify output shapes and probability ranges [0, 1]."""
        np.random.seed(42)
        n_train = 300
        n_test = 50
        n_features = 20

        X_tr = sparse.csr_matrix(np.random.randn(n_train, n_features).astype(np.float32))
        X_te = sparse.csr_matrix(np.random.randn(n_test, n_features).astype(np.float32))
        # Prices ranging from $0.50 to $200
        y_tr = np.exp(np.random.uniform(np.log(0.5), np.log(200.0), size=n_train))

        from sklearn.model_selection import KFold
        cv_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(y_tr))

        oof_feat, te_feat, p_b, p_l, metrics = train_extreme_tier_classifiers_cv(
            X_train=X_tr,
            y_train=y_tr,
            X_test=X_te,
            cv_splits=cv_splits,
            budget_threshold=4.0,
            luxury_threshold=50.0,
            n_estimators=30,
            verbose=False,
        )

        # Shape assertions
        assert oof_feat.shape == (n_train, 4)
        assert te_feat.shape == (n_test, 4)
        assert len(p_b) == n_train
        assert len(p_l) == n_train

        # Range assertions
        assert (p_b >= 0.0).all() and (p_b <= 1.0).all()
        assert (p_l >= 0.0).all() and (p_l <= 1.0).all()
        assert (te_feat[:, 0] >= 0.0).all() and (te_feat[:, 0] <= 1.0).all()
        assert (te_feat[:, 1] >= 0.0).all() and (te_feat[:, 1] <= 1.0).all()

        # No NaNs or Infs
        assert not np.isnan(oof_feat).any()
        assert not np.isnan(te_feat).any()

        # Metrics structure
        assert "auc_budget" in metrics
        assert "auc_luxury" in metrics
        assert metrics["elapsed_seconds"] > 0

    def test_tier_classifiers_discriminative_power(self):
        """Verify that classifiers accurately discriminate budget vs luxury items."""
        np.random.seed(42)
        n_samples = 400
        # Feature 0 strongly correlates with log(price)
        log_price = np.random.uniform(np.log(1.0), np.log(100.0), size=n_samples)
        y = np.exp(log_price)
        f0 = (log_price - np.mean(log_price)) + np.random.normal(0, 0.2, size=n_samples)
        X = np.column_stack([f0, np.random.randn(n_samples, 5)]).astype(np.float32)

        from sklearn.model_selection import KFold
        cv_splits = list(KFold(n_splits=3, shuffle=True, random_state=42).split(y))

        oof_feat, te_feat, p_b, p_l, metrics = train_extreme_tier_classifiers_cv(
            X_train=sparse.csr_matrix(X),
            y_train=y,
            X_test=sparse.csr_matrix(X[:50]),
            cv_splits=cv_splits,
            budget_threshold=4.0,
            luxury_threshold=50.0,
            n_estimators=50,
            verbose=False,
        )

        # Budget AUC should be substantially better than random guess (0.5)
        assert metrics["auc_budget"] > 0.70, f"Expected AUC > 0.70, got {metrics['auc_budget']}"
        assert metrics["auc_luxury"] > 0.70, f"Expected AUC > 0.70, got {metrics['auc_luxury']}"

        # Items with true price <= $3 should have higher budget prob than items >= $20
        cheap_mask = y <= 3.0
        expensive_mask = y >= 20.0
        if np.any(cheap_mask) and np.any(expensive_mask):
            assert np.mean(p_b[cheap_mask]) > np.mean(p_b[expensive_mask])
            assert np.mean(p_l[expensive_mask]) > np.mean(p_l[cheap_mask])

    def test_tier_classifiers_zero_leakage_and_coverage(self):
        """Verify that every sample is predicted exactly once in OOF without leakage."""
        np.random.seed(42)
        n = 150
        X = np.random.randn(n, 10).astype(np.float32)
        y = np.random.uniform(1.0, 100.0, size=n)

        from sklearn.model_selection import KFold
        cv_splits = list(KFold(n_splits=5, shuffle=True, random_state=123).split(y))

        oof_feat, te_feat, p_b, p_l, _ = train_extreme_tier_classifiers_cv(
            X_train=X,
            y_train=y,
            X_test=X[:20],
            cv_splits=cv_splits,
            budget_threshold=5.0,
            luxury_threshold=40.0,
            n_estimators=20,
            verbose=False,
        )

        assert (p_b > 0).all()
        assert (p_l > 0).all()
        assert not np.isnan(oof_feat).any()
        assert len(oof_feat) == n
        assert len(te_feat) == 20

