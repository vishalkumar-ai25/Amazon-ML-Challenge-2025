"""Unit tests for feature-conditioned stacking meta-learner."""
import numpy as np
import pandas as pd
import pytest

from src.metrics import smape
from src.stacking import (
    FeatureConditionedStacker,
    build_meta_features,
    evaluate_stacking_cv,
)


@pytest.fixture
def synthetic_multimodal_predictions():
    np.random.seed(42)
    n = 300
    y_true = np.random.uniform(5.0, 100.0, size=n)

    # Base models with different strengths
    adapter_pred = y_true * np.random.normal(1.0, 0.2, size=n)
    lgbm_pred = y_true * np.random.normal(1.0, 0.22, size=n)
    cat_pred = y_true * np.random.normal(1.0, 0.25, size=n)
    ridge_pred = y_true * np.random.normal(1.0, 0.40, size=n)

    oof_dict = {
        "adapter": np.maximum(adapter_pred, 0.5),
        "lgbm": np.maximum(lgbm_pred, 0.5),
        "cat": np.maximum(cat_pred, 0.5),
        "ridge": np.maximum(ridge_pred, 0.5),
    }

    conditioning_df = pd.DataFrame({
        "std_quantity": np.random.choice([1.0, 6.0, 12.0, 24.0], size=n),
        "has_image": np.random.choice([1.0, 0.0], p=[0.95, 0.05], size=n),
        "pack_qty": np.random.choice([1.0, 2.0, 6.0], size=n),
        "is_multipack": np.random.choice([0.0, 1.0], size=n),
    })

    return oof_dict, y_true, conditioning_df


class TestFeatureConditionedStacking:

    def test_build_meta_features_shape_and_values(self, synthetic_multimodal_predictions):
        oof_dict, _, conditioning_df = synthetic_multimodal_predictions
        X_meta = build_meta_features(oof_dict, conditioning_df)

        assert X_meta.ndim == 2
        assert X_meta.shape[0] == 300
        # 4 models + 1 ratio (adapter vs lgbm) + 4 conditioning cols = 9
        assert X_meta.shape[1] == 9
        assert not np.isnan(X_meta).any()

    def test_build_meta_features_without_conditioning(self, synthetic_multimodal_predictions):
        oof_dict, _, _ = synthetic_multimodal_predictions
        X_meta = build_meta_features(oof_dict, conditioning_df=None)

        assert X_meta.shape[0] == 300
        # 4 models + 1 ratio = 5
        assert X_meta.shape[1] == 5
        assert not np.isnan(X_meta).any()

    def test_stacker_fit_and_predict_constraints(self, synthetic_multimodal_predictions):
        oof_dict, y_true, conditioning_df = synthetic_multimodal_predictions

        stacker = FeatureConditionedStacker(alpha=10.0, calibrate_postprocess=True)
        stacker.fit(oof_dict, y_true, conditioning_df)

        preds = stacker.predict(oof_dict, conditioning_df)

        assert len(preds) == len(y_true)
        assert (preds > 0).all()
        assert not np.isnan(preds).any()
        assert stacker.optimal_multiplier > 0.5

    def test_evaluate_stacking_cv_reduces_smape(self, synthetic_multimodal_predictions):
        oof_dict, y_true, conditioning_df = synthetic_multimodal_predictions

        results = evaluate_stacking_cv(
            oof_dict=oof_dict,
            y_true=y_true,
            conditioning_df=conditioning_df,
            n_splits=5,
            seed=42,
            alpha=10.0,
        )

        assert "best_single_smape" in results
        assert "stacking_oof_smape" in results
        assert "smape_improvement" in results
        assert results["stacking_oof_smape"] <= results["best_single_smape"]
