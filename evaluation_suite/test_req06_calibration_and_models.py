"""REQ-14..16: Post-processing calibration, stacking and the neural adapter must
always emit strictly positive, finite prices (the challenge's hard output constraint)
and must not degrade SMAPE on the data they are fitted on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.metrics import smape
from src.postprocess import (
    align_test_distribution,
    apply_postprocessing,
    apply_power_law_calibration,
    optimize_clip_floor,
    optimize_global_multiplier,
    optimize_power_law_calibration,
)
from src.stacking import FeatureConditionedStacker, build_meta_features


@pytest.fixture(scope="module")
def oof_data():
    rng = np.random.default_rng(11)
    y = np.exp(rng.normal(2.6, 1.0, 600))
    pred = y * rng.uniform(0.6, 1.5, 600) * 1.08  # biased upward on purpose
    return y, pred


class TestScalarCalibration:
    def test_multiplier_does_not_worsen_smape(self, oof_data):
        y, p = oof_data
        alpha, base, cal = optimize_global_multiplier(y, p)
        assert 0.85 <= alpha <= 1.15
        assert cal <= base + 1e-9
        assert cal == pytest.approx(smape(y, p * alpha), rel=1e-6)

    def test_clip_floor_within_range(self, oof_data):
        y, p = oof_data
        floor, base, cal = optimize_clip_floor(y, p)
        assert 0.05 <= floor <= 1.5
        assert cal <= base + 1e-9

    def test_apply_postprocessing_positive_and_finite(self):
        out = apply_postprocessing(np.array([-3.0, 0.0, 1e-9, 5.0, 1e6]), multiplier=0.95, clip_min=0.05)
        assert (out > 0).all() and np.isfinite(out).all()
        assert out[0] == 0.05 and out[3] == pytest.approx(4.75)

    def test_apply_postprocessing_rejects_nan_input(self):
        with pytest.raises(AssertionError):
            apply_postprocessing(np.array([1.0, np.nan]))


class TestPowerLawCalibration:
    def test_identity_parameters_are_noop(self):
        p = np.array([0.5, 10.0, 250.0])
        np.testing.assert_allclose(apply_power_law_calibration(p, a=1.0, b=0.0), p)

    def test_preserves_rank_order(self):
        p = np.sort(np.random.default_rng(0).uniform(0.1, 1000, 100))
        out = apply_power_law_calibration(p, a=1.3, b=-0.4)
        assert (np.diff(out) >= 0).all()

    def test_exponent_is_clamped_to_prevent_inversion(self):
        p = np.array([1.0, 10.0, 100.0])
        out = apply_power_law_calibration(p, a=-2.0, b=0.0)
        assert (np.diff(out) >= 0).all()  # a<0 would invert ranking; must be clamped

    def test_zero_and_negative_inputs_are_floored(self):
        out = apply_power_law_calibration(np.array([0.0, -1.0]), a=1.0, b=0.0, clip_min=0.05)
        assert (out >= 0.05).all()

    def test_optimizer_does_not_worsen_smape(self, oof_data):
        y, p = oof_data
        a, b, base, cal = optimize_power_law_calibration(y, p)
        assert 0.7 <= a <= 1.4 and -1.5 <= b <= 1.5
        assert cal <= base + 1e-9

    def test_distribution_alignment_is_rank_preserving_and_positive(self, oof_data):
        y, p = oof_data
        aligned = align_test_distribution(p, y, blend_weight=0.15)
        assert (aligned > 0).all()
        assert np.array_equal(np.argsort(p), np.argsort(aligned))

    def test_alignment_zero_weight_is_identity(self, oof_data):
        y, p = oof_data
        np.testing.assert_allclose(align_test_distribution(p, y, blend_weight=0.0), np.maximum(p, 0.05))


class TestStacking:
    def test_meta_features_shape_and_finiteness(self):
        preds = {"adapter": np.array([1.0, 2.0, 0.0]), "lgbm": np.array([1.5, 2.5, 3.5])}
        cond = pd.DataFrame({"std_quantity": [1, np.nan, 3], "has_image": [1, 0, 1], "pack_qty": [1, 2, -1], "is_multipack": [0, 1, 0]})
        X = build_meta_features(preds, cond)
        assert X.shape == (3, 2 + 1 + 4)
        assert np.isfinite(X).all()

    def test_meta_features_length_mismatch_is_rejected(self):
        with pytest.raises(AssertionError):
            build_meta_features({"a": np.ones(3), "b": np.ones(2)})

    def test_stacker_predicts_positive_finite(self, oof_data):
        y, p = oof_data
        rng = np.random.default_rng(5)
        oof = {"adapter": p, "lgbm": y * rng.uniform(0.7, 1.3, len(y))}
        st = FeatureConditionedStacker(alpha=10.0).fit(oof, y)
        out = st.predict(oof)
        assert out.shape == y.shape
        assert (out > 0).all() and np.isfinite(out).all()
        assert smape(y, out) <= smape(y, p) + 1e-9


torch = pytest.importorskip("torch", reason="torch not installed")
from src.adapter import MultimodalPricingAdapter, train_adapter_cv  # noqa: E402


class TestNeuralAdapter:
    def test_forward_shapes_text_only_and_multimodal(self):
        torch.manual_seed(0)
        m = MultimodalPricingAdapter(text_dim=16)
        assert m(torch.randn(4, 16)).shape == (4,)
        m2 = MultimodalPricingAdapter(text_dim=16, vision_dim=8, tabular_dim=3)
        assert m2(torch.randn(4, 16), torch.randn(4, 8), torch.randn(4, 3)).shape == (4,)

    def test_zero_vision_vector_is_gated_out(self):
        torch.manual_seed(0)
        m = MultimodalPricingAdapter(text_dim=16, vision_dim=8).eval()
        t = torch.randn(3, 16)
        with torch.no_grad():
            a = m(t, torch.zeros(3, 8))
            b = m(t, torch.zeros(3, 8) + 1e-9)
        torch.testing.assert_close(a, b)

    def test_cv_training_emits_positive_predictions_for_every_test_row(self):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        n, d = 120, 12
        emb = rng.normal(size=(n, d)).astype(np.float32)
        y = np.exp(emb[:, 0] * 0.5 + 2.5)
        test_emb = rng.normal(size=(7, d)).astype(np.float32)
        oof, test_pred, scores = train_adapter_cv(
            emb, y, test_emb, n_folds=2, epochs=2, batch_size=32, device="cpu", verbose=False
        )
        assert oof.shape == (n,) and test_pred.shape == (7,)
        assert (oof > 0).all() and (test_pred > 0).all()
        assert np.isfinite(test_pred).all()
        assert len(scores) == 2 and all(0 <= s <= 200 for s in scores)
