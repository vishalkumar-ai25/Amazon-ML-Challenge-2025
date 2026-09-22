"""REQ-05: SMAPE metric implementation must match the challenge formula.

SMAPE = (1/n) * sum |pred - actual| / ((|actual| + |pred|)/2) * 100%
Worked example from the statement: actual=100, pred=120 -> 18.18%.
Bounded in [0, 200].
"""
from __future__ import annotations

import numpy as np
import pytest

from src.error_analysis import smape_per_sample
from src.metrics import smape
from src.objectives import compute_smape_grad_hess, lgb_smape_eval, xgb_smape_eval

torch = pytest.importorskip("torch", reason="torch not installed")
from src.adapter import DifferentiableSMAPELoss  # noqa: E402


class TestFormula:
    def test_statement_worked_example(self):
        assert smape([100.0], [120.0]) == pytest.approx(18.1818, abs=1e-3)

    def test_perfect_prediction_is_zero(self):
        assert smape([1.0, 50.0, 2796.0], [1.0, 50.0, 2796.0]) == 0.0

    def test_symmetric_in_arguments(self):
        a = np.array([3.0, 10.0, 200.0])
        b = np.array([4.0, 12.0, 100.0])
        assert smape(a, b) == pytest.approx(smape(b, a))

    def test_matches_reference_vectorised_formula(self):
        rng = np.random.default_rng(0)
        y = rng.uniform(0.5, 500, 1000)
        p = y * rng.uniform(0.3, 3.0, 1000)
        ref = np.mean(np.abs(p - y) / ((np.abs(y) + np.abs(p)) / 2)) * 100
        assert smape(y, p) == pytest.approx(ref, rel=1e-9)

    def test_percentage_scale_not_fraction(self):
        assert 10 < smape([100.0], [120.0]) < 100


class TestBounds:
    def test_upper_bound_200_when_prediction_is_far_off(self):
        assert smape([1.0], [1e9]) == pytest.approx(200.0, abs=1e-3)

    def test_never_exceeds_200_or_below_zero(self):
        rng = np.random.default_rng(1)
        y = rng.uniform(0.01, 3000, 5000)
        p = rng.uniform(0.01, 3000, 5000)
        s = smape(y, p)
        assert 0.0 <= s <= 200.0

    def test_scale_invariance(self):
        y = np.array([1.0, 2.0, 3.0])
        p = np.array([1.5, 1.0, 4.0])
        assert smape(y, p) == pytest.approx(smape(y * 1000, p * 1000))

    def test_zero_prediction_does_not_produce_nan(self):
        s = smape([10.0], [0.0])
        assert np.isfinite(s)
        assert s == pytest.approx(200.0, abs=1e-2)

    def test_both_zero_does_not_divide_by_zero(self):
        assert np.isfinite(smape([0.0], [0.0]))


class TestInputValidation:
    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            smape([1.0, 2.0], [1.0])

    def test_accepts_lists_tuples_and_arrays(self):
        assert smape((1.0, 2.0), [1.0, 2.0]) == 0.0
        assert smape(np.array([1.0]), np.array([1.0])) == 0.0

    def test_returns_python_float(self):
        assert isinstance(smape([1.0], [2.0]), float)

    def test_empty_input_does_not_return_negative(self):
        s = smape([], [])
        assert np.isnan(s) or s >= 0.0


class TestConsistencyAcrossImplementations:
    """The repo has several SMAPE variants (metric, objective eval, torch loss, error analysis).
    They must all agree with the canonical metric."""

    def test_per_sample_mean_equals_metric(self):
        y = np.array([10.0, 20.0, 100.0])
        p = np.array([12.0, 15.0, 130.0])
        assert smape_per_sample(y, p).mean() == pytest.approx(smape(y, p), abs=1e-6)

    def test_lgb_eval_on_log_labels_matches_metric(self):
        y = np.array([10.0, 20.0, 100.0])
        p = np.array([12.0, 15.0, 130.0])
        name, score, higher_is_better = lgb_smape_eval(np.log(y), np.log(p))
        assert name == "smape"
        assert higher_is_better is False
        assert score == pytest.approx(smape(y, p), rel=1e-6)

    def test_xgb_eval_matches_metric(self):
        y = np.array([10.0, 20.0, 100.0])
        p = np.array([12.0, 15.0, 130.0])
        out = xgb_smape_eval(np.log(p), np.log(y))
        # signature may be (preds, dtrain) or (name, score); accept both return shapes
        score = out[1] if isinstance(out, tuple) else out
        assert score == pytest.approx(smape(y, p), rel=1e-3)

    def test_torch_loss_matches_metric_in_price_space(self):
        y = torch.tensor([10.0, 20.0, 100.0])
        p = torch.tensor([12.0, 15.0, 130.0])
        loss = DifferentiableSMAPELoss(predict_in_log=False, eps=0.0)(p, y)
        assert float(loss) == pytest.approx(smape(y.numpy(), p.numpy()), rel=1e-5)

    def test_torch_loss_log_space_is_differentiable(self):
        y = torch.tensor([10.0, 20.0])
        z = torch.log(torch.tensor([12.0, 15.0]), ).requires_grad_(True)
        loss = DifferentiableSMAPELoss(predict_in_log=True)(z, y)
        loss.backward()
        assert z.grad is not None and torch.isfinite(z.grad).all()

    def test_objective_gradient_sign_and_hessian_positivity(self):
        y = np.array([10.0, 10.0, 10.0])
        z = np.log(np.array([5.0, 10.0, 20.0]))
        g, h = compute_smape_grad_hess(z, y)
        assert g[0] < 0 and g[2] > 0            # under-prediction pushes up, over pushes down
        assert abs(g[1]) < 1e-3                 # near-zero gradient at the optimum
        assert (h > 0).all() and np.isfinite(g).all()
