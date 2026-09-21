"""Tests for Custom SMAPE Objective and Evaluation Metrics."""
import numpy as np
import pytest

from src.objectives import (
    compute_smape_grad_hess,
    lgb_smape_objective,
    lgb_smape_eval,
)
from src.metrics import smape


def test_smape_gradient_sign():
    """Verify gradient sign aligns with directional error."""
    y_true = np.array([10.0, 50.0, 100.0])
    
    # Overpredicted: z = ln(20) > ln(10) -> grad should be > 0 (push z down)
    z_over = np.log(np.array([20.0, 100.0, 200.0]))
    grad_over, hess_over = compute_smape_grad_hess(z_over, y_true)
    assert (grad_over > 0).all(), "Overprediction must produce positive gradient"
    assert (hess_over > 0).all(), "Hessian must be strictly positive"

    # Underpredicted: z = ln(5) < ln(10) -> grad should be < 0 (push z up)
    z_under = np.log(np.array([5.0, 25.0, 50.0]))
    grad_under, hess_under = compute_smape_grad_hess(z_under, y_true)
    assert (grad_under < 0).all(), "Underprediction must produce negative gradient"
    assert (hess_under > 0).all(), "Hessian must be strictly positive"


def test_smape_gradient_numerical_finite_difference():
    """Verify analytical gradient matches numerical central difference derivative."""
    y_true = np.array([15.0, 30.0, 85.0, 250.0])
    z_pred = np.log(np.array([12.0, 45.0, 70.0, 300.0]))
    
    analytical_grad, _ = compute_smape_grad_hess(z_pred, y_true, eps=1e-5)
    
    # Numerical central difference: (L(z + delta) - L(z - delta)) / (2 * delta)
    delta = 1e-5
    def loss_func(z):
        y_hat = np.exp(z)
        return 2.0 * np.abs(y_hat - y_true) / (y_hat + y_true)
    
    numerical_grad = (loss_func(z_pred + delta) - loss_func(z_pred - delta)) / (2.0 * delta)
    
    np.testing.assert_allclose(analytical_grad, numerical_grad, rtol=1e-3, atol=1e-4)


def test_lgb_custom_objective_training():
    """Verify LightGBM can train with the custom SMAPE objective without error."""
    import lightgbm as lgb
    
    np.random.seed(42)
    X = np.random.randn(200, 10)
    y_price = np.random.uniform(5.0, 100.0, size=200)
    y_log = np.log(y_price)
    
    # 1. Test scikit-learn LGBMRegressor API
    reg = lgb.LGBMRegressor(
        objective=lgb_smape_objective,
        n_estimators=20,
        learning_rate=0.1,
        num_leaves=15,
        verbosity=-1,
    )
    reg.fit(X[:150], y_log[:150])
    preds_sk = np.exp(reg.predict(X[150:]))
    score_sk = smape(y_price[150:], preds_sk)
    assert 0 < score_sk < 200, f"Expected valid SMAPE score, got {score_sk}"

    # 2. Test low-level lgb.train API
    train_data = lgb.Dataset(X[:150], label=y_log[:150])
    val_data = lgb.Dataset(X[150:], label=y_log[150:], reference=train_data)
    
    params = {
        "objective": lgb_smape_objective,
        "learning_rate": 0.1,
        "num_leaves": 15,
        "verbosity": -1,
    }
    
    bst = lgb.train(
        params,
        train_data,
        num_boost_round=20,
        feval=lgb_smape_eval,
        valid_sets=[val_data],
        callbacks=[lgb.log_evaluation(0)],
    )
    
    preds_train = np.exp(bst.predict(X[150:]))
    score_train = smape(y_price[150:], preds_train)
    assert 0 < score_train < 200, f"Expected valid SMAPE score, got {score_train}"


def test_hessian_strictly_positive_and_floored():
    """Verify that hessians never drop below min_hess floor even on extreme outliers."""
    # Extreme outlier cases: massive overprediction and massive underprediction
    z_preds = np.array([-3.0, 0.0, 2.0, 5.0, 9.0])
    y_trues = np.array([1000.0, 0.01, 15.0, 0.1, 5000.0])
    
    grad, hess = compute_smape_grad_hess(z_preds, y_trues, min_hess=0.05)
    
    assert (hess >= 0.05).all(), f"Hessian dropped below min_hess floor: {hess}"
    assert not np.isnan(grad).any()
    assert not np.isnan(hess).any()
    assert not np.isinf(grad).any()
    assert not np.isinf(hess).any()


def test_charbonnier_smoothing_continuity():
    """Verify smooth transition of gradient around zero error without step discontinuity."""
    y_val = 20.0
    z_exact = np.log(y_val)
    
    # Deltas approaching zero from both sides
    deltas = np.array([-1e-2, -1e-4, -1e-6, 0.0, 1e-6, 1e-4, 1e-2])
    z_test = z_exact + deltas
    y_test = np.full_like(z_test, y_val)
    
    grads, _ = compute_smape_grad_hess(z_test, y_test, eps=1e-4)
    
    # At exact equality (delta=0), gradient should be zero
    assert grads[3] == pytest.approx(0.0, abs=1e-6)
    # Signs should be strictly monotonic across the zero crossing
    assert np.all(np.diff(grads) > 0), "Gradients should be strictly monotonic across the minimum"


def test_dual_signature_lgb_and_xgb():
    """Verify dual signature support for both LightGBM and XGBoost functions."""
    from src.objectives import xgb_smape_objective, xgb_smape_eval

    # Mock DMatrix / Dataset object with get_label()
    class MockDataContainer:
        def __init__(self, label):
            self._label = label
        def get_label(self):
            return self._label

    y_log = np.log(np.array([10.0, 25.0, 50.0]))
    z_preds = np.log(np.array([12.0, 20.0, 60.0]))
    container = MockDataContainer(y_log)

    # 1. Test Dataset/DMatrix API: (preds, data)
    g1_lgb, h1_lgb = lgb_smape_objective(z_preds, container)
    name1_lgb, score1_lgb, _ = lgb_smape_eval(z_preds, container)
    assert len(g1_lgb) == 3 and len(h1_lgb) == 3
    assert name1_lgb == "smape" and score1_lgb > 0

    g1_xgb, h1_xgb = xgb_smape_objective(z_preds, container)
    name1_xgb, score1_xgb = xgb_smape_eval(z_preds, container)
    assert len(g1_xgb) == 3 and len(h1_xgb) == 3
    assert name1_xgb == "smape" and score1_xgb > 0

    # 2. Test scikit-learn API: (y_true, y_pred)
    g2_lgb, h2_lgb = lgb_smape_objective(y_log, z_preds)
    name2_lgb, score2_lgb, _ = lgb_smape_eval(y_log, z_preds)
    np.testing.assert_allclose(g1_lgb, g2_lgb)
    np.testing.assert_allclose(h1_lgb, h2_lgb)
    assert score1_lgb == pytest.approx(score2_lgb)

    g2_xgb, h2_xgb = xgb_smape_objective(y_log, z_preds)
    name2_xgb, score2_xgb = xgb_smape_eval(y_log, z_preds)
    np.testing.assert_allclose(g1_xgb, g2_xgb)
    np.testing.assert_allclose(h1_xgb, h2_xgb)
    assert score1_xgb == pytest.approx(score2_xgb)


def test_extreme_clamping_prevents_overflow():
    """Verify extreme inputs (e.g. z = 100 or -100) are clamped without inf/nan."""
    extreme_z = np.array([-100.0, -50.0, 50.0, 100.0])
    y_true = np.array([1.0, 10.0, 50.0, 500.0])

    grad, hess = compute_smape_grad_hess(extreme_z, y_true)
    assert not np.isnan(grad).any()
    assert not np.isnan(hess).any()
    assert not np.isinf(grad).any()
    assert not np.isinf(hess).any()
