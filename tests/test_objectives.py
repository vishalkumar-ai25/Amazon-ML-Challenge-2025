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
