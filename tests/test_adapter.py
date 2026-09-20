"""Unit tests for Stratified K-Fold and Neural Pricing Adapter module."""
import numpy as np
import pytest

from src.ensemble import create_price_stratified_folds
from src.metrics import smape
from src.adapter import HAS_TORCH


class TestPriceStratifiedFolds:

    def test_stratified_folds_shape_and_splits(self):
        # Generate synthetic skewed prices
        np.random.seed(42)
        y = np.exp(np.random.normal(loc=2.5, scale=1.0, size=500))

        splits = create_price_stratified_folds(y, n_folds=5, n_bins=10, seed=42)
        assert len(splits) == 5

        all_val_indices = []
        for tr_idx, va_idx in splits:
            assert len(tr_idx) + len(va_idx) == 500
            assert len(np.intersect1d(tr_idx, va_idx)) == 0
            all_val_indices.extend(va_idx)

        # Ensure all samples are evaluated exactly once across 5 folds
        assert len(np.unique(all_val_indices)) == 500

    def test_stratified_folds_quantile_balance(self):
        np.random.seed(42)
        y = np.array([1.0] * 100 + [10.0] * 100 + [50.0] * 100 + [500.0] * 100 + [2500.0] * 100)
        splits = create_price_stratified_folds(y, n_folds=5, n_bins=5, seed=42)

        for _, va_idx in splits:
            val_prices = y[va_idx]
            # Each fold should contain cheap items and expensive items
            assert (val_prices < 5.0).sum() > 0
            assert (val_prices > 1000.0).sum() > 0


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural adapter tests")
class TestNeuralAdapterTorch:

    def test_smape_loss_module(self):
        import torch
        from src.adapter import DifferentiableSMAPELoss

        loss_fn = DifferentiableSMAPELoss(predict_in_log=True)
        # log(10.0) ≈ 2.3026
        y_pred = torch.tensor([[np.log(10.0)], [np.log(20.0)]], dtype=torch.float32, requires_grad=True)
        y_true = torch.tensor([[10.0], [20.0]], dtype=torch.float32)

        loss = loss_fn(y_pred, y_true)
        # Zero error should yield near zero loss
        assert float(loss.item()) < 0.1

        # Test gradient flow
        loss.backward()
        assert y_pred.grad is not None

    def test_multimodal_adapter_forward(self):
        import torch
        from src.adapter import MultimodalPricingAdapter

        model = MultimodalPricingAdapter(
            text_dim=128,
            vision_dim=64,
            tabular_dim=10,
            hidden_dim=32,
        )
        model.eval()

        batch_size = 4
        text_emb = torch.randn(batch_size, 128)
        vision_emb = torch.randn(batch_size, 64)
        tabular_feat = torch.randn(batch_size, 10)

        with torch.no_grad():
            output = model(text_emb, vision_emb, tabular_feat)

        assert output.shape == (batch_size,)
        assert not torch.isnan(output).any()
