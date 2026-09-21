"""Downstream Pricing Adapter & Differentiable SMAPE Loss for Amazon ML Challenge 2025.

Implements lightweight neural projection adapters that map precomputed frozen foundation
embeddings (text and vision) and engineered tabular features into optimal price predictions.
"""
from __future__ import annotations

import os
import time
from typing import Optional, Sequence

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ---------------------------------------------------------------------------
# Differentiable SMAPE Loss
# ---------------------------------------------------------------------------

if HAS_TORCH:

    class DifferentiableSMAPELoss(nn.Module):
        """Numerically stable differentiable Symmetric Mean Absolute Percentage Error (SMAPE) loss.

        Computes:
            SMAPE = 100 * mean( |y_pred - y_true| / ((|y_true| + |y_pred|) / 2 + eps) )

        Can operate either directly on positive prices or in log-space (with predict_in_log=True).
        """

        def __init__(self, eps: float = 1e-4, predict_in_log: bool = True):
            super().__init__()
            self.eps = eps
            self.predict_in_log = predict_in_log

        def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
            """Compute SMAPE loss.

            Args:
                y_pred: Predicted values (log(price) if predict_in_log=True, else price).
                y_true: Ground truth price values (strictly positive).

            Returns:
                Scalar SMAPE loss tensor.
            """
            if self.predict_in_log:
                # Clamp log-predictions to reasonable e-commerce bounds [exp(-2) ≈ 0.13, exp(9) ≈ 8103]
                y_pred_clamped = torch.clamp(y_pred.squeeze(-1), min=-2.5, max=9.0)
                pred_price = torch.exp(y_pred_clamped)
            else:
                pred_price = torch.clamp(y_pred.squeeze(-1), min=self.eps)

            true_price = torch.clamp(y_true.squeeze(-1), min=self.eps)

            abs_diff = torch.abs(pred_price - true_price)
            denominator = (torch.abs(true_price) + torch.abs(pred_price)) / 2.0 + self.eps

            return 100.0 * torch.mean(abs_diff / denominator)


    class MultimodalPricingAdapter(nn.Module):
        """Lightweight neural adapter that fuses frozen text, vision, and tabular embeddings.

        Architecture:
            Text Embeddings (e.g. 1024 / 768 dim)     -> Text Projection MLP (256)
            Vision Embeddings (e.g. 768 / 1152 dim)   -> Gated Vision Projection (128) [Optional]
            Tabular Features (e.g. 25+ dim)           -> Tabular Projection MLP (64) [Optional]
            Cross-Modal Gated Fusion -> LayerNorm -> Fusion MLP (128) -> Output Linear (1) [ln(price)]
        """

        def __init__(
            self,
            text_dim: int,
            vision_dim: Optional[int] = None,
            tabular_dim: Optional[int] = None,
            hidden_dim: int = 256,
            dropout: float = 0.2,
        ):
            super().__init__()
            self.has_vision = vision_dim is not None and vision_dim > 0
            self.has_tabular = tabular_dim is not None and tabular_dim > 0

            # Text projection
            self.text_proj = nn.Sequential(
                nn.Linear(text_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            fusion_input_dim = hidden_dim

            # Optional vision projection with dynamic cross-modal gating
            if self.has_vision:
                v_hidden = hidden_dim // 2
                self.vision_proj = nn.Sequential(
                    nn.Linear(vision_dim, v_hidden),
                    nn.LayerNorm(v_hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                # Cross-modal gate between text representation and vision representation
                self.cross_gate = nn.Sequential(
                    nn.Linear(hidden_dim + v_hidden, v_hidden),
                    nn.Sigmoid(),
                )
                fusion_input_dim += v_hidden

            # Optional tabular projection
            if self.has_tabular:
                t_hidden = hidden_dim // 4
                self.tabular_proj = nn.Sequential(
                    nn.Linear(tabular_dim, t_hidden),
                    nn.LayerNorm(t_hidden),
                    nn.GELU(),
                )
                fusion_input_dim += t_hidden

            # Multimodal fusion head
            self.fusion_head = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim // 2),
                nn.LayerNorm(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout / 2.0),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(
            self,
            text_emb: torch.Tensor,
            vision_emb: Optional[torch.Tensor] = None,
            tabular_feat: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            t_repr = self.text_proj(text_emb)
            reprs = [t_repr]

            if self.has_vision and vision_emb is not None:
                # Auto-detect vision presence mask: 1.0 if image features exist, 0.0 if missing/zero
                v_norm = torch.norm(vision_emb, p=2, dim=-1, keepdim=True)
                v_mask = (v_norm > 1e-4).float()

                v_raw = self.vision_proj(vision_emb)
                # Dynamic cross-modal gating: gates visual features based on multimodal relevance
                gate_input = torch.cat([t_repr, v_raw], dim=-1)
                v_gate = self.cross_gate(gate_input) * v_mask
                v_repr = v_raw * v_gate
                reprs.append(v_repr)

            if self.has_tabular and tabular_feat is not None:
                reprs.append(self.tabular_proj(tabular_feat))

            fused = torch.cat(reprs, dim=-1)
            return self.fusion_head(fused).squeeze(-1)


def train_adapter_cv(
    train_text_emb: np.ndarray,
    y_train: np.ndarray,
    test_text_emb: np.ndarray,
    *,
    train_vision_emb: Optional[np.ndarray] = None,
    test_vision_emb: Optional[np.ndarray] = None,
    train_tabular: Optional[np.ndarray] = None,
    test_tabular: Optional[np.ndarray] = None,
    cv_splits: Optional[list[tuple[np.ndarray, np.ndarray]]] = None,
    n_folds: int = 5,
    epochs: int = 40,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: Optional[str] = None,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train MultimodalPricingAdapter with K-Fold cross-validation on precomputed embeddings.

    Args:
        train_text_emb: (N, D_text) float32 numpy array.
        y_train: (N,) float price array.
        test_text_emb: (M, D_text) float32 numpy array.
        train_vision_emb: Optional (N, D_vision) array.
        test_vision_emb: Optional (M, D_vision) array.
        train_tabular: Optional (N, D_tab) array.
        test_tabular: Optional (M, D_tab) array.
        cv_splits: Precomputed list of (train_idx, val_idx) tuples.
        n_folds: Number of folds (used if cv_splits is None).
        epochs: Max epochs per fold.
        batch_size: Mini-batch size.
        lr: Peak learning rate.
        weight_decay: L2 penalty.
        device: 'cuda', 'mps', or 'cpu'. Auto-detects if None.
        verbose: Whether to log training progress.

    Returns:
        Tuple of (oof_predictions, test_predictions, fold_smape_scores).
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required to train the Neural Pricing Adapter.")

    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    from src.metrics import smape
    from src.ensemble import create_price_stratified_folds

    if cv_splits is None:
        cv_splits = create_price_stratified_folds(y_train, n_folds=n_folds)

    n_samples = len(y_train)
    n_test = len(test_text_emb)

    oof_preds = np.zeros(n_samples, dtype=np.float64)
    test_preds = np.zeros(n_test, dtype=np.float64)
    fold_scores: list[float] = []

    text_dim = train_text_emb.shape[1]
    vision_dim = train_vision_emb.shape[1] if train_vision_emb is not None else None
    tabular_dim = train_tabular.shape[1] if train_tabular is not None else None

    has_vision = vision_dim is not None
    has_tab = tabular_dim is not None

    criterion = DifferentiableSMAPELoss(predict_in_log=True)

    def _predict_test_batched(model: nn.Module, eval_batch_size: int = 1024) -> np.ndarray:
        """Evaluate test set in mini-batches to prevent GPU memory saturation."""
        model.eval()
        preds_list = []
        with torch.no_grad():
            for s_idx in range(0, n_test, eval_batch_size):
                e_idx = min(s_idx + eval_batch_size, n_test)
                b_text = torch.from_numpy(test_text_emb[s_idx:e_idx].astype(np.float32)).to(device)
                b_vis = (
                    torch.from_numpy(test_vision_emb[s_idx:e_idx].astype(np.float32)).to(device)
                    if has_vision and test_vision_emb is not None
                    else None
                )
                b_tab = (
                    torch.from_numpy(test_tabular[s_idx:e_idx].astype(np.float32)).to(device)
                    if has_tab and test_tabular is not None
                    else None
                )
                pred_log = model(b_text, b_vis, b_tab)
                pred_p = torch.exp(torch.clamp(pred_log, min=-2.5, max=9.0)).cpu().numpy()
                preds_list.append(pred_p)
        return np.concatenate(preds_list)

    for fold, (tr_idx, va_idx) in enumerate(cv_splits):
        t0 = time.time()

        # Build datasets
        tr_t = torch.from_numpy(train_text_emb[tr_idx].astype(np.float32))
        va_t = torch.from_numpy(train_text_emb[va_idx].astype(np.float32)).to(device)
        tr_y = torch.from_numpy(y_train[tr_idx].astype(np.float32))
        va_y_true = y_train[va_idx]

        tensors_tr = [tr_t]
        if has_vision:
            tensors_tr.append(torch.from_numpy(train_vision_emb[tr_idx].astype(np.float32)))
        if has_tab:
            tensors_tr.append(torch.from_numpy(train_tabular[tr_idx].astype(np.float32)))
        tensors_tr.append(tr_y)

        train_dataset = TensorDataset(*tensors_tr)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

        # Initialize model
        model = MultimodalPricingAdapter(
            text_dim=text_dim,
            vision_dim=vision_dim,
            tabular_dim=tabular_dim,
            hidden_dim=256,
            dropout=0.2,
        ).to(device)

        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

        best_val_smape = float("inf")
        best_val_preds: Optional[np.ndarray] = None
        patience = 8
        patience_counter = 0

        for epoch in range(epochs):
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                b_text = batch[0].to(device)
                curr_idx = 1
                b_vision = batch[curr_idx].to(device) if has_vision else None
                if has_vision:
                    curr_idx += 1
                b_tab = batch[curr_idx].to(device) if has_tab else None
                if has_tab:
                    curr_idx += 1
                b_y = batch[curr_idx].to(device)

                pred_log = model(b_text, b_vision, b_tab)
                loss = criterion(pred_log, b_y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                va_v = torch.from_numpy(train_vision_emb[va_idx].astype(np.float32)).to(device) if has_vision else None
                va_tab = torch.from_numpy(train_tabular[va_idx].astype(np.float32)).to(device) if has_tab else None

                val_pred_log = model(va_t, va_v, va_tab)
                val_pred_price = torch.exp(torch.clamp(val_pred_log, min=-2.5, max=9.0)).cpu().numpy()

                val_smape_score = smape(va_y_true, val_pred_price)

                if val_smape_score < best_val_smape:
                    best_val_smape = val_smape_score
                    best_val_preds = val_pred_price
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

        oof_preds[va_idx] = best_val_preds
        fold_scores.append(best_val_smape)

        if verbose:
            print(f"  [Adapter] Fold {fold+1}/{len(cv_splits)} SMAPE: {best_val_smape:.2f}% ({time.time() - t0:.1f}s)")

        # Predict test safely in mini-batches
        fold_test_pred = _predict_test_batched(model, eval_batch_size=1024)
        test_preds += fold_test_pred / len(cv_splits)

    return oof_preds, test_preds, fold_scores
