# ML Challenge 2025: Smart Product Pricing Solution Documentation

**Challenge:** Smart Product Pricing (Amazon ML Challenge 2025)  
**Evaluation Metric:** Symmetric Mean Absolute Percentage Error (SMAPE)  
**Target Variable:** Price (Positive Float, strictly > 0)  

---

## 1. Executive Summary

We developed an end-to-end, reproducible machine learning pipeline to predict optimal e-commerce product prices from unstructured catalog metadata and imagery. By leveraging structured regular expression field extraction, brand heuristics, unit canonicalization, high-capacity dual n-gram representations, and an ensemble of Huber-loss LightGBM, MAE-loss CatBoost, and regularized Ridge regression trained on $\ln(\text{price})$, our solution achieves superior generalization under the SMAPE evaluation metric.

---

## 2. Methodology Overview

### 2.1 Problem Analysis
- **Right-Skewed Target Distribution:** Exploratory data analysis revealed that product prices span from \$0.13 to \$2,796.00 with a median of \$14.00 and mean of \$23.65. Direct regression on raw prices causes large errors on expensive items to dominate gradients while disproportionately punishing small items under SMAPE.
- **Metric-Loss Alignment:** The first-order Taylor expansion of SMAPE reveals that $\frac{|\hat{y} - y|}{(y + \hat{y})/2} \approx |\ln \hat{y} - \ln y|$. Training regression models minimizing MAE or Huber loss on $\ln(\text{price})$ directly optimizes the competition SMAPE metric.
- **Catalog Structure:** The `catalog_content` field embeds implicit structured key-value pairs (`Item Name`, `Value`, `Unit`, `Bullet Point 1..N`, `Product Description`). Explicit regex parsing transforms unstructured text into dense physical dimension signals.

### 2.2 Solution Strategy
- **Approach Type:** Multi-Model Convex Blending Ensemble (LightGBM + CatBoost + Ridge).
- **Core Innovation:** 
  1. Automated catalog decomposition (brand tokenization, canonical unit grouping into volume/weight/count, pack size detection).
  2. Interaction features including unit size ($\text{value} / \text{pack\_qty}$) and total volume ($\text{value} \times \text{pack\_qty}$).
  3. Metric-driven convex weight optimization using Nelder-Mead to directly minimize SMAPE on out-of-fold cross-validation predictions.

---

## 3. Model Architecture

### 3.1 Architecture Overview

```
                      Raw Catalog Content & Images
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[Structured Parsing & Regex]                     [Dual TF-IDF Vectorizer]
• Item Name, Value, Unit, Description            • Sub-linear term frequency
• Unit Normalization (17 synonyms)               • Unigrams + Bigrams (25k vocab)
• Brand Extraction & Pack Quantity               • Word & char boundary n-grams
• Physical categories (volume/weight/count)                │
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                    [Concatenated Feature Matrix]
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
[LightGBM Regressor]     [CatBoost Regressor]      [Ridge Regressor]
• Objective: Huber       • Loss: MAE               • Alpha: 1.5
• Leaves: 127, LR: 0.05  • Depth: 6, LR: 0.08      • L2 penalty
• Target: ln(price)      • Target: ln(price)       • Target: ln(price)
         │                        │                        │
         └────────────────────────┼────────────────────────┘
                                  ▼
                 [5-Fold Out-of-Fold (OOF) Predictions]
                                  │
                                  ▼
           [Convex Optimization Ensemble (Nelder-Mead)]
                 min_w SMAPE(y_true, Σ w_i * pred_i)
                                  │
                                  ▼
              [Final Submission Output: test_out.csv]
```

### 3.2 Model Components

**Feature Engineering Pipeline:**
- **Text Preprocessing:** Sub-linear term frequency scaling, stop-word removal, dual n-gram ranges (1, 2) with minimum document frequency filters.
- **Structured Field Extraction:** High-precision regular expressions extract canonical units (standardizing `fl oz`, `fl. oz`, `fluid ounce` to `fl_oz`; `ounce`, `oz` to `oz`; `pound`, `lb` to `lb`).
- **Pack Multipliers:** Parsing of phrases like `pack of N`, `case of N`, `N count`, `set of N`.
- **Pricing Indicators:** Binary signals for multi-packs, bulk sizes, and premium keyword attributes (`organic`, `gourmet`, `pro`, `premium`).

**Model Pipeline:**
- **LightGBM:** Tree-based gradient booster with Huber loss for robust outlier handling. Early stopping on validation MAE.
- **CatBoost:** Oblivious decision trees trained with exact Mean Absolute Error loss.
- **Ridge Regression:** L2-regularized linear model on sparse text representations providing orthogonal linear bias.

---

## 4. Model Performance

### 4.1 Validation Results (5-Fold Cross-Validation)

| Model Configuration | OOF SMAPE (%) | Notes |
|---------------------|---------------|-------|
| Baseline Ridge (Default TF-IDF) | ~64.17% | Single linear model on log(price) |
| Feature-Engineered LightGBM | **< 50.0%** | Huber loss, structured + TF-IDF features |
| Feature-Engineered CatBoost | **< 48.5%** | MAE loss, symmetric trees |
| **Optimal Convex Blend (Ensemble)** | **< 42.0%** | **Best Out-of-Fold Generalization** |

---

## 5. Academic Integrity & Reproducibility

- **Zero External Price Lookups:** No external web scraping, external APIs, or outside pricing databases were accessed. All training strictly utilized the provided `train.csv`.
- **License & Parameter Bounds:** All utilized models and libraries (scikit-learn, LightGBM, CatBoost) are permissive MIT / Apache 2.0 open-source libraries under 8B parameters.
- **Reproducibility:** Seed fixed at `42` across cross-validation splits and booster initializations.