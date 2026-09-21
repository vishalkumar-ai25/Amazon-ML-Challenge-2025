# Project Context: Amazon ML Challenge 2025 (Smart Product Pricing)

## 1. Challenge Overview
- **Goal:** Predict the price of products on Amazon using product metadata (`catalog_content`) and image links (`image_link`).
- **Target Variable:** `price` (positive float, strictly > 0).
- **Evaluation Metric:** Symmetric Mean Absolute Percentage Error (**SMAPE**).
  $$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{|\hat{y}_i - y_i|}{(|y_i| + |\hat{y}_i|) / 2}$$
  - Range: 0% to 200% (lower is better).
  - First-order approximation in log-space: $\frac{|\hat{y} - y|}{(y + \hat{y})/2} \approx |\ln \hat{y} - \ln y|$.
  - Direct implication: Minimizing MAE or Huber loss on $\ln(\text{price})$ closely minimizes SMAPE.

## 2. Dataset Structure
- **Train Set:** `dataset/train.csv` (75,000 samples)
  - Columns: `sample_id`, `catalog_content`, `image_link`, `price`
  - Price summary: Min: $0.13, 25%: $6.80, Median: $14.00, Mean: $23.65, 75%: $28.63, Max: $2796.00 (Heavy right-skew).
- **Test Set:** `dataset/test.csv` (75,000 samples)
  - Columns: `sample_id`, `catalog_content`, `image_link`
  - Public leaderboard: Evaluated on 25,000 test samples.
  - Final evaluation: Evaluated on all 75,000 test samples.
- **Output Submission:** `dataset/test_out.csv`
  - Format: CSV with 2 columns (`sample_id`, `price`), exactly 75,000 rows matching `test.csv` IDs.
  - Current status: Verified & validated 75,000 rows, strictly positive prices, zero nulls.

## 3. Critical Rules & Constraints
1. **Zero External Price Lookups:** STRICTLY PROHIBITED to scrape Amazon, query product pricing APIs, or lookup prices externally. Disqualification applies.
2. **Model Constraints:** Open-weight models must be Apache 2.0 or MIT licensed, maximum 8 Billion parameters.
3. **Valid Output:** All prices must be positive floats (> 0.0), no missing sample IDs.

## 4. Current Baselines & Milestones
- **Initial Baseline (Ridge Regression on Log-Price):**
  - Features: Basic TF-IDF (15k n-grams) + regex extraction (`value`, `unit`, `pack_qty`).
  - Validation SMAPE: **64.17%**
- **Milestone 1 (Advanced Feature Engineering + Multi-Model GBDT Ensemble): COMPLETED**
  - Linear Ridge (35K TF-IDF): **67.92% SMAPE**
  - CatBoost GPU (1500 trees, Depth 6, MAE loss): **52.07% SMAPE**
  - LightGBM (127 leaves, Huber loss, 32 threads): **49.18% SMAPE**
  - **Optimal Convex Blend (Nelder-Mead on SMAPE):** **49.13% SMAPE**
  - Blending weights: 89.4% LightGBM + 10.6% CatBoost + 0.0% Ridge
- **Milestone 2 (Frozen Foundation Embeddings + Neural Pricing Adapter): COMPLETED**
  - Architecture: Frozen BGE-large (1024-dim) + Multimodal Neural Adapter (Differentiable SMAPE Loss) + LightGBM (Huber) + CatBoost GPU (MAE) + Stratified K-Fold.
  - Neural Adapter OOF SMAPE: **50.37%**
  - LightGBM on Physical features: **49.18%**
  - **Optimal 4-Way Convex Blend (Nelder-Mead on SMAPE):** **47.32% SMAPE**!
  - Blending weights: **59.2% LightGBM + 40.8% Neural Adapter** (0% CatBoost, 0% Ridge).
  - Test Submission: Exactly 75,000 positive float prices generated and validated on the remote RTX A4000 GPU server.
- **Next Targets:**
  - Milestone 3 (Multimodal Vision Features via SigLIP/DINOv2 + Qwen2.5-7B Embeddings): Target SMAPE < 42%

## 5. Repository Architecture & Directory Structure
```
Amazon-Ml-Prep/
├── AGENTS.md                  # Project rules and context (this file)
├── Documentation_template.md  # Fully authored official submission documentation
├── README.md                  # Problem description and challenge guidelines
├── sample_code.py             # Verified starter submission generator script
├── train_gpu.log              # Verified log of full 5-fold GPU training run (49.13% SMAPE)
├── run_gpu_foundation.sh      # One-click runner script for remote GPU machine
├── .gitignore                 # Ignore environments, caches, weights
├── configs/
│   └── default.yaml           # Centralized configuration (seed, folds, lgbm, tfidf)
├── dataset/
│   ├── train.csv              # 75,000 labeled training records
│   ├── test.csv               # 75,000 test records
│   ├── sample_test.csv        # Sample input
│   ├── sample_test_out.csv    # Sample output format
│   └── test_out.csv           # Current verified 49.13% SMAPE predictions
├── src/
│   ├── __init__.py
│   ├── data.py                # Schema validation, train/test loading, submission checking
│   ├── features.py            # Structured extraction, physical unit scale conversions, prompt formatting, TF-IDF
│   ├── metrics.py             # Numerically stable SMAPE evaluation metric
│   ├── ensemble.py            # Nelder-Mead convex weight optimizer + Price-Stratified K-Fold generator
│   ├── adapter.py             # Differentiable SMAPE Loss & Multimodal Neural Pricing Adapter
│   ├── extract_embeddings.py  # Frozen Foundation text/vision embedding extraction & caching
│   ├── gpu_train_foundation.py# Complete Foundation + Neural Adapter + GBDT GPU training pipeline
│   ├── gpu_train.py           # GBDT GPU training pipeline (RTX A4000)
│   ├── train.py               # Config-driven 5-fold CV training pipeline (Ridge + LightGBM)
│   ├── kaggle_pipeline.py     # Self-contained pipeline with auto dataset discovery
│   ├── download_images.py     # Concurrent, resilient image downloader with URL auditing
│   ├── utils.py               # Image download utility
│   ├── eda.py                 # Initial data inspection
│   └── test_baseline.py       # Legacy Ridge baseline script
├── tests/                     # Comprehensive test suite (107 tests, 105 passed, 2 skipped on non-torch)
│   ├── test_adapter.py        # Stratified K-fold & Neural Adapter tests (4 tests)
│   ├── test_data.py           # Data loader & schema validation tests (16 tests)
│   ├── test_ensemble.py       # Nelder-Mead blending & constraint tests (2 tests)
│   ├── test_features.py       # Physical conversions, prompt builder, regex, units (57 tests)
│   ├── test_metrics.py        # SMAPE mathematical property tests (9 tests)
│   ├── test_pipeline.py       # LightGBM & Ridge CV pipeline tests (5 tests)
│   ├── test_security_url.py   # SSRF & URL scheme whitelisting tests (12 tests)
│   └── test_submission.py     # Submission validation & constraint tests (2 tests)
├── notebooks/
│   └── kaggle_amazon_ml_solution.ipynb # Standalone Kaggle submission notebook
└── venv/                      # Local Python environment
```

## 6. Key Feature Engineering Levers Implemented
1. **Physical Scale Conversion (Standardized Base Units):**
   - Weights normalized into `total_grams` (`1 lb` = `16 oz` = `453.59g`).
   - Volumes normalized into `total_ml` (`1 fl_oz` = `29.57ml`).
   - Pieces normalized into `total_pieces`.
   - Unified standard quantity: `std_quantity = total_grams + total_ml + total_pieces`.
2. **Foundation Model Prompt Builder:**
   - Converts numbers to words (`12` $\to$ `twelve`) to prevent tokenizer fragmentation.
   - Generates structured prompt: `Product: {name} | Size: {val} {unit} | Multipack: {word} pack | Specifications: {bullets}`.
3. **Multimodal Neural Adapter:**
   - Differentiable SMAPE loss with backpropagation gradients.
   - Text Projection MLP (256) + Tabular Projection MLP (64) + Multimodal Fusion Head (128).
4. **Stratified Price K-Fold CV:**
   - Quantile discretization ensures heavy-tailed price distribution ($0.13 to $2796) is equally represented across all 5 folds.
