# Project Context: Amazon ML Challenge 2025 (Smart Product Pricing)

## 1. Challenge Overview
- **Goal:** Predict the price of products on Amazon using product metadata (`catalog_content`) and image links (`image_link`).
- **Target Variable:** `price` (positive float, strictly > 0).
- **Evaluation Metric:** Symmetric Mean Absolute Percentage Error (**SMAPE**).
  $$\text{SMAPE} = \frac{100\%}{n} \sum_{i=1}^n \frac{|\hat{y}_i - y_i|}{(|y_i| + |\hat{y}_i|) / 2}$$
  - Range: 0% to 200% (lower is better).
  - First-order approximation in log-space: $\frac{|\hat{y} - y|}{(y + \hat{y})/2} \approx |\ln \hat{y} - \ln y|$.
  - Direct implication: Minimizing MAE or Huber / custom SMAPE objective on $\ln(\text{price})$ closely minimizes SMAPE.

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

## 3. Critical Rules & Constraints
1. **Zero External Price Lookups:** STRICTLY PROHIBITED to scrape Amazon, query product pricing APIs, or lookup prices externally. Disqualification applies.
2. **Model Constraints:** Open-weight models must be Apache 2.0 or MIT licensed, maximum 8 Billion parameters.
3. **Valid Output:** All prices must be positive floats (> 0.0), no missing sample IDs.

## 4. Environment & Hardware Specs
- **Hardware:** Apple Silicon (arm64, 8 CPU cores, 8 GB Unified Memory).
- **Virtual Environment:** `./venv` (Python 3.13)
- **Key Installed Packages:** `scikit-learn 1.9.1`, `lightgbm 4.7.0`, `pandas 3.0.6`, `numpy 2.5.3`, `scipy 1.18.1`, `tqdm`.
- **Compute Strategy:**
  - Local: Fast iteration, memory-efficient feature engineering (sparse matrices, batch processing, chunked processing, gradient boosting).
  - High-Memory/GPU (if needed for vision/LLMs): Modular scripts ready for Kaggle/Colab T4/A100.

## 5. Current Baselines & Milestones
- **Baseline 1 (Ridge Regression on Log-Price):**
  - Features: Basic TF-IDF (10k/15k n-grams) + regex extraction (`value`, `unit`, `pack_qty`).
  - Validation SMAPE: **~64.17%** (5-Fold CV target).
- **Next Targets:**
  - Milestone 1 (Advanced Feature Engineering + LightGBM/CatBoost): Target SMAPE < 50%
  - Milestone 2 (Text Embeddings + Fine-tuned Loss): Target SMAPE < 42%
  - Milestone 3 (Multimodal Vision Features + Ensembling): Target SMAPE < 35%

## 6. Directory Structure
```
Amazon-Ml-Prep/
├── AGENTS.md                  # Project rules and context (this file)
├── Documentation_template.md  # Official submission documentation template
├── README.md                  # Problem description and challenge guidelines
├── sample_code.py             # Starter submission generator script
├── .gitignore                 # Ignore environments, cache, large checkpoints
├── dataset/
│   ├── train.csv              # 75,000 labeled training records
│   ├── test.csv               # 75,000 test records
│   ├── sample_test.csv        # Sample input
│   ├── sample_test_out.csv    # Sample output format
│   └── test_out.csv           # Current generated predictions
├── src/
│   ├── utils.py               # Image download utilities
│   ├── eda.py                 # Initial data inspection
│   ├── test_baseline.py       # Ridge 5-fold CV baseline script
│   └── download_data.py       # Kagglehub data downloader helper
├── notebooks/                 # Exploratory and prototyping notebooks
└── venv/                      # Local Python environment
```

## 7. Key Feature Engineering Levers
1. **Catalog Parsing:**
   - Extract `Item Name`, `Value`, `Unit`, `Bullet Point 1..5`, `Product Description`.
   - Unit Normalization: Standardize synonyms (e.g., `fl oz`, `fluid ounce`, `fl. oz` -> `fl_oz`; `ounce`, `oz` -> `oz`; `lb`, `pound` -> `lb`).
   - Pack Multiplier: Robust regex parsing for `pack of N`, `N count`, `case of N`, `box of N`.
   - Derived Ratios: Unit price proxy, total quantity proxy (`value * pack_qty`).
2. **Text Representation:**
   - Word + Char n-gram TF-IDF on Item Name vs Product Description.
   - Dense semantic text embeddings (e.g. MiniLM / BGE).
   - Brand Extraction: Extract leading brand tokens from `Item Name` and target-encode high-frequency brands.
3. **Vision Features (Optional / Multimodal):**
   - Pretrained ResNet/EfficientNet/CLIP embeddings from downloaded images.
4. **Modeling & Objectives:**
   - Custom Objective: Custom gradient/hessian for SMAPE or Smooth L1 / Huber on `log(price)`.
   - GBDT (LightGBM / CatBoost) + Linear (Ridge) stacking / blending.
