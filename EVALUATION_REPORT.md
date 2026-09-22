# Evaluation Report — Amazon ML Challenge 2025 (Smart Product Pricing)

Repository: `vishalkumar-ai25/Amazon-ML-Challenge-2025` @ `9e22eb0` (`feat(milestone6): ...`)
Evaluated against: *Smart Product Pricing Challenge* problem statement (input/output schema, positivity, SMAPE, licence/size limits, fair play, documentation, sample-file layout).
Environment: Ubuntu, Python 3.10.12, fresh virtualenv, CPU only, no network access during test execution (all HTTP mocked).
Constraint honoured: **no file under `src/`, `tests/`, `configs/`, `dataset/` or any root script was modified.** Only `evaluation_suite/` and this report were added.

---

## 1. Executive Summary

| Item | Result |
| :--- | :--- |
| **Overall Compliance Score** | **72 %** (14 Met, 5 Partial, 4 Omitted/Broken of 23 requirements; Partial counted at 0.5) |
| **Test Execution Summary (evaluation suite)** | **331 passed, 17 failed, 3 skipped — 351 total** |
| **Test Execution Summary (native `tests/`)** | 129 passed, 13 failed — 142 total (all 13 failures: `dataset/train.csv` / `dataset/test.csv` absent from clone) |
| **Combined run** | 460 passed, 30 failed, 3 skipped — 493 total |
| **Coverage of `src/` + `sample_code.py`** | 42 % (evaluation suite alone), 47 % (evaluation + native) |
| **Production Readiness Verdict** | **CONDITIONAL PASS** |

**Why Conditional Pass and not Pass.** The artefact that is actually submitted — `dataset/test_out.csv` — independently satisfies every hard output constraint that can be checked offline (exact `sample_id,price` header, 75,000 rows, unique integer IDs, strictly positive finite floats, non-degenerate distribution). The metric, URL security, licence/size selection and fair-play posture are all verified clean. However the repository **cannot be rebuilt or re-validated from the clone alone**: the datasets are git-ignored, there is no dependency manifest, the mandated `sample_code.py` does not even compile, exact `sample_id` alignment against `test.csv` is unverifiable, and the submission validator has real holes (accepts `inf`, duplicate IDs, and crashes with `TypeError` on non-numeric prices). The headline SMAPE figures (41.48 %) in `AGENTS.md`/`Documentation_template.md` are **repository claims** that could not be reproduced here (no data, images, embeddings or weights in the clone).

**Build / install ruling.** The first attempt to import the pipeline in a clean environment failed:

```
Traceback (most recent call last):
  File "<string>", line 1, in <module>
  File "/home/ubuntu/Amazon-ML-Challenge-2025/src/train.py", line 15, in <module>
    import lightgbm as lgb
ModuleNotFoundError: No module named 'lightgbm'
```

The repository ships **no** `requirements.txt` / `pyproject.toml` / `setup.py` / `environment.yml`, so there was no install step to fail — dependencies had to be inferred from imports and installed by hand (`numpy pandas scipy scikit-learn lightgbm catboost xgboost torch(cpu) pillow requests pyyaml pytest pytest-cov`). After that, every `src.*` module imports and both suites execute. Because the failure was the *absence* of an installation procedure rather than a failing one, the 0 % rule was **not** applied; instead the missing manifest is scored as an omitted non-functional requirement (REQ-22) and flagged as a critical defect in §4. If the evaluator prefers a strict reading of "fails to install dependencies", the score should be read as 0 %.

---

## 2. Traceability & Feature Matrix

Legend: **Met** — verified by executing code or inspecting the committed artefact; **Partial** — implemented with gaps proven by a failing evaluation test; **Omitted** — absent or non-functional. "Claim" means a statement made by the repository that could not be independently verified offline.

| Requirement ID / Summary | Implementation File/Line | Status | Notes / Deviations |
| :--- | :--- | :--- | :--- |
| REQ-01 Consume input schema `sample_id, catalog_content, image_link` (+ `price` in train) with validation | `src/data.py:13-38` (`load_train`), `src/data.py:40-65` (`load_test`) | Met | Missing-file → `FileNotFoundError`; missing column → `ValueError`. 8 evaluation tests pass. |
| REQ-02 Output CSV with exactly two columns `sample_id, price` | `src/train.py:216-240` (`save_submission`), `src/gpu_train_foundation.py:534-544`, `src/kaggle_pipeline.py:213-221`; artefact `dataset/test_out.csv` | Met | Committed file header is exactly `sample_id,price`; `to_csv(index=False)` everywhere. |
| REQ-03 Every test `sample_id` present, count identical to `test.csv`, IDs match | `src/train.py:236` (`expected_count=len(test_df)`), `src/data.py:99-103`, `src/gpu_train_foundation.py:538` | **Partial** | Row count (75,000) and ID uniqueness verified on the artefact. Exact ID-set/order alignment **unverifiable**: `dataset/test.csv` is git-ignored (evaluation test skipped; native `tests/test_submission.py` fails). `validate_submission` never checks IDs or duplicates — `test_duplicate_sample_ids_rejected` **fails**. |
| REQ-04 Predicted prices are positive float values | `src/data.py:91-96`, `src/train.py:233` (`np.maximum(pred, 1e-5)`), `src/postprocess.py:84-117` (floor + finite assert) | Met | Artefact: min 0.976, max 287.46, 0 NaN, 0 ±inf, 0 non-positive. Gap: validator accepts `np.inf` (see REQ-21). |
| REQ-05 Prediction produced for all 75,000 test samples | `dataset/test_out.csv` (75,000 data rows) | Met | Verified by `test_row_count_matches_challenge_test_size`. |
| REQ-06 Format identical to `dataset/sample_test_out.csv` | `dataset/sample_test_out.csv`, `dataset/test_out.csv` | Met | Same header, same dtypes (int64 id, float64 price); sample file IDs align 1:1 with `sample_test.csv` (100 rows). |
| REQ-07 Image download via `src/utils.py::download_images` with retry for throttling | `src/utils.py:57-70`, `src/utils.py:28-55`; hardened variant `src/download_images.py:62-122` (retries, tmp-file, PIL check) | Met | 32 mocked-HTTP tests pass: retry after 429/503, give-up after N, corrupt payload rejected, tmp-file cleanup, idempotency, `{sample_id}.jpg` naming. |
| REQ-08 Sample notebook `src/test.ipynb` referenced by statement/README | N/A (`notebooks/kaggle_amazon_ml_solution.ipynb` exists instead) | Omitted | `README.md:15` points to a file that does not exist. Cosmetic, but a broken statement reference. |
| REQ-09 `sample_code.py` generates a valid output file | `sample_code.py:5-19` (`predictor`), `sample_code.py:21-42` | **Omitted (Broken)** | **Does not compile**: docstring opened with `'''` (line 6) and closed with `"""` (line 16) → `SyntaxError: unterminated triple-quoted string literal`. `python sample_code.py` aborts. Also unseeded `random.uniform(5,500)`. `AGENTS.md` calls it "Verified starter submission generator" — false. 3 evaluation tests fail, 1 skipped. |
| REQ-10 Dataset files `dataset/train.csv`, `dataset/test.csv` available for the pipeline | `.gitignore` excludes them; `configs/default.yaml` (`paths.dataset_dir`) | Omitted | Only `sample_test.csv`, `sample_test_out.csv`, `test_out.csv` exist. `AGENTS.md` §5 lists both as present. Consequence: 13 native tests fail; pipeline cannot run; `src/download_data.py:7` fetches an unofficial Kaggle mirror `suvroo/amazon-ml` (provenance risk, not the portal download). |
| REQ-11 Final model MIT / Apache-2.0 licensed | `src/extract_embeddings.py:36,98,193,196` (`BAAI/bge-large-en-v1.5` MIT, `google/siglip-base-patch16-224` Apache-2.0, `facebook/dinov2-base` Apache-2.0); GBDTs: LightGBM (MIT), CatBoost (Apache-2.0), XGBoost (Apache-2.0); scikit-learn BSD-3 | Met | Verified statically: every HF model id in code is on the permissive allowlist; installed package metadata confirms LightGBM/CatBoost/sklearn licences. No hosted/closed LLM references anywhere. |
| REQ-12 Model ≤ 8 B parameters | same as REQ-11 | Met | Largest component BGE-large ≈ 0.335 B; SigLIP-base ≈ 0.203 B; DINOv2-base ≈ 0.086 B. Adapter (`src/adapter.py:69-160`) is a few-hundred-K-param MLP. |
| REQ-13 SMAPE implemented per competition formula (percentage, 0–200 bounded) | `src/metrics.py:21-58` (`smape`), `src/error_analysis.py:22-30` (`smape_per_sample`); surrogate `src/objectives.py:16-67`, `src/adapter.py:30-66` (`DifferentiableSMAPELoss`) | Met | Worked example 100→120 = 18.1818 % reproduced; symmetry, scale-invariance, [0,200] bounds, shape-mismatch `ValueError` all pass. Edge: empty input returns `nan` (not negative, not raised). |
| REQ-14 No external price lookup (fair play) | All `src/*.py`, `*.sh`, `configs/*.yaml`, notebook — static scan | Met | 0 hits for scraping/pricing-API/search-API tooling; `requests`/HF/kagglehub calls confined to `utils.py`, `download_images.py`, `extract_embeddings.py`, `download_data.py`; `price` column read only from `train.csv`. Caveat: `download_data.py` pulls a third-party dataset mirror — same data by claim, unverifiable. |
| REQ-15 One-page methodology document (methodology, architecture, feature engineering) | `Documentation_template.md` (887 words) | **Partial** | All required sections present, no template placeholders. 887 words exceeds a typical single page (~500–650 words of prose) — `test_document_is_roughly_one_page` fails. |
| REQ-16 Textual features from `catalog_content` | `src/features.py:22-82` (`extract_field`), `:109-133` (`extract_pack_quantity`), `:229-255` (unit conversions), `:294-380` (`extract_structured_features`), `:382-451` (TF-IDF), `:656-723` (advanced catalog) | Met | 30+ evaluation tests pass incl. "6 ft cable" not treated as pack-of-6, unit canonicalisation, brand extraction, TF-IDF vocab sharing. |
| REQ-17 Visual features from product images | `src/features.py:453-615` (metadata + SVD), `src/extract_embeddings.py:95-190` (SigLIP/DINOv2), `src/adapter.py:69-160` (`MultimodalPricingAdapter`, gated fusion) | Met | SVD fit-on-train-only and zero-vision gating verified with synthetic embeddings. End-to-end image→embedding path not executable offline (no images, no weights) — code path Met, results are a claim. |
| REQ-18 Ensemble methods | `src/ensemble.py:14-71` (Nelder-Mead convex blend), `:73-102` (`apply_blend`), `src/stacking.py:22-200`, `src/gpu_train_foundation.py` blend section | Met | Weights sum to 1, non-negative, blend ≤ worst single model, single-model degenerate case handled. |
| REQ-19 Outlier handling / preprocessing | `src/features.py:725-742` (`compute_iqr_training_mask`, log-IQR), `src/train.py:44-97` (log target + `clip_min`), `src/ensemble.py:104-140` (price-stratified folds), `src/postprocess.py:22-380` (multiplier, floor, power-law calibration) | Met | All calibration paths keep outputs finite/positive and rank-preserving; stratified folds partition & reproducible. |
| REQ-20 Robustness to malformed/missing catalog text | `src/features.py:366` (`.str.len()` on raw column), `src/features.py:414` (TF-IDF on raw series), `src/train.py:79` | **Partial** | Mixed NaN rows survive only when at least one string is present. All-NaN `catalog_content` column → `AttributeError` (`features.py:366`); any NaN reaching TF-IDF → `ValueError: np.nan is an invalid document` (`features.py:414`, via `train.py:79`). A single blank cell in `test.csv` would crash the CPU pipeline. 3 evaluation tests fail. |
| REQ-21 Submission validator rejects every invalid payload | `src/data.py:67-104` (`validate_submission`) | **Partial** | Rejects missing column, NaN, ≤0, wrong count. **Accepts `inf`**, **accepts duplicate `sample_id`**, and raises `TypeError` (not `ValueError`) for non-numeric price. 3 evaluation tests fail; 9 pass. |
| REQ-22 Reproducible build (declared dependencies, importable on CPU) | N/A — no manifest; `src/gpu_train_foundation.py:16-17`, `src/gpu_train.py:17-18`, `src/kaggle_pipeline.py:16-17` hard-import `lightgbm`/`catboost` at module scope | Omitted | See build ruling in §1. `AGENTS.md` claims "139 passed, 3 skipped"; observed on a clean clone: 129 passed, 13 failed. |
| REQ-23 Security hygiene (SSRF-safe URLs, no secrets/private infra in repo) | `src/utils.py:10-26`, `src/download_images.py:24-47` (HTTP/HTTPS allowlist); `run_gpu_foundation.sh:4,23`; `run_gpu_milestone5.sh:20` | **Partial** | URL allowlist rejects `file:`, `ftp:`, `gopher:`, `javascript:`, `data:`, scheme-less, host-less, non-string — 13/13 pass. No API keys/tokens found. **But** `run_gpu_foundation.sh` embeds an institutional SSH username and a private IPv4 address, and `run_gpu_milestone5.sh:20` hard-codes `$HOME/Amazon-Ml-Prep` (wrong repo name). |

**Score derivation:** Met = 14 (REQ-01,02,04,05,06,07,11,12,13,14,16,17,18,19); Partial = 5 (REQ-03,15,20,21,23); Omitted = 4 (REQ-08,09,10,22). (14 + 0.5×5) / 23 = 71.7 % → **72 %**.

---

## 3. Automated Test Suite Metrics

- **Framework:** pytest (native framework of the repo; `pytest-cov` added for coverage).
- **Location:** `evaluation_suite/` — 9 modules, 173 test functions, **351 collected tests** after parametrisation (the fair-play/secrets scanners are parametrised over every source file).
- **Total Tests Authored:** 351 (173 functions)
- **Total Passed / Failed / Skipped:** **331 / 17 / 3**
- **Code Coverage achieved by the evaluation suite:** **42 %** of `src/` + `sample_code.py` (2,230 statements, 1,286 missed). Combined with native tests: **47 %**.
- **Native suite alongside:** 129 passed / 13 failed of 142 (`tests/test_data.py::TestLoadTrain*`, `TestLoadTest*`, `tests/test_submission.py` — all `FileNotFoundError` on the git-ignored CSVs).
- **Runtime:** 4.9 s (evaluation), 2.8 s (native). Fully deterministic (fixed seeds, `np.random.default_rng`), no network (`requests.get` monkey-patched; `multiprocessing.Pool` replaced by a serial stub), no dependence on local paths beyond the repo root.
- **Raw logs:** `evaluation_suite/results/evaluation_suite_run.log`, `evaluation_suite/results/native_tests_run.log`, `evaluation_suite/results/combined_run.log`, `evaluation_suite/results/coverage_evaluation_suite.json`.

### Per-module coverage (evaluation suite only)

| Module | Stmts | Miss | Cover | Comment |
| :--- | ---: | ---: | ---: | :--- |
| `src/data.py` | 37 | 0 | 100 % | |
| `src/metrics.py` | 15 | 0 | 100 % | |
| `src/ensemble.py` | 45 | 3 | 93 % | |
| `src/adapter.py` | 165 | 20 | 88 % | CPU, 2-fold × 3 epochs |
| `src/utils.py` | 46 | 6 | 87 % | |
| `src/stacking.py` | 86 | 19 | 78 % | |
| `src/features.py` | 272 | 69 | 75 % | image-metadata worker not reachable without files |
| `src/knn_features.py` | 89 | 22 | 75 % | |
| `src/train.py` | 143 | 46 | 68 % | `main()` needs full dataset |
| `src/download_images.py` | 136 | 47 | 65 % | CLI `main()` uncovered |
| `src/postprocess.py` | 141 | 62 | 56 % | |
| `src/objectives.py` | 79 | 36 | 54 % | |
| `src/error_analysis.py` | 103 | 83 | 19 % | |
| `src/download_data.py`, `eda.py`, `extract_embeddings.py`, `gpu_train.py`, `gpu_train_foundation.py`, `kaggle_pipeline.py`, `test_baseline.py`, `sample_code.py` | 873 | 873 | 0 % | Require GPU/HF weights/full data, or (sample_code) do not compile |

### The 17 failing evaluation tests — each one is a repository finding, not a test defect

| # | Test | Finding |
| :-- | :--- | :--- |
| 1 | `test_req02::test_infinite_price_rejected` | `validate_submission` accepts `inf` |
| 2 | `test_req02::test_duplicate_sample_ids_rejected` | accepts duplicate IDs |
| 3 | `test_req02::test_non_numeric_price_raises_value_error` | raises `TypeError` instead of `ValueError` |
| 4 | `test_req04::test_structured_features_on_all_nan_catalog_column` | `AttributeError` at `features.py:366` |
| 5 | `test_req04::test_tfidf_tolerates_nan_text` | `ValueError` at `features.py:414` |
| 6 | `test_req05::test_pipeline_survives_empty_and_nan_catalogs_in_test` | CPU pipeline crashes on one NaN catalog row |
| 7 | `test_req08::test_no_hardcoded_private_hosts_or_usernames_in_scripts` | private IP + username in `run_gpu_foundation.sh` |
| 8 | `test_req09::test_statement_referenced_notebook_src_test_ipynb_exists` | `src/test.ipynb` missing |
| 9–10 | `test_req09::test_dataset_files_present_in_checkout[train.csv/test.csv]` | datasets git-ignored |
| 11 | `test_req09::test_sample_code_parses_and_declares_predictor` | `SyntaxError` in `sample_code.py` |
| 12 | `test_req09::test_sample_code_is_seeded_for_reproducibility` | unseeded RNG |
| 13 | `test_req09::test_sample_code_runs_end_to_end_on_sample_test` | script aborts |
| 14 | `test_req09::test_document_is_roughly_one_page` | 887 words |
| 15 | `test_req09::test_dependency_manifest_exists` | no requirements/pyproject |
| 16 | `test_req09::test_optional_heavy_dependencies_are_guarded_at_import` | module-scope GBDT imports in 3 pipelines |
| 17 | `test_req09::test_run_scripts_do_not_hardcode_a_foreign_repo_path` | `Amazon-Ml-Prep` path assumption |

Skipped (3): exact-ID alignment vs `test.csv` (file absent), backup-submission check (no `test_out_*.csv` backup present despite `AGENTS.md` listing one), `sample_code` predictor behaviour (cannot compile).

---

## 4. Technical Debt & Code Quality Defects

**Critical**

- **`sample_code.py` is syntactically invalid** (mismatched `'''`/`"""`, lines 6/16). The one script the problem statement names as the reference output generator cannot run; `AGENTS.md` describes it as "verified".
- **No dependency manifest.** A clean clone cannot be installed; the exact library versions behind the claimed 41.48 % SMAPE are unrecoverable. `torch`, `transformers`, `faiss`, `xgboost`, `catboost`, `lightgbm`, `kagglehub` are all implicit.
- **Training/test data and every derived artefact (images, embeddings, model weights, OOF predictions) are absent**, so no result in `AGENTS.md`/`Documentation_template.md` is reproducible from the repository. The only verifiable artefact is the final `test_out.csv`. The `.gitignore`-vs-`AGENTS.md` disagreement about what the `dataset/` directory contains is itself a documentation defect.
- **Submission validator gaps** (`src/data.py:67-104`): no finiteness check, no duplicate-ID check, no ID-set comparison against `test.csv`, and a non-numeric column produces an unhandled `TypeError`. `save_submission` masks negative predictions with `np.maximum(…, 1e-5)` rather than failing loudly — a model bug producing negative prices would silently ship a 0.00001 price.
- **Pipeline crashes on a single missing `catalog_content` cell** (`features.py:366`, `features.py:414`). The hidden 75 k test set is not guaranteed NaN-free; `gpu_train_foundation.py` and `kaggle_pipeline.py` share the same feature code.

**Security / hygiene**

- `run_gpu_foundation.sh:4,23` commits an institutional SSH login and a private IPv4 address of a GPU server. No API keys, HF tokens or cloud credentials were found (scanned all `.py/.sh/.yaml/.ipynb`).
- `src/download_data.py:7` downloads the competition data from an unofficial Kaggle mirror (`suvroo/amazon-ml`). Not a price lookup, but the provenance of the training labels is outside the organiser's distribution channel — worth disclosing in the methodology document.
- URL handling is sound: both `is_permitted_url` implementations reject non-HTTP(S), scheme-less, host-less and non-string inputs; downloads go through `.tmp` + `os.replace` and PIL `verify()`. Scheme comparison is case-insensitive (RFC-compliant).

**Severe code smells / architecture**

- **Duplicated logic:** `is_permitted_url` and image downloading exist twice (`src/utils.py` vs `src/download_images.py`); `prepare_features` logic is re-implemented in `train.py`, `gpu_train.py`, `gpu_train_foundation.py` and `kaggle_pipeline.py` with divergent behaviour.
- **Module-scope heavy imports** (`import lightgbm`, `from catboost import …`) in three pipeline entry points make the modules un-importable — and untestable — on any machine without those wheels; (`adapter.py`, by contrast, guards `torch` correctly behind `HAS_TORCH`).
- **Hard-coded environment assumptions:** `run_gpu_milestone5.sh:20` `REPO_DIR="$HOME/Amazon-Ml-Prep"` (repo is `Amazon-ML-Challenge-2025`); `run_gpu_foundation.sh` also refers to `Amazon-Ml-Prep`. Scripts fail out-of-the-box.
- **Documentation drift:** `AGENTS.md` claims 139/3 tests, a backup `test_out_47_32_smape.csv`, present datasets, a "verified" `sample_code.py`, and a prompt that converts numbers to words (`12 → twelve`); the actual `build_llm_prompt` emits `Size: 12 oz` (only the pack count is verbalised). `README.md:15` references a non-existent `src/test.ipynb`.
- **Metric edge cases:** `smape([], [])` returns `nan` with a numpy warning instead of raising; acceptable but undocumented.
- **`Documentation_template.md`** is 887 words — over the "1-page" brief; it also still carries the word "template" in its filename although it is the final document.

**Positive observations (for balance)**

- Log-target GBDT + calibrated blend architecture is metric-appropriate (SMAPE ≈ |Δ log price|).
- Anti-pattern guards in `AGENTS.md` §7 are actually respected in code: pack quantity is a feature not a multiplier (`"6 ft cable"` → `pack_qty = 1`), SVD fitted on train only, k-NN features built out-of-fold, calibration exponent clamped ≥ 0.5, all post-processing asserts finite positive output.
- Native test suite is broad (142 tests) and the 129 that do not need the data all pass.

---

### Reproduction

```bash
python -m venv venv && source venv/bin/activate
pip install numpy pandas scipy scikit-learn lightgbm catboost xgboost pillow requests pyyaml pytest pytest-cov
pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest evaluation_suite -v --cov=src --cov=sample_code   # 331 passed, 17 failed, 3 skipped
python -m pytest tests -v                                          # 129 passed, 13 failed (datasets absent)
```
