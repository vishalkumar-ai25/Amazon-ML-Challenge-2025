"""REQ-20..23 (static): Fair-play (no external price lookup), model licence / size
constraints, and secrets hygiene. All checks are deterministic source scans - no
network access.
"""
from __future__ import annotations

import os
import re

import pytest

from evaluation_suite.conftest import REPO_ROOT, SRC_DIR

# Modules whose *only* legitimate network use is downloading challenge images or
# open-weight model files. Everything else in src/ must be offline.
NETWORK_ALLOWLIST = {
    "utils.py",              # challenge-mandated image downloader
    "download_images.py",    # resilient image downloader
    "extract_embeddings.py", # Hugging Face open-weight model download
    "download_data.py",      # kagglehub mirror of the *provided* dataset (flagged separately)
}

# Known open-weight models used by the pipeline: (regex, licence, params in billions).
MODEL_REGISTRY = {
    r"BAAI/bge-large-en-v1\.5": ("MIT", 0.335),
    r"google/siglip-base-patch16-224": ("Apache-2.0", 0.203),
    r"facebook/dinov2-base": ("Apache-2.0", 0.086),
}
NON_PERMISSIVE_OR_OVERSIZED = [
    r"meta-llama/", r"Llama-2", r"Llama-3", r"mistralai/Mixtral", r"gemma", r"Qwen.*(14|32|72)B",
    r"gpt-4", r"claude", r"text-embedding-ada", r"openai\.", r"anthropic", r"google\.generativeai",
]

PRICE_LOOKUP_PATTERNS = [
    r"scrap(e|ing)\s+.*price", r"price.*scrap", r"amazon.*(api|paapi|product-advertising)",
    r"keepa", r"camelcamelcamel", r"pricecharting", r"rapidapi", r"serpapi",
    r"google\s*shopping", r"walmart.*api", r"ebay.*api", r"BeautifulSoup", r"selenium",
    r"playwright", r"requests\.get\([^)]*price", r"searchapi", r"webscrap",
]

NETWORK_CALL_PATTERNS = [
    r"\brequests\.(get|post|Session)\b", r"\burllib\.request\b", r"\burlopen\b",
    r"\bhttpx\.", r"\baiohttp\b", r"\bsocket\.", r"kagglehub\.", r"snapshot_download|hf_hub_download",
]

SECRET_PATTERNS = [
    r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token)\s*[:=]\s*['\"][A-Za-z0-9_\-]{12,}['\"]",
    r"AKIA[0-9A-Z]{16}",
    r"hf_[A-Za-z0-9]{20,}",
    r"sk-[A-Za-z0-9]{20,}",
    r"ghp_[A-Za-z0-9]{30,}",
    r"(?i)password\s*=\s*['\"][^'\"]{4,}['\"]",
]


def _py_files(root: str):
    for dirpath, _, files in os.walk(root):
        if any(part in dirpath for part in ("venv", "__pycache__", ".git", "evaluation_suite")):
            continue
        for f in files:
            if f.endswith((".py", ".sh", ".yaml", ".yml", ".ipynb")):
                yield os.path.join(dirpath, f)


def _read(path: str) -> str:
    with open(path, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


class TestNoExternalPriceLookup:
    @pytest.mark.parametrize("path", sorted(_py_files(REPO_ROOT)), ids=lambda p: os.path.relpath(p, REPO_ROOT))
    def test_no_price_lookup_or_scraping_tooling(self, path):
        src = _read(path)
        hits = [p for p in PRICE_LOOKUP_PATTERNS if re.search(p, src, flags=re.IGNORECASE)]
        assert not hits, f"possible external price lookup tooling in {path}: {hits}"

    @pytest.mark.parametrize(
        "path", sorted(p for p in _py_files(SRC_DIR) if p.endswith(".py")), ids=lambda p: os.path.basename(p)
    )
    def test_network_calls_confined_to_allowlisted_modules(self, path):
        src = _read(path)
        hits = [p for p in NETWORK_CALL_PATTERNS if re.search(p, src)]
        if os.path.basename(path) in NETWORK_ALLOWLIST:
            return
        assert not hits, f"unexpected network call in {path}: {hits}"

    def test_image_downloaders_only_fetch_image_link_urls(self):
        # Both downloaders must take the URL from the dataset row, not construct search/price URLs.
        for name in ("utils.py", "download_images.py"):
            src = _read(os.path.join(SRC_DIR, name))
            assert not re.search(r"requests\.get\(\s*f?['\"]https?://", src), f"{name} hard-codes a remote endpoint"

    def test_training_target_comes_only_from_train_csv(self):
        # The only place the 'price' column is read must be the training file.
        offenders = []
        for path in _py_files(SRC_DIR):
            src = _read(path)
            if re.search(r"read_csv\([^)]*price", src) or re.search(r"merge\([^)]*price", src, re.IGNORECASE):
                offenders.append(os.path.basename(path))
        assert not offenders, offenders

    def test_external_dataset_mirror_is_flagged(self):
        # download_data.py pulls a Kaggle mirror ("suvroo/amazon-ml"). This is documented here as a
        # provenance risk: the mirror is third-party and not the official portal download.
        src = _read(os.path.join(SRC_DIR, "download_data.py"))
        assert "kagglehub.dataset_download" in src
        assert "suvroo/amazon-ml" in src


class TestModelLicenceAndSize:
    def test_only_registered_permissive_models_are_referenced(self):
        src = _read(os.path.join(SRC_DIR, "extract_embeddings.py"))
        hf_ids = set(re.findall(r"['\"]([A-Za-z0-9_.\-]+/[A-Za-z0-9_.\-]+)['\"]", src))
        hf_ids = {h for h in hf_ids if not h.startswith(("data/", "dataset/", "models/", "./", "/"))}
        assert hf_ids, "no model ids found"
        for hid in hf_ids:
            assert any(re.fullmatch(p, hid) for p in MODEL_REGISTRY), f"unregistered model id {hid}"

    def test_registered_models_are_under_8b_and_permissive(self):
        for _, (licence, params_b) in MODEL_REGISTRY.items():
            assert licence in {"MIT", "Apache-2.0"}
            assert params_b < 8.0

    @pytest.mark.parametrize("path", sorted(_py_files(REPO_ROOT)), ids=lambda p: os.path.relpath(p, REPO_ROOT))
    def test_no_non_permissive_or_hosted_llm_references(self, path):
        src = _read(path)
        hits = [p for p in NON_PERMISSIVE_OR_OVERSIZED if re.search(p, src, flags=re.IGNORECASE)]
        assert not hits, f"{path}: {hits}"

    def test_documentation_asserts_licence_and_parameter_bounds(self):
        doc = _read(os.path.join(REPO_ROOT, "Documentation_template.md"))
        assert re.search(r"Apache\s*2\.0", doc) and "MIT" in doc
        assert re.search(r"8\s*B(illion)?\s*param", doc, flags=re.IGNORECASE)

    def test_gbdt_and_sklearn_dependencies_are_permissive(self):
        # LightGBM (MIT), CatBoost (Apache-2.0), scikit-learn (BSD-3), XGBoost (Apache-2.0) - all permissive.
        import importlib.metadata as md

        for dist, allowed in {
            "lightgbm": ("MIT",),
            "catboost": ("Apache",),
            "scikit-learn": ("BSD", "new BSD"),
        }.items():
            try:
                meta = md.metadata(dist)
            except md.PackageNotFoundError:
                pytest.skip(f"{dist} not installed")
            lic = " ".join(
                filter(None, [meta.get("License"), meta.get("License-Expression"), *(meta.get_all("Classifier") or [])])
            )
            assert any(a.lower() in lic.lower() for a in allowed), f"{dist}: {lic[:120]}"


class TestSecretsHygiene:
    @pytest.mark.parametrize("path", sorted(_py_files(REPO_ROOT)), ids=lambda p: os.path.relpath(p, REPO_ROOT))
    def test_no_hardcoded_credentials(self, path):
        src = _read(path)
        hits = [p for p in SECRET_PATTERNS if re.search(p, src)]
        assert not hits, f"{path}: {hits}"

    def test_no_hardcoded_private_hosts_or_usernames_in_scripts(self):
        # Operational hygiene: run scripts must not embed personal SSH targets.
        offenders = []
        for f in os.listdir(REPO_ROOT):
            if f.endswith(".sh"):
                src = _read(os.path.join(REPO_ROOT, f))
                if re.search(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", src) or re.search(r"\w+@\d{1,3}\.\d{1,3}", src):
                    offenders.append(f)
        assert not offenders, f"private host/user embedded in {offenders}"

    def test_gitignore_excludes_environments_and_weights(self):
        gi = _read(os.path.join(REPO_ROOT, ".gitignore"))
        for pat in ("venv/", "__pycache__/", "*.pt", "*.pkl"):
            assert pat in gi
