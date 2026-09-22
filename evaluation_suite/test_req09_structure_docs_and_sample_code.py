"""REQ-24..28 (static + dynamic): repository structure demanded by the problem statement,
sample_code.py behaviour, one-page methodology document, and build reproducibility.
"""
from __future__ import annotations

import ast
import os
import re
import runpy
import subprocess
import sys

import pandas as pd
import pytest

from evaluation_suite.conftest import DATASET_DIR, REPO_ROOT, SRC_DIR

README = os.path.join(REPO_ROOT, "README.md")
DOC = os.path.join(REPO_ROOT, "Documentation_template.md")


def _read(p):
    with open(p, encoding="utf-8", errors="ignore") as fh:
        return fh.read()


class TestMandatedFilesExist:
    @pytest.mark.parametrize(
        "rel",
        [
            "src/utils.py",
            "sample_code.py",
            "dataset/sample_test.csv",
            "dataset/sample_test_out.csv",
            "Documentation_template.md",
            "README.md",
        ],
    )
    def test_statement_referenced_file_exists(self, rel):
        assert os.path.exists(os.path.join(REPO_ROOT, rel)), rel

    def test_statement_referenced_notebook_src_test_ipynb_exists(self):
        # Problem statement: "See sample code in src/test.ipynb".
        assert os.path.exists(os.path.join(SRC_DIR, "test.ipynb"))

    @pytest.mark.parametrize("rel", ["dataset/train.csv", "dataset/test.csv"])
    def test_dataset_files_present_in_checkout(self, rel):
        # Git-ignored: the statement's file layout cannot be reproduced from the clone alone.
        assert os.path.exists(os.path.join(REPO_ROOT, rel)), f"{rel} missing (git-ignored)"

    def test_utils_exposes_download_images_function(self):
        tree = ast.parse(_read(os.path.join(SRC_DIR, "utils.py")))
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "download_images" in names

    def test_src_is_an_importable_package(self):
        assert os.path.exists(os.path.join(SRC_DIR, "__init__.py"))
        import src  # noqa: F401


class TestSampleCode:
    def test_sample_code_parses_and_declares_predictor(self):
        tree = ast.parse(_read(os.path.join(REPO_ROOT, "sample_code.py")))
        names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
        assert "predictor" in names

    def test_sample_code_predictor_outputs_positive_float(self):
        ns = {}
        code = _read(os.path.join(REPO_ROOT, "sample_code.py"))
        module_only = code.split("if __name__")[0]
        try:
            compiled = compile(module_only, "sample_code.py", "exec")
        except SyntaxError as exc:
            pytest.skip(f"sample_code.py does not compile (covered by the parse test): {exc}")
        exec(compiled, ns)
        for _ in range(50):
            p = ns["predictor"](1, "Item Name: x", "https://x/y.jpg")
            assert isinstance(p, float) and p > 0

    def test_sample_code_is_seeded_for_reproducibility(self):
        src = _read(os.path.join(REPO_ROOT, "sample_code.py"))
        assert re.search(r"random\.seed\(|np\.random\.seed\(|default_rng\(", src), "predictor is unseeded/non-deterministic"

    def test_sample_code_runs_end_to_end_on_sample_test(self, tmp_path):
        # Run sample_code.py in an isolated copy where dataset/test.csv := sample_test.csv.
        ds = tmp_path / "dataset"
        ds.mkdir()
        pd.read_csv(os.path.join(DATASET_DIR, "sample_test.csv")).to_csv(ds / "test.csv", index=False)
        (tmp_path / "sample_code.py").write_text(_read(os.path.join(REPO_ROOT, "sample_code.py")))
        res = subprocess.run([sys.executable, "sample_code.py"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
        assert res.returncode == 0, res.stderr
        out = pd.read_csv(ds / "test_out.csv")
        st = pd.read_csv(ds / "test.csv")
        assert list(out.columns) == ["sample_id", "price"]
        assert list(out["sample_id"]) == list(st["sample_id"])
        assert (out["price"] > 0).all()


class TestMethodologyDocument:
    def test_required_sections_present(self):
        doc = _read(DOC).lower()
        for kw in ("methodology", "architecture", "feature engineering"):
            assert kw in doc, kw

    def test_no_unfilled_template_placeholders(self):
        doc = _read(DOC)
        assert not re.search(r"\[(Your|Insert|TODO|Team Name|Describe)[^\]]*\]", doc, flags=re.IGNORECASE)
        assert "TODO" not in doc and "TBD" not in doc

    def test_document_is_roughly_one_page(self):
        # Statement asks for a 1-page document; ~500-600 words is a full page of prose.
        words = len(_read(DOC).split())
        assert words <= 650, f"{words} words exceeds a one-page document"

    def test_readme_states_smape_and_output_format(self):
        readme = _read(README)
        assert "SMAPE" in readme and "sample_id" in readme and "price" in readme


class TestBuildReproducibility:
    def test_dependency_manifest_exists(self):
        candidates = ["requirements.txt", "pyproject.toml", "setup.py", "environment.yml", "Pipfile"]
        assert any(os.path.exists(os.path.join(REPO_ROOT, c)) for c in candidates), (
            "no dependency manifest; the repo cannot be installed reproducibly"
        )

    def test_optional_heavy_dependencies_are_guarded_at_import(self):
        # Modules that hard-import GPU-only / optional libs at module scope break `import src.x` on CPU boxes.
        offenders = []
        for f in ("gpu_train_foundation.py", "gpu_train.py", "kaggle_pipeline.py"):
            src = _read(os.path.join(SRC_DIR, f))
            tree = ast.parse(src)
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    mod = (node.module if isinstance(node, ast.ImportFrom) else node.names[0].name) or ""
                    if mod.split(".")[0] in {"lightgbm", "catboost", "xgboost", "torch", "faiss"}:
                        offenders.append(f"{f}:{mod}")
        assert not offenders, offenders

    def test_run_scripts_do_not_hardcode_a_foreign_repo_path(self):
        for f in ("run_gpu_foundation.sh", "run_gpu_milestone5.sh"):
            p = os.path.join(REPO_ROOT, f)
            if not os.path.exists(p):
                continue
            src = _read(p)
            assert "Amazon-Ml-Prep" not in src, f"{f} assumes $HOME/Amazon-Ml-Prep, but the repo is Amazon-ML-Challenge-2025"

    def test_core_modules_import_without_optional_dependencies(self):
        for m in ("src.data", "src.metrics", "src.features", "src.ensemble", "src.postprocess", "src.utils", "src.download_images"):
            __import__(m)

    def test_default_config_output_path_is_inside_dataset_dir(self):
        import yaml

        cfg = yaml.safe_load(_read(os.path.join(REPO_ROOT, "configs", "default.yaml")))
        assert cfg["paths"]["output_file"] == "dataset/test_out.csv"
