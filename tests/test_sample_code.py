"""Tests for starter submission script sample_code.py."""
import ast
import os
import subprocess
import sys
import pandas as pd
import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLE_CODE_PATH = os.path.join(REPO_ROOT, "sample_code.py")
SAMPLE_TEST_CSV = os.path.join(REPO_ROOT, "dataset", "sample_test.csv")


def test_sample_code_compiles_without_syntax_error():
    """Verify sample_code.py has valid Python syntax and parses into an AST."""
    with open(SAMPLE_CODE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    tree = ast.parse(content, filename="sample_code.py")
    assert tree is not None, "Failed to parse sample_code.py"


def test_sample_code_declares_predictor_interface():
    """Verify sample_code.py declares predictor(sample_id, catalog_content, image_link)."""
    with open(SAMPLE_CODE_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    tree = ast.parse(content, filename="sample_code.py")
    functions = {node.name: node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    assert "predictor" in functions, "predictor function not found in sample_code.py"
    
    args = [arg.arg for arg in functions["predictor"].args.args]
    assert args == ["sample_id", "catalog_content", "image_link"], f"Unexpected predictor signature: {args}"


def test_sample_code_predictor_returns_positive_float():
    """Verify predictor returns a valid positive float."""
    ns = {}
    with open(SAMPLE_CODE_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    module_code = code.split('if __name__ == "__main__":')[0]
    exec(module_code, ns)
    
    predict_fn = ns["predictor"]
    for i in range(10):
        val = predict_fn(100 + i, "Test Catalog Content", "https://example.com/test.jpg")
        assert isinstance(val, (int, float)), f"Expected float, got {type(val)}"
        assert float(val) > 0.0, f"Predicted price must be > 0, got {val}"


def test_sample_code_runs_end_to_end(tmp_path):
    """Verify sample_code.py runs cleanly as a CLI script and generates valid output."""
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    
    # Place sample test data as dataset/test.csv
    sample_df = pd.read_csv(SAMPLE_TEST_CSV)
    sample_df.to_csv(dataset_dir / "test.csv", index=False)
    
    # Copy sample_code.py into tmp_path
    with open(SAMPLE_CODE_PATH, "r", encoding="utf-8") as f:
        code = f.read()
    (tmp_path / "sample_code.py").write_text(code, encoding="utf-8")
    
    # Execute sample_code.py in the isolated directory
    res = subprocess.run(
        [sys.executable, "sample_code.py"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert res.returncode == 0, f"Execution failed with stderr:\n{res.stderr}"
    
    out_file = dataset_dir / "test_out.csv"
    assert out_file.exists(), "test_out.csv was not generated"
    
    out_df = pd.read_csv(out_file)
    assert list(out_df.columns) == ["sample_id", "price"]
    assert len(out_df) == len(sample_df)
    assert (out_df["price"] > 0).all()
