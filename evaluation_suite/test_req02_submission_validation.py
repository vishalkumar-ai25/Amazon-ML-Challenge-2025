"""REQ-03/04 (dynamic): src.data.validate_submission and src.train.save_submission.

Happy path, boundary values (price == 0, tiny positive), and malformed payloads
(NaN, Inf, negative, duplicate IDs, missing columns, wrong dtype, wrong row count).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data import load_test, load_train, validate_submission
from src.train import save_submission


def _sub(prices, ids=None):
    ids = list(range(len(prices))) if ids is None else ids
    return pd.DataFrame({"sample_id": ids, "price": prices})


class TestValidateSubmissionHappyPath:
    def test_valid_submission_passes(self):
        validate_submission(_sub([1.0, 2.5, 300.0]))

    def test_expected_count_matches(self):
        validate_submission(_sub([1.0, 2.0]), expected_count=2)

    def test_tiny_positive_price_is_accepted(self):
        validate_submission(_sub([1e-9]))

    def test_extra_columns_are_tolerated(self):
        df = _sub([1.0, 2.0])
        df["extra"] = 1
        validate_submission(df)


class TestValidateSubmissionRejectsInvalid:
    def test_zero_price_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(_sub([0.0, 1.0]))

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(_sub([-1.0, 1.0]))

    def test_nan_price_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(_sub([np.nan, 1.0]))

    def test_missing_price_column_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(pd.DataFrame({"sample_id": [1, 2]}))

    def test_missing_sample_id_column_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(pd.DataFrame({"price": [1.0, 2.0]}))

    def test_row_count_mismatch_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(_sub([1.0, 2.0]), expected_count=3)

    def test_empty_submission_with_expected_count_rejected(self):
        with pytest.raises(ValueError):
            validate_submission(_sub([]), expected_count=1)

    def test_infinite_price_rejected(self):
        # Challenge: "positive float values". +inf is not a usable float price.
        with pytest.raises(ValueError):
            validate_submission(_sub([np.inf, 1.0]))

    def test_duplicate_sample_ids_rejected(self):
        # Challenge: every test sample_id must appear exactly once.
        with pytest.raises(ValueError):
            validate_submission(_sub([1.0, 2.0], ids=[7, 7]))

    def test_non_numeric_price_raises_value_error(self):
        with pytest.raises(ValueError):
            validate_submission(_sub(["abc", "1.0"]))


class TestSaveSubmission:
    def test_writes_exact_two_columns_and_count(self, tmp_path, synthetic_frames):
        _, test_df = synthetic_frames
        out = tmp_path / "test_out.csv"
        preds = np.full(len(test_df), 12.5)
        save_submission(test_df, preds, str(out))
        written = pd.read_csv(out)
        assert list(written.columns) == ["sample_id", "price"]
        assert len(written) == len(test_df)
        assert list(written["sample_id"]) == list(test_df["sample_id"])
        assert (written["price"] > 0).all()

    def test_negative_predictions_are_floored_to_positive(self, tmp_path, synthetic_frames):
        _, test_df = synthetic_frames
        out = tmp_path / "test_out.csv"
        preds = np.linspace(-5, 5, len(test_df))
        save_submission(test_df, preds, str(out))
        assert (pd.read_csv(out)["price"] > 0).all()

    def test_length_mismatch_raises(self, tmp_path, synthetic_frames):
        _, test_df = synthetic_frames
        with pytest.raises(ValueError):
            save_submission(test_df, np.ones(len(test_df) - 1), str(tmp_path / "x.csv"))

    def test_nan_predictions_raise_and_do_not_write(self, tmp_path, synthetic_frames):
        _, test_df = synthetic_frames
        out = tmp_path / "x.csv"
        preds = np.ones(len(test_df))
        preds[0] = np.nan
        with pytest.raises(ValueError):
            save_submission(test_df, preds, str(out))
        assert not out.exists()


class TestInputSchemaLoading:
    def test_load_train_and_test_from_valid_dir(self, synthetic_dataset_dir):
        tr = load_train(synthetic_dataset_dir)
        te = load_test(synthetic_dataset_dir)
        assert {"sample_id", "catalog_content", "image_link", "price"} <= set(tr.columns)
        assert {"sample_id", "catalog_content", "image_link"} <= set(te.columns)
        assert "price" not in te.columns

    def test_missing_files_raise_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_train(str(tmp_path))
        with pytest.raises(FileNotFoundError):
            load_test(str(tmp_path))

    def test_missing_required_column_raises(self, tmp_path):
        pd.DataFrame({"sample_id": [1], "catalog_content": ["x"]}).to_csv(tmp_path / "train.csv", index=False)
        pd.DataFrame({"sample_id": [1], "image_link": ["x"]}).to_csv(tmp_path / "test.csv", index=False)
        with pytest.raises(ValueError):
            load_train(str(tmp_path))
        with pytest.raises(ValueError):
            load_test(str(tmp_path))

    def test_empty_csv_with_header_only_loads_zero_rows(self, tmp_path):
        (tmp_path / "test.csv").write_text("sample_id,catalog_content,image_link\n")
        assert len(load_test(str(tmp_path))) == 0
