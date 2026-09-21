"""Unit tests for Multimodal Vision Features pipeline (SigLIP / DINOv2 / metadata)."""
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
import pytest
from PIL import Image
from scipy.sparse import csr_matrix

from src.features import (
    extract_visual_metadata_features,
    build_vision_svd_features,
    build_feature_matrix,
    build_numeric_matrix,
    VISUAL_METADATA_COLS,
    NUMERIC_COLS,
)
from src.download_images import (
    is_image_valid,
    audit_downloaded_images,
    is_permitted_url,
)
from src.adapter import HAS_TORCH


@pytest.fixture
def temp_image_environment():
    """Create a temporary directory with valid, corrupt, and tiny images."""
    temp_dir = tempfile.mkdtemp()
    
    # 1. Valid image (> 1KB)
    valid_path = os.path.join(temp_dir, "101.jpg")
    img = Image.new("RGB", (300, 200), color=(200, 50, 80))
    img.save(valid_path, "JPEG", quality=95)
    # Ensure it is > 1024 bytes
    assert os.path.getsize(valid_path) > 1024

    # 2. Another valid image with distinct color
    valid_path2 = os.path.join(temp_dir, "102.jpg")
    img2 = Image.new("RGB", (150, 150), color=(30, 180, 220))
    img2.save(valid_path2, "JPEG", quality=95)

    # 3. Tiny / truncated file (< 1KB)
    tiny_path = os.path.join(temp_dir, "103.jpg")
    with open(tiny_path, "wb") as f:
        f.write(b"tiny header" * 10)  # ~110 bytes

    # 4. Corrupt file (> 1KB but invalid image data)
    corrupt_path = os.path.join(temp_dir, "104.jpg")
    with open(corrupt_path, "wb") as f:
        f.write(b"CORRUPT_BYTES_DATA" * 100)  # ~1800 bytes

    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestImageIntegrityAndAuditing:

    def test_is_image_valid_cases(self, temp_image_environment):
        valid_path = os.path.join(temp_image_environment, "101.jpg")
        tiny_path = os.path.join(temp_image_environment, "103.jpg")
        corrupt_path = os.path.join(temp_image_environment, "104.jpg")
        missing_path = os.path.join(temp_image_environment, "999.jpg")

        assert is_image_valid(valid_path) is True
        assert is_image_valid(tiny_path) is False
        assert is_image_valid(corrupt_path) is False
        assert is_image_valid(missing_path) is False

    def test_audit_downloaded_images(self, temp_image_environment):
        df = pd.DataFrame({"sample_id": [101, 102, 103, 104, 999]})
        report = audit_downloaded_images(df, temp_image_environment)

        assert report["total"] == 5
        assert report["valid"] == 2  # 101 and 102
        assert report["corrupt"] == 2  # 103 (<1KB) and 104 (corrupt)
        assert report["missing"] == 1  # 999

    def test_permitted_url_whitelist(self):
        assert is_permitted_url("https://m.media-amazon.com/images/I/abc.jpg") is True
        assert is_permitted_url("http://images.amazon.com/test.png") is True
        assert is_permitted_url("ftp://malicious.com/file.jpg") is False
        assert is_permitted_url("file:///etc/passwd") is False
        assert is_permitted_url("javascript:alert(1)") is False
        assert is_permitted_url("") is False


class TestVisualMetadataExtraction:

    def test_extract_visual_metadata_features(self, temp_image_environment):
        df = pd.DataFrame({"sample_id": [101, 102, 999]})
        meta_df = extract_visual_metadata_features(df, temp_image_environment)

        assert list(meta_df.columns) == VISUAL_METADATA_COLS
        assert len(meta_df) == 3

        # Sample 101: valid image
        assert meta_df.loc[0, "has_image"] == 1.0
        assert meta_df.loc[0, "img_width"] > 0
        assert meta_df.loc[0, "img_height"] > 0
        assert meta_df.loc[0, "img_aspect_ratio"] == pytest.approx(300 / 200, rel=1e-2)
        assert meta_df.loc[0, "img_file_size_kb"] > 0
        assert 0.0 <= meta_df.loc[0, "img_mean_luminance"] <= 1.0
        assert meta_df.loc[0, "img_colorfulness"] > 0.0

        # Sample 999: missing image
        assert meta_df.loc[2, "has_image"] == 0.0
        assert meta_df.loc[2, "img_width"] == 0.0
        assert meta_df.loc[2, "img_height"] == 0.0
        assert meta_df.loc[2, "img_file_size_kb"] == 0.0


class TestVisionSVDReduction:

    def test_build_vision_svd_features(self):
        np.random.seed(42)
        train_emb = np.random.randn(100, 768).astype(np.float32)
        test_emb = np.random.randn(40, 768).astype(np.float32)

        train_svd, test_svd = build_vision_svd_features(train_emb, test_emb, n_components=32)

        assert train_svd.shape == (100, 32)
        assert test_svd.shape == (40, 32)
        assert not np.isnan(train_svd).any()
        assert not np.isnan(test_svd).any()

    def test_build_feature_matrix_with_vision_svd(self):
        text_csr = csr_matrix(np.ones((10, 50), dtype=np.float32))
        num_csr = csr_matrix(np.ones((10, 15), dtype=np.float32))
        v_svd = np.ones((10, 32), dtype=np.float32)

        full_matrix = build_feature_matrix(text_csr, num_csr, vision_svd_features=v_svd)
        assert full_matrix.shape == (10, 50 + 15 + 32)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch is required for neural adapter tests")
class TestMultimodalAdapterVisionGating:

    def test_cross_modal_gated_forward(self):
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
            out_with_vision = model(text_emb, vision_emb, tabular_feat)

        assert out_with_vision.shape == (batch_size,)
        assert not torch.isnan(out_with_vision).any()

        # Test with missing vision embedding (zeros)
        zero_vision = torch.zeros(batch_size, 64)
        with torch.no_grad():
            out_missing_vision = model(text_emb, zero_vision, tabular_feat)

        assert out_missing_vision.shape == (batch_size,)
        assert not torch.isnan(out_missing_vision).any()
