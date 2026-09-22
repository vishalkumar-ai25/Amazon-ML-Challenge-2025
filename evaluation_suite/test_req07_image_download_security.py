"""REQ-17..19: Image download helper (src/utils.py::download_images as mandated by the
statement, plus src/download_images.py), URL validation (SSRF / local-file guard), retry
behaviour under throttling, and integrity handling. All HTTP traffic is mocked.
"""
from __future__ import annotations

import io
import os
from unittest import mock

import numpy as np
import pandas as pd
import pytest

import src.download_images as dl
import src.utils as utils

PIL = pytest.importorskip("PIL", reason="Pillow not installed")
from PIL import Image  # noqa: E402


def _png_bytes(size=(64, 64)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buf, format="PNG")
    data = buf.getvalue()
    # is_image_valid requires > 1KB; pad PNG with a tEXt-safe trailing chunk is complex,
    # so use a bigger image if needed.
    if len(data) <= 1024:
        buf = io.BytesIO()
        arr = (np.random.default_rng(0).integers(0, 255, (128, 128, 3))).astype("uint8")
        Image.fromarray(arr).save(buf, format="PNG")
        data = buf.getvalue()
    assert len(data) > 1024
    return data


class _FakeResponse:
    def __init__(self, status=200, body=b"", raise_exc=None):
        self.status_code = status
        self._body = body
        self._raise = raise_exc

    def __enter__(self):
        if self._raise:
            raise self._raise
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise dl.requests.HTTPError(f"status {self.status_code}")

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]


class TestURLValidation:
    @pytest.mark.parametrize(
        "url",
        [
            "https://m.media-amazon.com/images/I/71XfHPR36-L.jpg",
            "http://images.amazon.com/a.png",
            "  https://x.com/a.jpg  ",
            "HTTPS://UPPER.COM/x.jpg",  # scheme is case-insensitive per RFC 3986
        ],
    )
    def test_http_https_permitted(self, url):
        assert utils.is_permitted_url(url) is True
        assert dl.is_permitted_url(url) is True

    @pytest.mark.parametrize(
        "url",
        [
            "file:///etc/passwd",
            "ftp://host/img.jpg",
            "gopher://host/x",
            "javascript:alert(1)",
            "data:image/png;base64,AAAA",
            "//no-scheme.com/img.jpg",
            "https://",
            "http:///path-without-host",
            "",
            "   ",
            None,
            123,
            b"https://bytes.com/x.jpg",
        ],
    )
    def test_non_http_or_malformed_rejected(self, url):
        assert utils.is_permitted_url(url) is False
        assert dl.is_permitted_url(url) is False

    def test_both_modules_agree_on_permitted_schemes(self):
        assert utils.PERMITTED_SCHEMES == dl.PERMITTED_SCHEMES == {"http", "https"}


class TestLegacyUtilsDownloader:
    def test_download_image_writes_file_on_success(self, tmp_path):
        body = b"x" * 5000
        with mock.patch.object(utils.requests, "get", return_value=_FakeResponse(200, body)) as g:
            utils.download_image("https://img.example.com/a.jpg", str(tmp_path))
        assert (tmp_path / "a.jpg").read_bytes() == body
        assert not (tmp_path / "a.jpg.tmp").exists()
        g.assert_called_once()

    def test_download_image_skips_forbidden_scheme_without_network(self, tmp_path):
        with mock.patch.object(utils.requests, "get") as g:
            utils.download_image("file:///etc/passwd", str(tmp_path))
        g.assert_not_called()
        assert list(tmp_path.iterdir()) == []

    def test_download_image_http_error_leaves_no_partial_file(self, tmp_path, capsys):
        with mock.patch.object(utils.requests, "get", return_value=_FakeResponse(503, b"")):
            utils.download_image("https://img.example.com/b.jpg", str(tmp_path))
        assert list(tmp_path.iterdir()) == []
        assert "Not able to download" in capsys.readouterr().out

    def test_download_image_network_exception_is_swallowed(self, tmp_path):
        with mock.patch.object(utils.requests, "get", side_effect=utils.requests.ConnectionError("boom")):
            utils.download_image("https://img.example.com/c.jpg", str(tmp_path))  # must not raise
        assert list(tmp_path.iterdir()) == []

    def test_download_image_is_idempotent(self, tmp_path):
        (tmp_path / "d.jpg").write_bytes(b"existing")
        with mock.patch.object(utils.requests, "get") as g:
            utils.download_image("https://img.example.com/d.jpg", str(tmp_path))
        g.assert_not_called()

    def test_download_images_creates_folder_and_dispatches(self, tmp_path):
        folder = tmp_path / "imgs"
        # Replace the multiprocessing pool with a serial stub so mocks apply in-process.
        class _Pool:
            def __init__(self, *a, **k): ...
            def __enter__(self): return self
            def __exit__(self, *a): return False
            def imap(self, fn, it): return map(fn, it)

        with mock.patch.object(utils.multiprocessing, "Pool", _Pool), mock.patch.object(
            utils.requests, "get", return_value=_FakeResponse(200, b"y" * 2048)
        ):
            utils.download_images(["https://img.example.com/1.jpg", "https://img.example.com/2.jpg"], str(folder))
        assert sorted(p.name for p in folder.iterdir()) == ["1.jpg", "2.jpg"]


class TestResilientDownloader:
    def test_success_with_integrity_check(self, tmp_path):
        target = tmp_path / "1.jpg"
        with mock.patch.object(dl.requests, "get", return_value=_FakeResponse(200, _png_bytes())):
            assert dl.download_single_image("https://img.example.com/1.jpg", str(target)) is True
        assert dl.is_image_valid(str(target))

    def test_retries_on_throttling_then_succeeds(self, tmp_path):
        target = tmp_path / "2.jpg"
        responses = [_FakeResponse(429, b""), _FakeResponse(503, b""), _FakeResponse(200, _png_bytes())]
        with mock.patch.object(dl.requests, "get", side_effect=responses) as g, mock.patch.object(dl.time, "sleep"):
            assert dl.download_single_image("https://img.example.com/2.jpg", str(target), retries=2) is True
        assert g.call_count == 3

    def test_gives_up_after_retries_exhausted(self, tmp_path):
        target = tmp_path / "3.jpg"
        with mock.patch.object(dl.requests, "get", return_value=_FakeResponse(500, b"")) as g, mock.patch.object(dl.time, "sleep"):
            assert dl.download_single_image("https://img.example.com/3.jpg", str(target), retries=2) is False
        assert g.call_count == 3
        assert not target.exists() and not (tmp_path / "3.jpg.tmp").exists()

    def test_corrupt_payload_rejected_by_integrity_check(self, tmp_path):
        target = tmp_path / "4.jpg"
        with mock.patch.object(dl.requests, "get", return_value=_FakeResponse(200, b"not an image" * 200)), mock.patch.object(dl.time, "sleep"):
            assert dl.download_single_image("https://img.example.com/4.jpg", str(target), retries=0) is False
        assert not target.exists()

    def test_forbidden_url_short_circuits(self, tmp_path):
        with mock.patch.object(dl.requests, "get") as g:
            assert dl.download_single_image("file:///etc/passwd", str(tmp_path / "x.jpg")) is False
        g.assert_not_called()

    def test_dataset_download_names_files_by_sample_id_and_counts(self, tmp_path):
        df = pd.DataFrame(
            {
                "sample_id": [11, 22, 33],
                "image_link": ["https://i.example.com/a.jpg", "ftp://bad/b.jpg", np.nan],
            }
        )
        with mock.patch.object(dl.requests, "get", return_value=_FakeResponse(200, _png_bytes())):
            stats = dl.download_dataset_images(df, str(tmp_path), max_workers=2, retries=0)
        assert stats == {"success": 1, "failed": 1}  # NaN row dropped, ftp rejected
        assert (tmp_path / "11.jpg").exists()

    def test_audit_reports_missing_corrupt_valid(self, tmp_path):
        (tmp_path / "1.jpg").write_bytes(_png_bytes())
        (tmp_path / "2.jpg").write_bytes(b"junk" * 500)
        df = pd.DataFrame({"sample_id": [1, 2, 3]})
        assert dl.audit_downloaded_images(df, str(tmp_path)) == {"total": 3, "valid": 1, "missing": 1, "corrupt": 1}

    def test_is_image_valid_rejects_tiny_or_missing(self, tmp_path):
        assert dl.is_image_valid(str(tmp_path / "missing.jpg")) is False
        (tmp_path / "tiny.jpg").write_bytes(b"x" * 10)
        assert dl.is_image_valid(str(tmp_path / "tiny.jpg")) is False
