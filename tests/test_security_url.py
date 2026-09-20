"""Security audit tests for URL opening and scheme validation.

Verifies:
- Whitelist auditing restricts schemes strictly to http:// and https://
- Rejection of file://, ftp://, javascript:, data: schemes (SSRF and LFI prevention)
- No usage of vulnerable urllib urlopen/urlretrieve APIs
"""
import pytest
from src.download_images import is_permitted_url
from src.utils import is_permitted_url as utils_is_permitted_url


@pytest.mark.parametrize("url,expected", [
    ("https://m.media-amazon.com/images/I/71XfHPR36-L.jpg", True),
    ("http://images.amazon.com/product123.jpg", True),
    ("file:///etc/passwd", False),
    ("file:///C:/Windows/system.ini", False),
    ("ftp://anonymous@ftp.example.com/pub/file.txt", False),
    ("javascript:alert(1)", False),
    ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAUA", False),
    ("gopher://evil.com", False),
    ("", False),
    (None, False),
    (12345, False),
])
def test_is_permitted_url_schemes(url, expected):
    """Ensure that only http/https URLs with valid network locations are permitted."""
    assert is_permitted_url(url) == expected
    assert utils_is_permitted_url(url) == expected


def test_no_legacy_urllib_open_calls():
    """Static analysis test ensuring no legacy urllib urlopen/urlretrieve are present."""
    import inspect
    import src.download_images as d_mod
    import src.utils as u_mod

    d_src = inspect.getsource(d_mod)
    u_src = inspect.getsource(u_mod)

    banned_tokens = [
        "urlopen",
        "urlretrieve",
        "URLopener",
        "FancyURLopener",
    ]

    for token in banned_tokens:
        assert token not in d_src, f"Banned call '{token}' found in download_images.py"
        assert token not in u_src, f"Banned call '{token}' found in utils.py"
