import pytest

import nltk.pathsec
from nltk.downloader import urlopen


@pytest.fixture(autouse=True)
def enable_enforcement():
    """Force the sentinel to raise exceptions instead of warnings for testing."""
    original_enforce = nltk.pathsec.ENFORCE
    nltk.pathsec.ENFORCE = True
    yield
    nltk.pathsec.ENFORCE = original_enforce


@pytest.fixture
def mock_original_urlopen(monkeypatch):
    """Prevent actual network calls for valid URLs so tests are fast and offline."""
    monkeypatch.setattr(nltk.pathsec, "_original_urlopen", lambda *args, **kwargs: True)


def test_valid_http_url(mock_original_urlopen):
    """Ensure standard, safe external URLs pass through unobstructed."""
    assert urlopen(
        "https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml"
    )
    assert urlopen("http://nltk.org/nltk_data/")


def test_ssrf_invalid_scheme():
    """Ensure local file access via URL schemes is strictly blocked."""
    with pytest.raises(PermissionError, match="Invalid scheme"):
        urlopen("file:///etc/passwd")

    with pytest.raises(PermissionError, match="Invalid scheme"):
        urlopen("ftp://example.com/data.zip")


def test_ssrf_loopback_ip():
    """Block requests to the local machine (prevents targeting internal admin panels)."""
    with pytest.raises(PermissionError, match="Blocked SSRF attempt"):
        urlopen("http://127.0.0.1/admin")

    with pytest.raises(PermissionError, match="Blocked SSRF attempt"):
        urlopen("http://localhost:8080/api")


def test_ssrf_cloud_metadata_link_local():
    """Block requests to AWS/GCP/Azure Instance Metadata Services (The primary SSRF threat)."""
    with pytest.raises(PermissionError, match="Blocked SSRF attempt"):
        urlopen("http://169.254.169.254/latest/meta-data/iam/security-credentials/")


def test_ssrf_ip_obfuscation():
    """
    Test if the sentinel catches IP obfuscation techniques that bypass regex/string matching.
    2852039166 is the decimal representation of 169.254.169.254.
    """
    with pytest.raises(PermissionError, match="Blocked SSRF attempt"):
        urlopen("http://2852039166/latest/meta-data/")


def test_ssrf_allows_private_intranet(mock_original_urlopen):
    """
    Ensure legitimate corporate/university local mirrors are NOT blocked.
    If these fail, NLTK becomes unusable for enterprise air-gapped environments.
    """
    assert urlopen("http://192.168.1.100/nltk_data/index.xml")
    assert urlopen("http://10.0.0.5/mirror/index.xml")
