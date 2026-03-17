from urllib.error import URLError

import pytest

from nltk.downloader import Downloader


@pytest.fixture(autouse=True)
def enable_enforcement():
    """
    Dynamically toggle enforcement if pathsec exists (our branch).
    If pathsec doesn't exist (e.g., PR #3520), just yield and let its native logic run.
    """
    try:
        import nltk.pathsec

        original_enforce = nltk.pathsec.ENFORCE
        nltk.pathsec.ENFORCE = True
        yield
        nltk.pathsec.ENFORCE = original_enforce
    except ImportError:
        # We are on a branch without pathsec.py, proceed normally
        yield


def test_valid_http_url():
    """Ensure standard, safe external URLs pass through unobstructed."""
    dl = Downloader(
        server_index_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml"
    )
    try:
        dl.index()
    except URLError:
        pass  # URLError is fine (network issue), we just don't want a security block


def test_ssrf_invalid_scheme():
    """Ensure local file access via URL schemes is strictly blocked."""
    dl = Downloader(server_index_url="file:///etc/passwd")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


def test_ssrf_loopback_ip():
    """Block requests to the local machine."""
    dl = Downloader(server_index_url="http://127.0.0.1/admin")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


def test_ssrf_cloud_metadata_link_local():
    """Block requests to AWS/GCP/Azure Instance Metadata Services."""
    dl = Downloader(server_index_url="http://169.254.169.254/latest/meta-data/")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


def test_ssrf_ip_obfuscation():
    """
    Test if the security catches IP obfuscation techniques.
    2852039166 is the decimal representation of 169.254.169.254.
    If the branch uses simple string matching, it will attempt a network call
    (raising URLError/HTTPError) instead of safely blocking it (ValueError/PermissionError).
    """
    dl = Downloader(server_index_url="http://2852039166/latest/meta-data/")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()
