import builtins
import io
import os
import zipfile
from urllib.error import URLError

import pytest

import nltk.downloader  # We will inspect this module directly
from nltk.downloader import Downloader


@pytest.fixture(autouse=True)
def enable_enforcement():
    """
    Dynamically toggle enforcement if pathsec exists.
    If on a branch without pathsec.py, proceed normally.
    """
    try:
        import nltk.pathsec

        original_enforce = nltk.pathsec.ENFORCE
        nltk.pathsec.ENFORCE = True
        yield
        nltk.pathsec.ENFORCE = original_enforce
    except ImportError:
        yield


# --- SSRF NETWORK TESTS ---


def test_valid_http_url():
    dl = Downloader(
        server_index_url="https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/index.xml"
    )
    try:
        dl.index()
    except URLError:
        pass


def test_ssrf_invalid_scheme():
    dl = Downloader(server_index_url="file:///etc/passwd")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


def test_ssrf_loopback_ip():
    dl = Downloader(server_index_url="http://127.0.0.1/admin")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


def test_ssrf_cloud_metadata_link_local():
    dl = Downloader(server_index_url="http://169.254.169.254/latest/meta-data/")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


def test_ssrf_ip_obfuscation():
    """Will FAIL on PR #3520 because string-matching misses the decimal IP."""
    dl = Downloader(server_index_url="http://2852039166/latest/meta-data/")
    # Added URLError to account for Windows DNS resolution failure
    with pytest.raises((ValueError, PermissionError, URLError)):
        dl.index()


# --- PATH TRAVERSAL TESTS ---


def test_path_traversal_absolute():
    """
    Test if absolute paths bypass standard relative traversal checks.
    Will FAIL on PR #3520 because standard builtins.open does not check path boundaries.
    """
    # Dynamically grab the 'open' function NLTK's downloader is currently using
    target_open = getattr(nltk.downloader, "open", builtins.open)

    with pytest.raises((ValueError, PermissionError)):
        target_open("/etc/passwd", "r")


# --- ZIP-SLIP TESTS ---


def create_malicious_zip(filename):
    """Helper to create malicious zip files in memory."""
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, "w") as zf:
        zinfo = zipfile.ZipInfo(filename)
        zf.writestr(zinfo, b"malicious content")
    mem_zip.seek(0)
    return mem_zip


def test_zip_slip_traversal():
    """
    Test standard ../ Zip-Slip traversal.
    Will FAIL on PR #3520 because standard zipfile silently sanitizes/ignores
    the traversal rather than proactively blocking it and raising an alert.
    """
    # Dynamically grab the 'ZipFile' class NLTK's downloader is currently using
    TargetZipFile = getattr(nltk.downloader, "ZipFile", zipfile.ZipFile)

    malicious_zip = create_malicious_zip("../../../evil.sh")
    with pytest.raises((ValueError, PermissionError)):
        with TargetZipFile(malicious_zip, "r") as zf:
            zf.extractall("/tmp/nltk_extract")


def test_zip_slip_absolute_path():
    """
    Test Zip-Slip using an absolute path.
    Will FAIL on PR #3520 because standard zipfile silently ignores the absolute
    root rather than proactively raising a security alert.
    """
    TargetZipFile = getattr(nltk.downloader, "ZipFile", zipfile.ZipFile)

    malicious_zip = create_malicious_zip("/etc/cron.d/evil_cron")
    with pytest.raises((ValueError, PermissionError)):
        with TargetZipFile(malicious_zip, "r") as zf:
            zf.extractall("/tmp/nltk_extract")
