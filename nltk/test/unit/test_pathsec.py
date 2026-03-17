import io
import os
import zipfile
from urllib.error import URLError

import pytest

from nltk.downloader import Downloader


@pytest.fixture(autouse=True)
def enable_enforcement():
    """
    Dynamically toggle enforcement if pathsec exists.
    If on a branch without pathsec.py, proceed normally to test legacy logic.
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
    dl = Downloader(server_index_url="http://2852039166/latest/meta-data/")
    with pytest.raises((ValueError, PermissionError)):
        dl.index()


# --- PATH TRAVERSAL TESTS ---


def test_path_traversal_absolute():
    """Test if absolute paths bypass standard relative traversal checks."""
    try:
        from nltk.pathsec import open as secure_open

        with pytest.raises((ValueError, PermissionError)):
            secure_open("/etc/passwd", "r")
    except ImportError:
        pytest.skip("pathsec module not present on this branch")


def test_path_traversal_null_byte():
    """Test if null byte injections bypass extension/string checks in standard open()."""
    try:
        from nltk.pathsec import open as secure_open

        with pytest.raises((ValueError, PermissionError)):
            secure_open("safe_corpus.txt\0../../../etc/passwd", "r")
    except ImportError:
        pytest.skip("pathsec module not present on this branch")


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
    """Test standard ../ Zip-Slip traversal."""
    try:
        from nltk.pathsec import ZipFile as SecureZipFile

        malicious_zip = create_malicious_zip("../../../evil.sh")
        with pytest.raises((ValueError, PermissionError)):
            with SecureZipFile(malicious_zip, "r") as zf:
                zf.extractall("/tmp/nltk_extract")
    except ImportError:
        pytest.skip("pathsec module not present on this branch")


def test_zip_slip_absolute_path():
    """Test Zip-Slip using an absolute path (bypasses simple ../ checks)."""
    try:
        from nltk.pathsec import ZipFile as SecureZipFile

        malicious_zip = create_malicious_zip("/etc/cron.d/evil_cron")
        with pytest.raises((ValueError, PermissionError)):
            with SecureZipFile(malicious_zip, "r") as zf:
                zf.extractall("/tmp/nltk_extract")
    except ImportError:
        pytest.skip("pathsec module not present on this branch")
