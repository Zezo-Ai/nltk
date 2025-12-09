import re

import pytest

import nltk.data as data


def test_normalize_rejects_no_protocol_traversal():
    """No-protocol traversal sequences should be rejected."""
    with pytest.raises(ValueError):
        data.normalize_resource_url("../../etc/passwd")

    with pytest.raises(ValueError):
        data.normalize_resource_url("../relative/../etc/passwd")


def test_normalize_rejects_no_protocol_backslashes():
    """Windows-style backslash traversal should be rejected when no protocol is present."""
    with pytest.raises(ValueError):
        data.normalize_resource_url("..\\..\\etc\\passwd")


def test_normalize_allows_package_paths():
    """Valid package-style resource names should still be treated as nltk: URLs."""
    out = data.normalize_resource_url("corpora/brown")
    assert out.startswith(
        "nltk:"
    ), "Package-style paths should be treated as 'nltk:' URLs"


def test_find_rejects_traversal_direct_call():
    """Defense-in-depth: direct calls to find() should reject traversal-like names."""
    with pytest.raises(ValueError):
        data.find("../../etc/passwd")


def _match_zip_non_greedy(resource_name):
    """Pattern we expect the implementation to use: first (left-most) .zip captured."""
    return re.match(r"(.*?\.zip)/?(.*)$", resource_name)


def test_zipfile_regex_captures_first_zip_in_nested_paths():
    resource = "dir1/dir2/a.zip/b.zip/c.txt"
    m = _match_zip_non_greedy(resource)
    assert m is not None
    zipfile, zipentry = m.groups()
    assert zipfile == "dir1/dir2/a.zip"
    assert zipentry == "b.zip/c.txt"


def test_zipfile_regex_single_zip_case():
    resource = "corpora/chat80.zip/chat80/cities.pl"
    m = _match_zip_non_greedy(resource)
    assert m is not None
    zipfile, zipentry = m.groups()
    assert zipfile == "corpora/chat80.zip"
    assert zipentry == "chat80/cities.pl"


def test_zipfile_regex_no_zip_returns_none():
    resource = "corpora/chat80/cities.pl"
    m = _match_zip_non_greedy(resource)
    assert m is None
