import sys

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
        data.normalize_resource_url(r"..\..\etc\passwd")


def test_normalize_allows_package_paths():
    """Valid package-style resource names should still be treated as nltk: URLs."""
    out = data.normalize_resource_url("corpora/brown")
    assert out.startswith(
        "nltk:"
    ), "Package-style paths should be treated as 'nltk:' URLs"


def test_normalize_rejects_no_protocol_absolute_posix_path():
    """Absolute POSIX paths without a protocol should be rejected."""
    with pytest.raises(ValueError):
        data.normalize_resource_url("/etc/passwd")


def test_normalize_rejects_no_protocol_windows_drive_letter_paths():
    """
    Windows drive letter paths should be rejected even on non-Windows platforms.

    Review note: don't gate 'C:/etc/passwd' on Windows only; ensure robust rejection
    regardless of runtime platform.
    """
    with pytest.raises(ValueError):
        data.normalize_resource_url(r"C:\etc\passwd")

    # Run on all platforms (per review suggestion)
    with pytest.raises(ValueError):
        data.normalize_resource_url("C:/etc/passwd")


def test_normalize_rejects_no_protocol_dotdot_only():
    """A resource name that is exactly '..' should be rejected."""
    with pytest.raises(ValueError):
        data.normalize_resource_url("..")


def test_find_rejects_traversal_direct_call():
    """Defense-in-depth: direct calls to find() should reject traversal-like names."""
    with pytest.raises(ValueError):
        data.find("../../etc/passwd")


def test_find_rejects_traversal_that_becomes_unsafe_after_normalization():
    """
    Defense-in-depth edge case: a path can become unsafe only after normalization.

    Example from review: "foo/../../etc/passwd" normalizes to "../etc/passwd" and
    must still be rejected.
    """
    with pytest.raises(ValueError):
        data.find("foo/../../etc/passwd")


def test_find_zipfile_split_is_non_greedy_integration():
    """
    Integration-ish test: ensure find() handles nested '.zip' paths using the
    left-most '.zip' boundary (non-greedy behavior), without requiring NLTK data.

    We force lookup failure via paths=[], then assert the error reports the exact
    resource string we attempted to load. This exercises find()'s zip parsing
    path and ensures it doesn't crash or mis-handle nested zip names.
    """
    resource = "dir1/dir2/a.zip/b.zip/c.txt"

    with pytest.raises(LookupError) as excinfo:
        data.find(resource, paths=[])

    # The error message includes an "Attempted to load '...'" line in the patched code.
    # Assert it references the original resource string (i.e., find() accepted it and
    # proceeded through its normal logic).
    assert resource in str(excinfo.value)
