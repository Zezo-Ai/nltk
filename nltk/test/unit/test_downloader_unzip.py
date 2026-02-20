import os
import sys
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

from nltk.downloader import ErrorMessage, _unzip_iter


def _make_zip(file_path: Path, members: dict[str, bytes]) -> None:
    """
    Create a ZIP file at file_path, with the given arcname->content mapping.
    """
    with zipfile.ZipFile(file_path, "w") as zf:
        for arcname, content in members.items():
            zf.writestr(arcname, content)


def _run_unzip_iter(zip_path: Path, extract_root: Path, verbose: bool = False):
    """
    Convenience wrapper that runs _unzip_iter and returns the list of yielded
    messages (if any).
    """
    return list(_unzip_iter(str(zip_path), str(extract_root), verbose=verbose))


class TestSecureUnzip:
    """
    Tests for the validate-then-extract strategy in ``_unzip_iter``.

    The implementation scans every member for security violations (path
    traversal, absolute paths, symlink escapes, null bytes) *before*
    extracting anything.  If any member fails validation the entire archive
    is rejected and nothing is written to disk.
    """

    def test_normal_relative_paths_are_extracted(self, tmp_path: Path) -> None:
        """
        A ZIP with only safe, relative paths should fully extract under the
        given root, and should not yield any ErrorMessage.
        """
        zip_path = tmp_path / "safe.zip"
        extract_root = tmp_path / "extract"

        members = {
            "pkg/file.txt": b"hello",
            "pkg/subdir/other.txt": b"world",
        }
        _make_zip(zip_path, members)

        messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

        # No ErrorMessage should be yielded for valid archives.
        assert not any(isinstance(m, ErrorMessage) for m in messages)

        assert (extract_root / "pkg" / "file.txt").read_bytes() == b"hello"
        assert (extract_root / "pkg" / "subdir" / "other.txt").read_bytes() == b"world"

    def test_zip_slip_with_parent_directory_component_is_blocked(
        self, tmp_path: Path
    ) -> None:
        """
        An entry containing ``..`` that would escape the target directory
        must not be written outside the extraction root, and must cause
        _unzip_iter to yield an ErrorMessage.

        The entire archive is rejected: even safe entries must NOT be
        extracted when any member fails validation.
        """
        zip_path = tmp_path / "zip_slip_parent.zip"
        extract_root = tmp_path / "extract"

        outside_target = (extract_root / ".." / "outside.txt").resolve()

        members = {
            "pkg/good.txt": b"ok",
            "../outside.txt": b"evil",
        }
        _make_zip(zip_path, members)

        messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

        err_msgs = [m for m in messages if isinstance(m, ErrorMessage)]
        assert (
            err_msgs
        ), "Expected an ErrorMessage for a Zip-Slip parent-directory attempt"

        combined_messages = " ".join(str(m.message) for m in err_msgs)
        assert "Zip Slip" in combined_messages and "blocked" in combined_messages

        assert not outside_target.exists()

        # Fail-fast: nothing should be extracted from a malicious archive.
        assert not (extract_root / "pkg" / "good.txt").exists()

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Absolute POSIX paths are not meaningful on Windows",
    )
    def test_zip_slip_with_absolute_posix_path_is_blocked(self, tmp_path: Path) -> None:
        """
        An entry with an absolute POSIX path (e.g. ``/tmp/evil``) must not be
        extracted as-is; it should not overwrite arbitrary filesystem paths,
        and should result in an ErrorMessage.

        The entire archive is rejected when any member fails validation.
        """
        zip_path = tmp_path / "zip_slip_abs_posix.zip"
        extract_root = tmp_path / "extract"

        absolute_target = Path("/tmp") / f"nltk_zip_slip_test_{os.getpid()}"

        try:
            members = {
                "pkg/good.txt": b"ok",
                str(absolute_target): b"evil",
            }
            _make_zip(zip_path, members)

            messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

            err_msgs = [m for m in messages if isinstance(m, ErrorMessage)]
            assert (
                err_msgs
            ), "Expected an ErrorMessage for absolute-path Zip-Slip attempt"

            combined_messages = " ".join(str(m.message) for m in err_msgs)
            assert "Zip Slip" in combined_messages and "blocked" in combined_messages

            assert not absolute_target.exists()

            # Fail-fast: nothing should be extracted from a malicious archive.
            assert not (extract_root / "pkg" / "good.txt").exists()
        finally:
            if absolute_target.exists():
                try:
                    absolute_target.unlink()
                except OSError:
                    pass

    def test_entries_resolved_outside_root_are_blocked_via_symlink(
        self, tmp_path: Path
    ) -> None:
        """
        If there is a pre-existing symlink below the extraction root that
        points outside the root, writing through that symlink should not
        be allowed to escape the root.

        The entire archive is rejected when any member fails validation.
        """
        if not hasattr(os, "symlink"):
            pytest.skip("Symlinks not supported on this platform")

        zip_path = tmp_path / "zip_slip_symlink.zip"
        extract_root = tmp_path / "extract"
        outside_dir = tmp_path / "outside_dir"
        outside_dir.mkdir()
        outside_target = outside_dir / "evil.txt"

        members = {
            "pkg/good.txt": b"ok",
            "dir_link/evil.txt": b"evil",
        }
        _make_zip(zip_path, members)

        extract_root.mkdir()
        os.symlink(outside_dir, extract_root / "dir_link")

        messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

        assert not outside_target.exists()
        assert any(isinstance(m, ErrorMessage) for m in messages)

        # Fail-fast: nothing should be extracted from a malicious archive.
        assert not (extract_root / "pkg" / "good.txt").exists()

    def test_bad_zipfile_yields_errormessage(self, tmp_path: Path) -> None:
        """
        A corrupt or non-zip file should cause _unzip_iter to yield an
        ErrorMessage instead of raising an unhandled exception.
        """
        zip_path = tmp_path / "not_a_zip.txt"
        zip_path.write_bytes(b"this is not a zip archive")
        extract_root = tmp_path / "extract"

        messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

        assert any(isinstance(m, ErrorMessage) for m in messages)

        # If the implementation chooses to create the root directory at all,
        # it should not leave partially extracted content.
        if extract_root.exists():
            assert not any(extract_root.iterdir())

    def test_null_byte_in_member_name_is_blocked(self, tmp_path: Path) -> None:
        """
        A member name containing a null byte must be rejected.  Null bytes
        can cause path truncation on some platforms, so they are never
        legitimate in archive entry names.

        The entire archive is rejected when any member fails validation.

        Note: CPython's zipfile module truncates names at null bytes on
        read, so we patch ``namelist()`` to simulate a library that
        preserves them.
        """
        zip_path = tmp_path / "null_byte.zip"
        extract_root = tmp_path / "extract"

        _make_zip(zip_path, {"pkg/good.txt": b"ok", "pkg/evil.txt": b"evil"})

        poisoned_names = ["pkg/good.txt", "pkg/evil\x00.txt"]

        with patch("zipfile.ZipFile.namelist", return_value=poisoned_names):
            messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

        err_msgs = [m for m in messages if isinstance(m, ErrorMessage)]
        assert err_msgs, "Expected an ErrorMessage for null-byte entry name"

        combined_messages = " ".join(str(m.message) for m in err_msgs)
        assert "Null byte" in combined_messages and "blocked" in combined_messages

        # Nothing should be extracted.
        assert not (extract_root / "pkg" / "good.txt").exists()

    def test_multiple_violation_types_all_reported_and_nothing_extracted(
        self, tmp_path: Path
    ) -> None:
        """
        An archive that combines several different violation types (path
        traversal and absolute path) must report every violation and
        extract nothing.  This verifies that the validation scan does not
        short-circuit after the first bad entry.
        """
        zip_path = tmp_path / "multi_violation.zip"
        extract_root = tmp_path / "extract"

        absolute_target = Path("/tmp") / f"nltk_multi_viol_test_{os.getpid()}"

        try:
            members = {
                "data/a.txt": b"aaa",
                "../traversal.txt": b"evil1",
                str(absolute_target): b"evil2",
                "data/b.txt": b"bbb",
            }
            _make_zip(zip_path, members)

            messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

            err_msgs = [m for m in messages if isinstance(m, ErrorMessage)]
            assert len(err_msgs) >= 2, (
                "Expected at least two ErrorMessages for multiple violations"
            )

            combined = " ".join(str(m.message) for m in err_msgs)
            assert "Zip Slip" in combined

            assert not absolute_target.exists()

            if extract_root.exists():
                assert not any(extract_root.iterdir())
        finally:
            if absolute_target.exists():
                try:
                    absolute_target.unlink()
                except OSError:
                    pass

    def test_unzip_iter_verbose_writes_to_stdout(self, capsys, tmp_path: Path) -> None:
        """
        When verbose=True, _unzip_iter should write a status line to stdout.
        This checks that existing user-visible behaviour is preserved.
        """
        zip_path = tmp_path / "verbose.zip"
        extract_root = tmp_path / "extract"

        members = {"pkg/file.txt": b"data"}
        _make_zip(zip_path, members)

        _run_unzip_iter(zip_path, extract_root, verbose=True)
        captured = capsys.readouterr()
        assert "Unzipping" in captured.out
