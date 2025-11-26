import os
import sys
import zipfile
from pathlib import Path

import pytest

from nltk.downloader import ErrorMessage, _unzip_iter


def _make_zip(file_path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(file_path, "w") as zf:
        for arcname, content in members.items():
            zf.writestr(arcname, content)


def _run_unzip_iter(zip_path: Path, extract_root: Path, verbose: bool = False):
    return list(_unzip_iter(str(zip_path), str(extract_root), verbose=verbose))


class TestSecureUnzip:
    """
    Tests for secure ZIP extraction behaviour in nltk.downloader._unzip_iter.

    These tests validate that classic Zip-Slip primitives (.., absolute
    paths) are blocked, and that bad zipfiles are handled gracefully.

    A stronger defence against pre-existing symlinks is desirable for
    NLTK, but is currently tracked as an expected failure (xfail) test
    below.
    """

    def test_normal_relative_paths_are_extracted(self, tmp_path: Path):
        zip_path = tmp_path / "safe.zip"
        extract_root = tmp_path / "extract"

        members = {
            "pkg/file.txt": b"hello",
            "pkg/subdir/other.txt": b"world",
        }
        _make_zip(zip_path, members)

        messages = _run_unzip_iter(zip_path, extract_root, verbose=False)
        assert not any(isinstance(m, ErrorMessage) for m in messages)

        assert (extract_root / "pkg" / "file.txt").read_bytes() == b"hello"
        assert (extract_root / "pkg" / "subdir" / "other.txt").read_bytes() == b"world"

    def test_zip_slip_with_parent_directory_component_is_blocked(self, tmp_path: Path):
        """
        An entry containing ``..`` that would escape the target directory
        should not be written outside the extraction root.
        """
        zip_path = tmp_path / "zip_slip_parent.zip"
        extract_root = tmp_path / "extract"
        outside_target = tmp_path / "outside.txt"

        members = {
            "pkg/good.txt": b"ok",
            # This would escape extract_root if not validated.
            "../outside.txt": b"evil",
        }
        _make_zip(zip_path, members)

        messages = _run_unzip_iter(zip_path, extract_root, verbose=False)

        # Nothing must be written outside the extraction root.
        assert not outside_target.exists()
        assert not (extract_root / ".." / "outside.txt").exists()

        # Safe entry should still be extracted.
        assert (extract_root / "pkg" / "good.txt").read_bytes() == b"ok"

        # ErrorMessage is allowed but not required; the key property is that
        # the write outside the root did not occur.

    @pytest.mark.skipif(
        sys.platform.startswith("win"),
        reason="Absolute POSIX paths are not meaningful on Windows",
    )
    def test_zip_slip_with_absolute_posix_path_is_blocked(self, tmp_path: Path):
        """
        An entry with an absolute POSIX path (e.g. ``/tmp/evil``) must not be
        extracted as-is; it should not overwrite arbitrary filesystem paths.
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

            # Absolute path must not be created.
            assert not absolute_target.exists()

            # Safe entry must be extracted under extract_root.
            assert (extract_root / "pkg" / "good.txt").read_bytes() == b"ok"
        finally:
            # Best-effort cleanup if the implementation under test behaves
            # incorrectly and creates the file.
            if absolute_target.exists():
                try:
                    absolute_target.unlink()
                except OSError:
                    pass

    @pytest.mark.xfail(
        reason=(
            "Current implementation does not prevent writes via pre-existing "
            "symlinks inside the extraction root. Hardening NLTK's downloader "
            "against this more advanced escape vector is desirable but not yet "
            "implemented."
        )
    )
    def test_entries_resolved_outside_root_are_blocked_via_symlink(
        self, tmp_path: Path
    ):
        """
        DESIRED (but currently not enforced) behaviour:

        If there is a pre-existing symlink below the extraction root that
        points outside the root, writing through that symlink should not
        be allowed to escape the root.

        This test documents the desired hardening behaviour and is marked
        as xfail until _unzip_iter is tightened to defend against this
        class of attacks.
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
        # Pre-existing symlink inside extract_root pointing *outside* it.
        os.symlink(outside_dir, extract_root / "dir_link")

        _run_unzip_iter(zip_path, extract_root, verbose=False)

        # Desired property (not currently met by the implementation):
        assert not outside_target.exists()
        assert (extract_root / "pkg" / "good.txt").read_bytes() == b"ok"

    def test_bad_zipfile_yields_errormessage(self, tmp_path: Path):
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

    def test_unzip_iter_verbose_writes_to_stdout(self, capsys, tmp_path: Path):
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
