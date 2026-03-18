# Natural Language Toolkit: Centralized I/O security sentinel
#
# Copyright (C) 2001-2026 NLTK Project
# Author: Eric Kafe <kafe.eric@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#

"""Centralized I/O security sentinel for NLTK."""
import builtins
import ipaddress
import os
import socket
import sys
import warnings
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen as _original_urlopen

ENFORCE = False

# Cache for expensive Path.resolve() operations
_ALLOWED_ROOTS_CACHE = None
_LAST_DATA_PATHS = None


def _get_allowed_roots():
    global _ALLOWED_ROOTS_CACHE, _LAST_DATA_PATHS

    current_paths = []
    if "nltk.data" in sys.modules:
        current_paths = list(getattr(sys.modules["nltk.data"], "path", []))

    env_paths = os.environ.get("NLTK_DATA", "")

    # Cache invalidation: Check if NLTK data paths or env vars have changed at runtime
    current_state = (current_paths, env_paths)
    if _ALLOWED_ROOTS_CACHE is not None and _LAST_DATA_PATHS == current_state:
        return _ALLOWED_ROOTS_CACHE

    roots = set()

    # 1. Dynamic check of runtime NLTK data paths
    for p in current_paths:
        try:
            roots.add(Path(p).resolve())
        except Exception:
            continue

    # 2. Environment variables
    for p in env_paths.split(os.pathsep):
        if p:
            try:
                roots.add(Path(p).resolve())
            except Exception:
                continue

    # 3. Standard locations (Note: os.getcwd() removed from standard roots)
    standard_locs = ["~/nltk_data", "/usr/share/nltk_data", "/usr/lib/nltk_data"]
    for loc in standard_locs:
        try:
            p = Path(loc).expanduser().resolve()
            if p.exists():
                roots.add(p)
        except Exception:
            continue

    # 4. System temp dir
    import tempfile

    try:
        roots.add(Path(tempfile.gettempdir()).resolve())
    except Exception:
        pass

    _ALLOWED_ROOTS_CACHE = roots
    _LAST_DATA_PATHS = current_state
    return roots


def validate_path(path_input, context="NLTK"):
    # Short-circuit for integer file descriptors (e.g., open(3))
    if isinstance(path_input, int) or not path_input or not str(path_input).strip():
        return

    try:
        raw_source = path_input.path if hasattr(path_input, "path") else path_input
        try:
            raw = os.fspath(raw_source)
        except TypeError:
            raw = str(raw_source)

        if "://" in raw:
            parsed = urlparse(raw)
            # Network URLs handled elsewhere; allow valid URL schemes to bypass path checks
            if parsed.scheme in ("http", "https", "ftp"):
                return
            elif parsed.scheme == "file":
                raw = unquote(parsed.path)
            # Windows drive letters (C://) will fall through to normal path validation

        lower_raw = raw.lower()
        if ".zip" in lower_raw:
            zip_idx = lower_raw.rfind(".zip") + 4
            target = Path(raw[:zip_idx]).resolve()
        else:
            target = Path(raw).resolve()

        allowed = _get_allowed_roots()

        # Check standard secure roots first
        if any(target == root or root in target.parents for root in allowed):
            return

        # Fallback: check CWD to prevent breaking local student scripts
        try:
            cwd = Path(os.getcwd()).resolve()
            if target == cwd or cwd in target.parents:
                warnings.warn(
                    f"Security Warning [{context}]: Path {target} allowed via current working directory. "
                    "Relying on CWD is insecure for web/server environments. Please configure NLTK_DATA.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return
        except Exception:
            pass

        # If it fails standard roots AND cwd
        msg = f"Security Violation [{context}]: Unauthorized path {target}"
        if ENFORCE:
            raise PermissionError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    except Exception:
        if ENFORCE:
            raise


def validate_zip_archive(
    zip_obj_or_path, target_root, specific_member=None, context="ZipAudit"
):
    try:
        target = Path(target_root).resolve()
        target_str = str(target)

        def _audit(zf):
            # If a specific member is provided, only check that one. Otherwise, check all.
            members_to_check = (
                [specific_member] if specific_member is not None else zf.namelist()
            )

            for name in members_to_check:
                name_str = name.filename if hasattr(name, "filename") else str(name)
                if "\0" in name_str:
                    raise ValueError(f"Null byte in ZIP member: {name_str}")

                # Fast, in-memory path math
                member_path_str = os.path.abspath(os.path.join(target_str, name_str))

                if (
                    not member_path_str.startswith(target_str + os.sep)
                    and member_path_str != target_str
                ):
                    msg = f"Security Violation [{context}]: Traversal member '{name_str}' detected."
                    if ENFORCE:
                        raise PermissionError(msg)
                    else:
                        warnings.warn(msg, RuntimeWarning, stacklevel=3)

        if isinstance(zip_obj_or_path, zipfile.ZipFile):
            _audit(zip_obj_or_path)
        else:
            with zipfile.ZipFile(zip_obj_or_path, "r") as zf:
                _audit(zf)
    except Exception:
        if ENFORCE:
            raise


def validate_network_url(url_input, context="NetworkIO"):
    """Validates remote URLs to prevent SSRF while allowing legitimate internal mirrors."""
    if not url_input or not str(url_input).strip():
        return

    try:
        parsed = urlparse(str(url_input))
        if parsed.scheme not in ("http", "https"):
            msg = f"Security Violation [{context}]: Invalid scheme '{parsed.scheme}'. Only http/https allowed."
            if ENFORCE:
                raise PermissionError(msg)
            else:
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            return

        hostname = parsed.hostname
        if not hostname:
            return

        # Use getaddrinfo to capture both IPv4 and IPv6 resolutions
        try:
            addr_info = socket.getaddrinfo(hostname, None)
            for result in addr_info:
                ip_str = result[4][0]
                ip_obj = ipaddress.ip_address(ip_str)

                # Block loopback, link-local, multicast, and private ranges
                if (
                    ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_private
                ):
                    msg = f"Security Violation [{context}]: Blocked SSRF attempt to restricted IP {ip_str} ({hostname})"
                    if ENFORCE:
                        raise PermissionError(msg)
                    else:
                        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        except (socket.gaierror, ValueError):
            pass

    except Exception:
        if ENFORCE:
            raise


# --- CENTRALIZED I/O WRAPPERS ---


def open(
    file,
    mode="r",
    buffering=-1,
    encoding=None,
    errors=None,
    newline=None,
    closefd=True,
    opener=None,
):
    validate_path(file, context="pathsec.open")
    return builtins.open(
        file,
        mode=mode,
        buffering=buffering,
        encoding=encoding,
        errors=errors,
        newline=newline,
        closefd=closefd,
        opener=opener,
    )


def urlopen(url, *args, **kwargs):
    url_string = url.full_url if hasattr(url, "full_url") else str(url)
    validate_network_url(url_string, context="pathsec.urlopen")
    return _original_urlopen(url, *args, **kwargs)


class ZipFile(zipfile.ZipFile):
    def extractall(self, path=None, members=None, pwd=None):
        target = path if path is not None else os.getcwd()
        # If specific members are passed, check only those. Otherwise, check the whole archive.
        if members is not None:
            for member in members:
                validate_zip_archive(
                    self,
                    target,
                    specific_member=member,
                    context="pathsec.ZipFile.extractall",
                )
        else:
            validate_zip_archive(self, target, context="pathsec.ZipFile.extractall")

        super().extractall(path, members, pwd)

    def extract(self, member, path=None, pwd=None):
        target = path if path is not None else os.getcwd()
        # ONLY audit the specific member being extracted
        validate_zip_archive(
            self, target, specific_member=member, context="pathsec.ZipFile.extract"
        )
        return super().extract(member, path, pwd)


__all__ = [
    "validate_path",
    "validate_zip_archive",
    "validate_network_url",
    "open",
    "urlopen",
    "ZipFile",
    "ENFORCE",
]
