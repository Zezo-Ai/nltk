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
import urllib.request
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse

# Security Enforcement Toggle
ENFORCE = False

_ALLOWED_ROOTS_CACHE = None
_LAST_DATA_PATHS = None


def _get_allowed_roots():
    """Dynamically determines allowed directories based on NLTK data paths."""
    global _ALLOWED_ROOTS_CACHE, _LAST_DATA_PATHS

    current_paths = []
    if "nltk.data" in sys.modules:
        current_paths = list(getattr(sys.modules["nltk.data"], "path", []))

    env_paths = os.environ.get("NLTK_DATA", "")
    current_state = (current_paths, env_paths)

    if _ALLOWED_ROOTS_CACHE is not None and _LAST_DATA_PATHS == current_state:
        return _ALLOWED_ROOTS_CACHE

    roots = set()
    # FIX: Use os.pathsep for environment variables (Copilot High)
    for p in current_paths + env_paths.split(os.pathsep):
        if p:
            try:
                raw_p = p.path if hasattr(p, "path") else p
                roots.add(Path(str(raw_p)).resolve())
            except Exception:
                continue

    import tempfile

    for loc in ["~/nltk_data", "/usr/share/nltk_data", tempfile.gettempdir()]:
        try:
            p = Path(loc).expanduser().resolve()
            if p.exists():
                roots.add(p)
        except Exception:
            continue

    _ALLOWED_ROOTS_CACHE = roots
    _LAST_DATA_PATHS = current_state
    return roots


def validate_path(path_input, context="NLTK"):
    """Ensures file access is restricted to allowed data directories."""
    if isinstance(path_input, int) or not path_input or not str(path_input).strip():
        return
    try:
        raw = path_input.path if hasattr(path_input, "path") else str(path_input)

        if "://" in raw:
            parsed = urlparse(raw)
            if parsed.scheme in ("http", "https", "ftp"):
                return
            if parsed.scheme == "file":
                raw = urllib.request.url2pathname(parsed.path)

        target = Path(raw).resolve()

        allowed_roots = _get_allowed_roots()
        if any(target == root or target.is_relative_to(root) for root in allowed_roots):
            return

        # 5. CWD Fallback (Explicit Opt-In for ENFORCE mode)
        try:
            cwd = Path(os.getcwd()).resolve()
            if target == cwd or target.is_relative_to(cwd):
                if any(cwd == root for root in allowed_roots):
                    return

                msg = (
                    f"Security Violation [{context}]: CWD access is restricted in ENFORCE mode. "
                    "To allow local data, use: nltk.data.path.append('.')"
                )

                if ENFORCE:
                    raise PermissionError(msg)
                else:
                    warnings.warn(
                        f"Security Warning [{context}]: Path {target} allowed via CWD.",
                        RuntimeWarning,
                        stacklevel=3,
                    )
                    return
        except Exception:
            pass

        msg = f"Security Violation [{context}]: Unauthorized path {target}"
        if ENFORCE:
            raise PermissionError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)
    except (PermissionError, ValueError):
        raise
    except Exception as e:
        if ENFORCE:
            raise PermissionError(f"Path validation failed [{context}]: {e}")


def validate_zip_archive(
    zip_obj_or_path, target_root, specific_member=None, context="ZipAudit"
):
    """Enhanced Zip-Slip protection with null-byte detection."""
    try:
        target = Path(target_root).resolve()
        target_str = str(target)
        # Normalize target paths for cross-platform, case-insensitive comparison
        target_norm_eq = os.path.normcase(target_str)
        # Ensure trailing separator for prefix check (e.g., 'C:\\data\\')
        target_norm_prefix = os.path.normcase(os.path.join(target_str, ""))

        def _audit(zf):
            members_to_check = (
                [specific_member] if specific_member is not None else zf.namelist()
            )
            for name in members_to_check:
                name_str = name.filename if hasattr(name, "filename") else str(name)
                if "\0" in name_str:
                    raise ValueError(f"Null byte in ZIP member: {name_str}")
                member_path_str = os.path.abspath(os.path.join(target_str, name_str))
                member_norm = os.path.normcase(member_path_str)
                if not (
                    member_norm.startswith(target_norm_prefix)
                    or member_norm == target_norm_eq
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
    except (PermissionError, ValueError):
        raise
    except (OSError, zipfile.BadZipFile) as e:
        if ENFORCE:
            raise PermissionError(f"Zip validation failed [{context}]: {e}") from e


@lru_cache(maxsize=256)
def _resolve_hostname(hostname):
    """Cached hostname resolution to mitigate DNS rebinding (Copilot Medium)."""
    try:
        return socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    except Exception:
        return []


def validate_network_url(url_input, context="NetworkIO"):
    """Hardened URL validation with SSRF protection."""
    if not url_input or not str(url_input).strip():
        return
    try:
        parsed = urlparse(str(url_input))

        # FIX: Cross-route file scheme to path validation (Copilot High)
        if parsed.scheme == "file":
            validate_path(
                urllib.request.url2pathname(parsed.path),
                context=f"{context}.file_scheme",
            )
            return

        if parsed.scheme not in ("http", "https"):
            msg = (
                f"Security Violation [{context}]: Unsupported scheme '{parsed.scheme}'."
            )
            if ENFORCE:
                raise PermissionError(msg)
            else:
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            return

        for result in _resolve_hostname(parsed.hostname or ""):
            ip = ipaddress.ip_address(result[4][0])
            if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_private:
                msg = f"Security Violation [{context}]: Blocked SSRF attempt to restricted IP {ip}"
                if ENFORCE:
                    raise PermissionError(msg)
                else:
                    warnings.warn(msg, RuntimeWarning, stacklevel=3)
    except (PermissionError, ValueError):
        raise
    except Exception as e:
        if ENFORCE:
            raise PermissionError(f"URL validation failed [{context}]: {e}")


class _ValidatingRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Ensures that every step of a redirect chain is re-validated against SSRF (Copilot High)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        validate_network_url(newurl, context="NetworkRedirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def urlopen(url, *args, **kwargs):
    """Secure wrapper for urllib.request.urlopen with redirect validation."""
    url_str = url.full_url if hasattr(url, "full_url") else str(url)
    validate_network_url(url_str, context="pathsec.urlopen")
    urllib.request.install_opener(
        urllib.request.build_opener(_ValidatingRedirectHandler())
    )
    return urllib.request.urlopen(url, *args, **kwargs)


def open(file, mode="r", **kwargs):
    """Secure wrapper for builtins.open."""
    validate_path(file, context="pathsec.open")
    return builtins.open(file, mode=mode, **kwargs)


class ZipFile(zipfile.ZipFile):
    """Secure wrapper for zipfile.ZipFile."""

    def __init__(self, file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            validate_path(file, context="pathsec.ZipFile")
        super().__init__(file, *args, **kwargs)

    def extract(self, member, path=None, pwd=None):
        validate_zip_archive(
            self,
            path or os.getcwd(),
            specific_member=member,
            context="pathsec.ZipFile.extract",
        )
        return super().extract(member, path, pwd)

    def extractall(self, path=None, members=None, pwd=None):
        validate_zip_archive(
            self, path or os.getcwd(), context="pathsec.ZipFile.extractall"
        )
        super().extractall(path, members, pwd)


__all__ = [
    "validate_path",
    "validate_network_url",
    "validate_zip_archive",
    "open",
    "urlopen",
    "ZipFile",
    "ENFORCE",
]
