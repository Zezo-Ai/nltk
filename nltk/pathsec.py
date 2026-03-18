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
from functools import lru_cache
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen as _original_urlopen

# Security Enforcement Toggle
ENFORCE = False

_ALLOWED_ROOTS_CACHE = None
_LAST_DATA_PATHS = None

def _get_allowed_roots():
    global _ALLOWED_ROOTS_CACHE, _LAST_DATA_PATHS

    current_paths = []
    if "nltk.data" in sys.modules:
        current_paths = list(getattr(sys.modules["nltk.data"], "path", []))

    env_paths = os.environ.get("NLTK_DATA", "")
    current_state = (current_paths, env_paths)

    if _ALLOWED_ROOTS_CACHE is not None and _LAST_DATA_PATHS == current_state:
        return _ALLOWED_ROOTS_CACHE

    roots = set()

    # Resolve search paths (including any PathPointers)
    for p in current_paths + env_paths.split(os.sep):
        if p:
            try:
                raw_p = p.path if hasattr(p, "path") else p
                roots.add(Path(str(raw_p)).resolve())
            except:
                continue

    # Trust the NLTK library directory for internal .map/.tab files
    try:
        import nltk
        roots.add(Path(nltk.__file__).parent.resolve())
    except:
        pass

    # Trust standard data locations and the system TEMP directory
    import tempfile
    for loc in ["~/nltk_data", "/usr/share/nltk_data", tempfile.gettempdir()]:
        try:
            p = Path(loc).expanduser().resolve()
            if p.exists():
                roots.add(p)
        except:
            continue

    _ALLOWED_ROOTS_CACHE = roots
    _LAST_DATA_PATHS = current_state
    return roots

def validate_path(path_input, context="NLTK"):
    if isinstance(path_input, int) or not path_input or not str(path_input).strip():
        return
    try:
        # 1. Handle Pointers
        raw = path_input.path if hasattr(path_input, "path") else str(path_input)

        # 2. URL Handling
        if "://" in raw:
            parsed = urlparse(raw)
            if parsed.scheme in ("http", "https", "ftp"):
                return
            if parsed.scheme == "file":
                raw = unquote(parsed.path)

        # 3. ZIP TRANSPARENCY: Truncate to the archive file
        lower_raw = raw.lower()
        if ".zip" in lower_raw:
            zip_idx = lower_raw.find(".zip") + 4
            raw = raw[:zip_idx]

        target = Path(raw).resolve()

        # 4. Containment Check
        if any(target == root or target.is_relative_to(root) for root in _get_allowed_roots()):
            return

        # 5. CWD Fallback (Safety Valve)
        try:
            cwd = Path(os.getcwd()).resolve()
            if target == cwd or target.is_relative_to(cwd):
                warnings.warn(
                    f"Security Warning [{context}]: Path {target} allowed via CWD.",
                    RuntimeWarning,
                    stacklevel=3,
                )
                return
        except:
            pass

        msg = f"Security Violation [{context}]: Unauthorized path {target}"
        if ENFORCE:
            raise PermissionError(msg)
        else:
            warnings.warn(msg, RuntimeWarning, stacklevel=3)

    except (PermissionError, ValueError):
        raise
    except (OSError, TypeError) as e:
        if ENFORCE:
            raise PermissionError(f"Path validation failed [{context}]: {e}") from e

def validate_zip_archive(zip_obj_or_path, target_root, specific_member=None, context="ZipAudit"):
    """Enhanced Zip-Slip protection with null-byte detection."""
    try:
        target = Path(target_root).resolve()
        target_str = str(target)

        def _audit(zf):
            members_to_check = [specific_member] if specific_member is not None else zf.namelist()
            for name in members_to_check:
                name_str = name.filename if hasattr(name, "filename") else str(name)
                
                # Null-byte protection (from upstream)
                if "\0" in name_str:
                    raise ValueError(f"Null byte in ZIP member: {name_str}")

                # Path traversal check
                member_path_str = os.path.abspath(os.path.join(target_str, name_str))
                if not member_path_str.startswith(target_str + os.sep) and member_path_str != target_str:
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

def validate_network_url(url_input, context="NetworkIO"):
    """Hardened URL validation with SSRF protection and timeouts."""
    if not url_input or not str(url_input).strip():
        return
    try:
        parsed = urlparse(str(url_input))
        if parsed.scheme not in ("http", "https"):
            msg = f"Security Violation [{context}]: Invalid scheme '{parsed.scheme}'."
            if ENFORCE:
                raise PermissionError(msg)
            else:
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
            return

        hostname = parsed.hostname
        if not hostname:
            return

        # Upstream hardening: Explicit timeout for DNS resolution
        try:
            old_timeout = socket.getdefaulttimeout()
            socket.setdefaulttimeout(5)
            try:
                addr_info = socket.getaddrinfo(hostname, None)
            finally:
                socket.setdefaulttimeout(old_timeout)

            for result in addr_info:
                ip_str = result[4][0]
                ip_obj = ipaddress.ip_address(ip_str)

                # Block loopback, link-local, multicast, and private ranges
                if ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast or ip_obj.is_private:
                    msg = f"Security Violation [{context}]: Blocked SSRF attempt to {ip_str} ({hostname})"
                    if ENFORCE:
                        raise PermissionError(msg)
                    else:
                        warnings.warn(msg, RuntimeWarning, stacklevel=3)
        except (socket.gaierror, ValueError):
            pass
        except socket.timeout:
            if ENFORCE:
                raise PermissionError(f"Security Violation [{context}]: DNS resolution timed out for {hostname}")

    except PermissionError:
        raise
    except (OSError, ValueError) as e:
        if ENFORCE:
            raise PermissionError(f"URL validation failed [{context}]: {e}") from e

def open(file, mode="r", **kwargs):
    validate_path(file, context="pathsec.open")
    return builtins.open(file, mode=mode, **kwargs)

def urlopen(url, *args, **kwargs):
    validate_network_url(
        url.full_url if hasattr(url, "full_url") else str(url),
        context="pathsec.urlopen",
    )
    return _original_urlopen(url, *args, **kwargs)

class ZipFile(zipfile.ZipFile):
    def __init__(self, file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            validate_path(file, context="pathsec.ZipFile")
        super().__init__(file, *args, **kwargs)

    def extract(self, member, path=None, pwd=None):
        validate_zip_archive(self, path or os.getcwd(), specific_member=member, context="pathsec.ZipFile.extract")
        return super().extract(member, path, pwd)

    def extractall(self, path=None, members=None, pwd=None):
        validate_zip_archive(self, path or os.getcwd(), context="pathsec.ZipFile.extractall")
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
