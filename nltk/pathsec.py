import builtins
import ipaddress
import os
import socket
import warnings
import zipfile
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen as _original_urlopen

ENFORCE = False
_ALLOWED_CACHE = None


def _get_allowed_roots():
    global _ALLOWED_CACHE
    if _ALLOWED_CACHE is not None:
        return _ALLOWED_CACHE

    roots = set()
    for p in os.environ.get("NLTK_DATA", "").split(os.pathsep):
        if p:
            try:
                roots.add(Path(p).resolve())
            except Exception:
                continue

    standard_locs = [
        "~/nltk_data",
        "/usr/share/nltk_data",
        "/usr/lib/nltk_data",
        os.getcwd(),
    ]
    for loc in standard_locs:
        try:
            p = Path(loc).expanduser().resolve()
            if p.exists():
                roots.add(p)
        except Exception:
            continue

    import tempfile

    try:
        roots.add(Path(tempfile.gettempdir()).resolve())
    except Exception:
        pass

    _ALLOWED_CACHE = roots
    return roots


def validate_path(path_input, context="NLTK"):
    if not path_input or not str(path_input).strip():
        return

    try:
        raw = str(path_input.path if hasattr(path_input, "path") else path_input)
        if "://" in raw:
            parsed = urlparse(raw)
            if parsed.scheme == "file":
                raw = unquote(parsed.path)
            else:
                return  # Network URLs are handled by validate_network_url

        lower_raw = raw.lower()
        if ".zip" in lower_raw:
            zip_idx = lower_raw.rfind(".zip") + 4
            target = Path(raw[:zip_idx]).resolve()
        else:
            target = Path(raw).resolve()

        allowed = _get_allowed_roots()
        if not any(target == root or root in target.parents for root in allowed):
            msg = f"Security Violation [{context}]: Unauthorized path {target}"
            if ENFORCE:
                raise PermissionError(msg)
            else:
                warnings.warn(msg, RuntimeWarning, stacklevel=3)
    except Exception:
        if ENFORCE:
            raise


def validate_zip_archive(zip_input, target_root, context="ZipAudit"):
    try:
        target = Path(target_root).resolve()
        with zipfile.ZipFile(zip_input, "r") as zf:
            for name in zf.namelist():
                if "\0" in name:
                    raise ValueError(f"Null byte in ZIP member: {name}")

                member_path = (target / name).resolve()
                if not str(member_path).startswith(str(target)):
                    msg = f"Security Violation [{context}]: Traversal member '{name}' detected."
                    if ENFORCE:
                        raise PermissionError(msg)
                    else:
                        warnings.warn(msg, RuntimeWarning, stacklevel=3)
    except Exception:
        if ENFORCE:
            raise


def validate_network_url(url_input, context="NetworkIO"):
    """Validates remote URLs to prevent SSRF while allowing legitimate internal mirrors."""
    if not url_input or not str(url_input).strip():
        return

    try:
        parsed = urlparse(str(url_input))

        # 1. Scheme Enforcement
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

        # 2. DNS Resolution
        try:
            ip_str = socket.gethostbyname(hostname)
            ip_obj = ipaddress.ip_address(ip_str)
        except socket.gaierror:
            return

        # 3. Targeted IP Blacklisting
        if ip_obj.is_loopback or ip_obj.is_link_local or ip_obj.is_multicast:
            msg = f"Security Violation [{context}]: Blocked SSRF attempt to restricted IP {ip_str} ({hostname})"
            if ENFORCE:
                raise PermissionError(msg)
            else:
                warnings.warn(msg, RuntimeWarning, stacklevel=3)

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
    """
    Intercepts built-in open() to enforce filesystem security boundaries.
    Acts as a centralized guard against path traversal just before disk access.
    """
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
    """
    Intercepts network requests to enforce SSRF protection.
    """
    url_string = url.full_url if hasattr(url, "full_url") else str(url)
    validate_network_url(url_string, context="pathsec.urlopen")
    return _original_urlopen(url, *args, **kwargs)
