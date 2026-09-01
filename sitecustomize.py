"""Local Python startup customizations for this project.

Some Windows/Anaconda installations can contain malformed certificates in the
Windows certificate store. Tornado/Streamlit creates a default SSL context at
import time, and Python may fail with::

    ssl.SSLError: [ASN1: NOT_ENOUGH_DATA] not enough data

This module is imported automatically by Python when the project root is on
``PYTHONPATH``/the current working directory. It points SSL-aware libraries to
Certifi's maintained CA bundle before Streamlit/Tornado imports SSL contexts.
"""

from __future__ import annotations

import os
import ssl


def _configure_certifi_ca_bundle() -> None:
    """Prefer Certifi's CA bundle for HTTPS verification when available."""

    try:
        import certifi
    except Exception:
        return

    ca_bundle = certifi.where()
    os.environ.setdefault("SSL_CERT_FILE", ca_bundle)
    os.environ.setdefault("REQUESTS_CA_BUNDLE", ca_bundle)


def _patch_windows_default_cert_loading() -> None:
    """Prevent malformed Windows certificate store entries from breaking startup.

    Some Anaconda/Windows environments raise ``ssl.SSLError`` while importing
    Streamlit/Tornado because ``SSLContext.load_default_certs`` consults the
    Windows trust store at import time. If that trust store contains malformed
    certificates, Python aborts before the app can start.

    We keep normal certificate handling intact but gracefully ignore the
    specific certificate-store parsing error so libraries can fall back to the
    Certifi bundle configured above.
    """

    original_load_default_certs = ssl.SSLContext.load_default_certs

    def _safe_load_default_certs(self: ssl.SSLContext, purpose: ssl.Purpose = ssl.Purpose.SERVER_AUTH) -> None:  # type: ignore[override]
        try:
            original_load_default_certs(self, purpose)
        except ssl.SSLError as exc:
            if "NOT_ENOUGH_DATA" in str(exc):
                return
            raise

    ssl.SSLContext.load_default_certs = _safe_load_default_certs  # type: ignore[assignment]


def _configure_openmp_runtime() -> None:
    """Prevent common OpenMP startup conflicts on Windows.

    Some scientific Python stacks pull in multiple native libraries that each
    bundle Intel OpenMP runtime binaries. On Windows this can abort Streamlit
    startup with ``OMP: Error #15`` before the UI is rendered.

    ``KMP_DUPLICATE_LIB_OK`` is an accepted workaround for local development
    environments where we prefer app startup over a hard crash.
    """

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


_configure_certifi_ca_bundle()
_patch_windows_default_cert_loading()
_configure_openmp_runtime()