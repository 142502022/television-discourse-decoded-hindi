"""V2 pipeline: media acquisition and preprocessing.

Importing this package ensures Python-level HTTPS uses certifi's CA bundle.
The python.org macOS Python ships with no usable CA store; setting the
standard TLS environment variables here fixes downloads made via urllib,
requests, torch.hub, and Transformers. Idempotent and overridable.
"""

import os

__version__ = "0.1.0"


def _ensure_ca_bundle() -> None:
    try:
        import certifi
    except ImportError:
        return
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        os.environ.setdefault(var, certifi.where())


_ensure_ca_bundle()