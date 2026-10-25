"""Keep HTTPS verification enabled while allowing the OS trust store to be used.

FastF1 uses Requests (which normally relies on certifi) while the optional data
sources use urllib. On Windows, a locally installed root certificate may be in
the Windows trust store but absent from certifi. The merged bundle below makes
that trusted certificate available to both clients. Set F1_CA_BUNDLE to an
explicit PEM bundle when using a managed proxy or a custom CA.
"""
import base64
import os
import ssl
from pathlib import Path


_BUNDLE_VARS = ("REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE", "SSL_CERT_FILE")


def _validate_bundle(value):
    path = Path(os.path.expandvars(os.path.expanduser(value))).resolve()
    if not path.is_file():
        raise RuntimeError(f"Configured CA bundle does not exist: {path}")
    return str(path)


def _default_ca_bundle():
    try:
        import certifi
        path = Path(certifi.where())
        if path.is_file():
            return path
    except (ImportError, OSError):
        pass
    path = ssl.get_default_verify_paths().cafile
    return Path(path) if path and Path(path).is_file() else None


def _windows_ca_bundle(base):
    if os.name != "nt" or not hasattr(ssl, "enum_certificates"):
        return None
    try:
        roots = {}
        for store in ("ROOT", "CA"):
            for der, encoding, trust in ssl.enum_certificates(store):
                server_auth = trust is True or (bool(trust) and "1.3.6.1.5.5.7.3.1" in trust)
                if encoding != "x509_asn" or not server_auth:
                    continue
                encoded = base64.b64encode(der).decode("ascii")
                roots[der] = "-----BEGIN CERTIFICATE-----\n" + "\n".join(
                    encoded[i:i + 64] for i in range(0, len(encoded), 64)
                ) + "\n-----END CERTIFICATE-----\n"
    except (OSError, ssl.SSLError):
        return None
    if not roots:
        return None

    output = Path(os.environ.get("F1_TLS_CACHE_DIR", "cache/certs")) / "windows-ca-bundle.pem"
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        original = base.read_text(encoding="ascii") if base and base.is_file() else ""
        output.write_text(original.rstrip() + "\n" + "".join(roots.values()), encoding="ascii")
    except OSError:
        return None
    return output.resolve()


def configure_tls():
    """Configure Requests and urllib to use a shared, verified CA bundle."""
    explicit = os.environ.get("F1_CA_BUNDLE")
    if explicit:
        bundle = _validate_bundle(explicit)
        for name in _BUNDLE_VARS:
            os.environ[name] = bundle
        return bundle

    # Respect any conventional bundle configured by the user, then share it
    # with the client stack that does not already have its own setting.
    configured = next((os.environ[name] for name in _BUNDLE_VARS if os.environ.get(name)), None)
    if configured:
        bundle = _validate_bundle(configured)
        for name in _BUNDLE_VARS:
            os.environ.setdefault(name, bundle)
        return bundle

    bundle = _windows_ca_bundle(_default_ca_bundle())
    if bundle:
        for name in _BUNDLE_VARS:
            os.environ.setdefault(name, str(bundle))
        return str(bundle)
    return None
