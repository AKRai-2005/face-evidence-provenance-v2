"""Deterministic canonical form for the evidence object.

The whole point: a stranger with `python -c` must be able to recompute our hash
from canonical.json and get the same digest we notarised, on a different OS,
Python build and CPU. Every rule below exists because some plausible
implementation would otherwise silently produce a different byte string.

Rules (SS6 of the brief):
  * UTF-8 throughout, Unicode NFC normalisation on every string.
  * Keys sorted lexicographically, recursively.
  * json.dumps(sort_keys=True, separators=(",", ":"), ensure_ascii=False).
  * NO FLOATS ANYWHERE. Similarity is stored in integer basis points.
  * Timestamps RFC 3339 UTC, second precision, trailing Z.
  * URLs: lowercase scheme and host, strip fragment, strip tracking params,
    preserve path case.
"""
from __future__ import annotations

import datetime as _dt
import unicodedata
import urllib.parse

SCHEMA = "hhgoa2026.task3.evidence.v1"

# Stripped because they vary per referral and would change the hash for what is
# demonstrably the same page. Observed live: Wikimedia's own API hands back URLs
# carrying utm_source/utm_campaign/utm_content.
TRACKING_PREFIXES = ("utm_",)
TRACKING_EXACT = {
    "fbclid", "gclid", "igshid", "mc_cid", "mc_eid", "msclkid",
    "ref", "ref_src", "ref_url", "spm", "yclid", "_ga", "s_kwcid",
}


class CanonicalizationError(ValueError):
    pass


def nfc(s: str) -> str:
    """NFC-normalise. Composed vs decomposed accents are visually identical but
    different bytes, and would produce different hashes for the same title."""
    return unicodedata.normalize("NFC", s)


def rfc3339(ts: _dt.datetime | int | float) -> str:
    """UTC, second precision, trailing Z. Sub-second precision is dropped
    deliberately: it is not meaningful here and varies between runs."""
    if isinstance(ts, (int, float)):
        dt = _dt.datetime.fromtimestamp(ts, tz=_dt.timezone.utc)
    else:
        dt = ts.astimezone(_dt.timezone.utc) if ts.tzinfo else ts.replace(
            tzinfo=_dt.timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def normalise_url(url: str) -> str:
    """Lowercase scheme and host, drop fragment and tracking params, keep path
    case (paths are case-sensitive on most servers, hosts are not)."""
    if not url:
        return ""
    p = urllib.parse.urlsplit(url.strip())
    if not p.scheme or not p.netloc:
        return nfc(url.strip())

    scheme = p.scheme.lower()
    host = p.hostname.lower() if p.hostname else ""
    if p.port and not ((scheme == "http" and p.port == 80) or
                       (scheme == "https" and p.port == 443)):
        host = f"{host}:{p.port}"
    if p.username:
        host = f"{p.username}@{host}" if not p.password else f"{p.username}:***@{host}"

    kept = [
        (k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True)
        if not (k.lower() in TRACKING_EXACT
                or any(k.lower().startswith(x) for x in TRACKING_PREFIXES))
    ]
    query = urllib.parse.urlencode(kept, doseq=True)
    return nfc(urllib.parse.urlunsplit((scheme, host, p.path, query, "")))


def domain_of(url: str) -> str:
    host = urllib.parse.urlsplit(normalise_url(url)).hostname or ""
    return host[4:] if host.startswith("www.") else host


def to_basis_points(x: float) -> int:
    """0.7413 -> 7413.

    Float repr differs across platforms and BLAS builds. Storing a float would
    make the hash unreproducible on someone else's machine, which would defeat
    the entire point of publishing it.
    """
    return int(round(float(x) * 10_000))


def _clean(value, *, path: str = "$"):
    """Recursively normalise, and reject anything that cannot be canonicalised."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise CanonicalizationError(
            f"float at {path}: {value!r}. Floats are not permitted in the "
            "canonical object -- convert to integer basis points first."
        )
    if isinstance(value, int) or value is None:
        return value
    if isinstance(value, str):
        return nfc(value)
    if isinstance(value, (list, tuple)):
        return [_clean(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k in sorted(value):
            if not isinstance(k, str):
                raise CanonicalizationError(f"non-string key at {path}: {k!r}")
            out[nfc(k)] = _clean(value[k], path=f"{path}.{k}")
        return out
    raise CanonicalizationError(f"unsupported type at {path}: {type(value).__name__}")


def canonicalize(obj: dict) -> str:
    """Return the exact string whose UTF-8 bytes get hashed."""
    import json

    return json.dumps(
        _clean(obj),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def canonical_bytes(obj: dict) -> bytes:
    return canonicalize(obj).encode("utf-8")
