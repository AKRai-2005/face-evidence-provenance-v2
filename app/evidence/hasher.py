"""Hashing helpers for evidence and media."""
from __future__ import annotations

import hashlib
import pathlib

from .canonicalizer import canonical_bytes


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def evidence_hash(evidence: dict) -> str:
    """SHA-256 over the canonical UTF-8 bytes of the evidence object.

    This is the value written to the blockchain. A judge must be able to
    reproduce it from canonical.json with nothing but the standard library:

        python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" canonical.json
    """
    return sha256_bytes(canonical_bytes(evidence))
