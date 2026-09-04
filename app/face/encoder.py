"""Embedding-derived values that leave the process.

Nothing biometric is ever published. The only embedding-derived value that
reaches the blockchain is a salted commitment, which is one-way.
"""
from __future__ import annotations

import hashlib
import hmac

import numpy as np

QUANT_SCALE = 127  # int8 range; see quantize_embedding()


def quantize_embedding(emb: np.ndarray) -> bytes:
    """Deterministic int8 quantisation of an L2-normalised embedding.

    Floats cannot be hashed reproducibly -- IEEE-754 repr and accumulation order
    differ across platforms and BLAS builds, so the same face could yield two
    different commitments. Quantising to int8 makes the commitment stable while
    staying far too coarse to reconstruct a usable face template.
    """
    if emb.ndim != 1:
        raise ValueError(f"expected a 1-D embedding, got shape {emb.shape}")
    q = np.clip(np.rint(np.asarray(emb, dtype=np.float64) * QUANT_SCALE), -128, 127)
    return q.astype(np.int8).tobytes()


def derive_record_salt(master_salt_hex: str, run_id: str) -> str:
    """Per-record salt = HMAC-SHA256(master_salt, run_id), hex.

    The point is DAMAGE CONTAINMENT when a commitment is opened.

    With a single deployment-wide salt, proving that one record concerns a given
    subject means revealing that salt -- which immediately makes EVERY other
    record we have ever written brute-forceable against a face gallery. The act
    of substantiating one claim would compromise all the others, which makes the
    capability effectively unusable.

    Deriving each record's salt from the master via HMAC means a record can be
    opened by revealing only ITS salt. The master stays secret, and no other
    record is weakened. HMAC (not plain concatenation) because it is the
    construction designed for keyed derivation and is not vulnerable to
    length-extension.

    Compromise of the master is still total -- that is inherent to any scheme
    where we can re-derive -- but it is now one failure instead of a routine
    consequence of normal use.
    """
    if len(master_salt_hex) < 32:
        raise ValueError("master commitment salt too short (need >= 32 hex chars)")
    return hmac.new(bytes.fromhex(master_salt_hex),
                    run_id.encode("utf-8"), hashlib.sha256).hexdigest()


def subject_commitment(emb: np.ndarray, salt_hex: str) -> str:
    """SHA-256(salt || quantized_embedding), hex.

    Why a commitment and not the embedding: a public ledger is immutable and
    world-readable. Publishing a face template there could never be undone. The
    commitment still lets us later PROVE a record concerns a given subject, by
    revealing the salt and re-deriving -- without broadcasting biometrics.

    `salt_hex` should be a PER-RECORD salt from derive_record_salt(), not the
    master. Passing the master still works and is what older runs did, but it
    couples every record's exposure together -- see derive_record_salt().

    The salt is what makes this safe. Without it, an unsalted hash of a
    quantised embedding is brute-forceable against a face gallery.
    """
    if len(salt_hex) < 32:
        raise ValueError("commitment salt too short (need >= 32 hex chars)")
    return hashlib.sha256(bytes.fromhex(salt_hex) + quantize_embedding(emb)).hexdigest()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity. Inputs are already L2-normalised by ArcFace, so this
    is a dot product -- but we renormalise defensively rather than assume."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a / na, b / nb))


def to_basis_points(x: float) -> int:
    """0.7413 -> 7413. The canonical evidence object contains no floats."""
    return int(round(x * 10_000))
