"""Embedding-derived values that leave the process.

Nothing biometric is ever published. The only embedding-derived value that
reaches the blockchain is a salted commitment, which is one-way.
"""
from __future__ import annotations

import hashlib

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


def subject_commitment(emb: np.ndarray, salt_hex: str) -> str:
    """SHA-256(salt || quantized_embedding), hex.

    Why a commitment and not the embedding: a public ledger is immutable and
    world-readable. Publishing a face template there could never be undone. The
    commitment still lets us later PROVE a record concerns a given subject, by
    revealing the salt and re-deriving -- without broadcasting biometrics.

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
