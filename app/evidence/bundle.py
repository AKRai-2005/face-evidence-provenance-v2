"""Assemble the canonical evidence object and the verifiable bundle.

Two distinct artefacts, and the distinction matters:

  evidence  -- the canonical object. Exactly these fields, canonicalised, are
               what SHA-256 covers and what is notarised on chain. Nothing may
               be added to it later without changing the hash.

  bundle    -- evidence + context (chain receipt, run metadata, runner-up
               scores, provider notes). The bundle is what a verifier reads.
               It carries the evidence verbatim so the verifier can recompute
               the hash independently, plus the transaction to check it against.
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

from .canonicalizer import (SCHEMA, canonicalize, domain_of, normalise_url,
                            rfc3339, to_basis_points)
from .hasher import evidence_hash

BUNDLE_SCHEMA = "hhgoa2026.task3.bundle.v1"


@dataclasses.dataclass
class EvidenceInputs:
    input_image_sha256: str
    input_face_phash: str
    matched_image_sha256: str
    matched_image_phash: str
    phash_hamming_distance: int | None
    face_similarity: float
    face_similarity_lo: float
    face_similarity_hi: float
    tta_views: int
    evidence_strength: str
    threshold: float
    faces_in_candidate: int
    matched_face_index: int
    matched_face_bbox: tuple[int, int, int, int]
    source_url: str
    page_title: str
    search_provider: str
    search_query_image_sha256: str
    candidates_examined: int
    retrieved_at: object          # datetime or epoch seconds
    model: str
    verdict: str
    exact_match_count: int | None = None
    metric: str = "cosine_l2normed"


def build_evidence(i: EvidenceInputs) -> dict:
    """The canonical evidence object. No floats -- see canonicalizer."""
    url = normalise_url(i.source_url)
    ev = {
        "schema": SCHEMA,
        "input_image_sha256": i.input_image_sha256,
        "input_face_phash": i.input_face_phash,
        "matched_image_sha256": i.matched_image_sha256,
        "matched_image_phash": i.matched_image_phash,
        "phash_hamming_distance": (
            int(i.phash_hamming_distance) if i.phash_hamming_distance is not None else None
        ),
        "face_similarity_bp": to_basis_points(i.face_similarity),
        "face_similarity_lo_bp": to_basis_points(i.face_similarity_lo),
        "face_similarity_hi_bp": to_basis_points(i.face_similarity_hi),
        "tta_views": int(i.tta_views),
        "evidence_strength": i.evidence_strength,
        "threshold_bp": to_basis_points(i.threshold),
        "faces_in_candidate": int(i.faces_in_candidate),
        "matched_face_index": int(i.matched_face_index),
        "matched_face_bbox": [int(v) for v in i.matched_face_bbox],
        "source_url": url,
        "source_domain": domain_of(url),
        "page_title": i.page_title or "",
        "search_provider": i.search_provider,
        "search_query_image_sha256": i.search_query_image_sha256,
        "candidates_examined": int(i.candidates_examined),
        "retrieved_at": rfc3339(i.retrieved_at),
        "model": i.model,
        "metric": i.metric,
        "verdict": i.verdict,
    }
    if i.exact_match_count is not None:
        ev["provider_exact_match_count"] = int(i.exact_match_count)
    return ev


def build_bundle(
    *,
    evidence: dict,
    run_id: str,
    chain: dict | None,
    runner_up: list[dict] | None = None,
    notes: str = "",
) -> dict:
    """Evidence plus everything a verifier needs, without altering the evidence."""
    return {
        "schema": BUNDLE_SCHEMA,
        "run_id": run_id,
        "evidence": evidence,
        "evidence_sha256": evidence_hash(evidence),
        "chain": chain or {},
        "runner_up_scores": runner_up or [],
        "notes": notes,
    }


def write_artifacts(run_dir: pathlib.Path, evidence: dict, bundle: dict) -> dict:
    """Write canonical.json, evidence.json and bundle.json.

    canonical.json holds the EXACT bytes that were hashed, so the documented
    one-liner reproduces the digest with no re-serialisation step in between:

        python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" canonical.json
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    canonical = canonicalize(evidence).encode("utf-8")

    paths = {
        "canonical": run_dir / "canonical.json",
        "evidence": run_dir / "evidence.json",
        "bundle": run_dir / "bundle.json",
    }
    paths["canonical"].write_bytes(canonical)
    paths["evidence"].write_text(json.dumps(evidence, indent=2, ensure_ascii=False),
                                 encoding="utf-8")
    paths["bundle"].write_text(json.dumps(bundle, indent=2, ensure_ascii=False),
                               encoding="utf-8")
    return paths
