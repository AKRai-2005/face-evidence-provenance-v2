"""Evidence and bundle construction.

bundle.py builds the object whose hash is written to an immutable public ledger.
A defect here is not a bug that can be patched later -- it is a permanently
recorded false claim. It had no direct tests until now.
"""
import datetime as dt

import pytest

from app.evidence.bundle import (BUNDLE_SCHEMA, EvidenceInputs, build_bundle,
                                 build_evidence, write_artifacts)
from app.evidence.canonicalizer import SCHEMA, CanonicalizationError, canonicalize
from app.evidence.hasher import evidence_hash

# The 19 fields the build brief specifies for the canonical evidence object.
REQUIRED = [
    "schema", "input_image_sha256", "matched_image_sha256", "matched_image_phash",
    "phash_hamming_distance", "face_similarity_bp", "threshold_bp",
    "faces_in_candidate", "matched_face_index", "matched_face_bbox", "source_url",
    "source_domain", "page_title", "search_provider", "search_query_image_sha256",
    "candidates_examined", "retrieved_at", "model", "metric",
]


def _inputs(**over):
    base = dict(
        input_image_sha256="aa" * 32,
        input_face_phash="a5a152873b5c2e67",
        matched_image_sha256="bb" * 32,
        matched_image_phash="d0c11a8f352c67ed",
        phash_hamming_distance=28,
        face_similarity=0.9623,
        face_similarity_lo=0.9445,
        face_similarity_hi=0.9679,
        tta_views=6,
        evidence_strength="strong",
        threshold=0.2149,
        faces_in_candidate=1,
        matched_face_index=0,
        matched_face_bbox=(388, 201, 690, 556),
        source_url="https://WWW.Example.com/Path?utm_source=x&q=1#frag",
        page_title="A title",
        search_provider="serpapi_google_lens",
        search_query_image_sha256="aa" * 32,
        candidates_examined=12,
        retrieved_at=dt.datetime(2026, 9, 4, 11, 40, 4, tzinfo=dt.timezone.utc),
        model="insightface_buffalo_l_arcface_w600k_r50",
        verdict="DISTINCT_PHOTO",
    )
    base.update(over)
    return EvidenceInputs(**base)


# --- schema completeness -------------------------------------------------
def test_every_field_the_brief_requires_is_present():
    ev = build_evidence(_inputs())
    assert [k for k in REQUIRED if k not in ev] == []


def test_schema_string_is_the_current_version():
    assert build_evidence(_inputs())["schema"] == SCHEMA


# --- the no-floats rule --------------------------------------------------
def test_evidence_contains_no_floats_anywhere():
    """Floats do not hash reproducibly across platforms, so a single one would
    silently make the published hash unverifiable on someone else's machine."""
    ev = build_evidence(_inputs())

    def walk(v, path="$"):
        assert not isinstance(v, float), f"float at {path}: {v!r}"
        if isinstance(v, dict):
            for k, vv in v.items():
                walk(vv, f"{path}.{k}")
        elif isinstance(v, list):
            for i, vv in enumerate(v):
                walk(vv, f"{path}[{i}]")

    walk(ev)
    canonicalize(ev)          # would raise if any float survived


def test_similarity_is_stored_as_basis_points():
    ev = build_evidence(_inputs(face_similarity=0.7413, threshold=0.63))
    assert ev["face_similarity_bp"] == 7413
    assert ev["threshold_bp"] == 6300


def test_tta_interval_is_carried_as_integers():
    ev = build_evidence(_inputs())
    assert ev["face_similarity_lo_bp"] == 9445
    assert ev["face_similarity_hi_bp"] == 9679
    assert ev["face_similarity_lo_bp"] <= ev["face_similarity_bp"] <= ev["face_similarity_hi_bp"]


# --- URL and timestamp normalisation -------------------------------------
def test_source_url_is_normalised_and_domain_derived():
    """The two fields deliberately differ. source_url is the real URL, so it
    keeps 'www.'; source_domain is what goes on chain, so it is the canonical
    domain. Both drop the fragment and utm_* tracking params."""
    ev = build_evidence(_inputs())
    assert ev["source_url"] == "https://www.example.com/Path?q=1"
    assert "utm_source" not in ev["source_url"]
    assert "#frag" not in ev["source_url"]
    assert ev["source_domain"] == "example.com"


def test_retrieved_at_is_rfc3339_utc_seconds():
    assert build_evidence(_inputs())["retrieved_at"] == "2026-09-04T11:40:04Z"


def test_null_phash_distance_survives_as_null_not_zero():
    """None means 'not measurable'. Coercing it to 0 would assert the strongest
    possible same-photo claim on missing data."""
    ev = build_evidence(_inputs(phash_hamming_distance=None))
    assert ev["phash_hamming_distance"] is None


def test_optional_exact_match_count_is_omitted_when_unknown():
    assert "provider_exact_match_count" not in build_evidence(_inputs())
    ev = build_evidence(_inputs(exact_match_count=0))
    assert ev["provider_exact_match_count"] == 0


# --- bundle --------------------------------------------------------------
def test_bundle_carries_evidence_verbatim_and_its_hash():
    ev = build_evidence(_inputs())
    b = build_bundle(evidence=ev, run_id="r1", chain={"tx_hash": "0xabc"})
    assert b["schema"] == BUNDLE_SCHEMA
    assert b["evidence"] == ev, "verifier must see the evidence unmodified"
    assert b["evidence_sha256"] == evidence_hash(ev)


def test_bundle_context_does_not_change_the_evidence_hash():
    """Runner-up scores and notes are context. If they entered the hash, adding
    a note after the fact would look like tampering."""
    ev = build_evidence(_inputs())
    a = build_bundle(evidence=ev, run_id="r1", chain=None)
    b = build_bundle(evidence=ev, run_id="r1", chain={"tx_hash": "0x1"},
                     runner_up=[{"source": "x"}], notes="anything")
    assert a["evidence_sha256"] == b["evidence_sha256"]


# --- artifacts on disk ---------------------------------------------------
def test_canonical_json_bytes_hash_to_the_published_value(tmp_path):
    """The README tells a reviewer to sha256 canonical.json directly. If the
    file were re-serialised rather than written verbatim, that would fail."""
    import hashlib

    ev = build_evidence(_inputs())
    b = build_bundle(evidence=ev, run_id="r1", chain=None)
    paths = write_artifacts(tmp_path, ev, b)
    raw = paths["canonical"].read_bytes()
    assert hashlib.sha256(raw).hexdigest() == b["evidence_sha256"]


def test_canonical_json_has_no_newline_for_eol_safety(tmp_path):
    ev = build_evidence(_inputs())
    paths = write_artifacts(tmp_path, ev, build_bundle(evidence=ev, run_id="r", chain=None))
    raw = paths["canonical"].read_bytes()
    assert b"\n" not in raw and b"\r" not in raw


def test_a_float_smuggled_into_evidence_is_rejected_not_rounded():
    ev = build_evidence(_inputs())
    ev["sneaky"] = 0.5
    with pytest.raises(CanonicalizationError):
        canonicalize(ev)
