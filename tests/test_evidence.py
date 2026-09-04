"""Canonicalization is the contract a third party re-runs. Test it hard."""
import datetime as dt
import hashlib

import pytest

from app.evidence.canonicalizer import (CanonicalizationError, canonical_bytes,
                                        canonicalize, domain_of, nfc,
                                        normalise_url, rfc3339, to_basis_points)
from app.evidence.hasher import evidence_hash


# --- determinism ---------------------------------------------------------
def test_key_order_does_not_change_output():
    a = {"b": 1, "a": 2, "c": {"z": 1, "y": 2}}
    b = {"c": {"y": 2, "z": 1}, "a": 2, "b": 1}
    assert canonicalize(a) == canonicalize(b)


def test_keys_are_sorted_recursively():
    assert canonicalize({"b": {"d": 1, "c": 2}, "a": 3}) == '{"a":3,"b":{"c":2,"d":1}}'


def test_no_whitespace_in_output():
    out = canonicalize({"a": 1, "b": [1, 2], "c": {"d": "e"}})
    assert " " not in out and "\n" not in out


def test_hash_is_stable_across_key_order():
    a = {"schema": "x", "face_similarity_bp": 7413, "source_domain": "example.com"}
    b = {"source_domain": "example.com", "schema": "x", "face_similarity_bp": 7413}
    assert evidence_hash(a) == evidence_hash(b)


def test_hash_matches_the_documented_one_liner():
    """The README tells a judge to sha256 canonical.json directly. That must
    equal what we computed, or the whole verification story collapses."""
    ev = {"schema": "hhgoa2026.task3.evidence.v1", "face_similarity_bp": 7413}
    blob = canonical_bytes(ev)
    assert evidence_hash(ev) == hashlib.sha256(blob).hexdigest()


# --- floats are rejected -------------------------------------------------
def test_float_is_rejected_with_a_useful_message():
    with pytest.raises(CanonicalizationError) as ei:
        canonicalize({"similarity": 0.7413})
    assert "float" in str(ei.value) and "basis points" in str(ei.value)


def test_nested_float_is_rejected_and_names_its_path():
    with pytest.raises(CanonicalizationError) as ei:
        canonicalize({"a": {"b": [1, 2, 3.5]}})
    assert "$.a.b[2]" in str(ei.value)


def test_bool_survives_and_is_not_treated_as_int_or_float():
    assert canonicalize({"ok": True}) == '{"ok":true}'


def test_basis_points_conversion():
    assert to_basis_points(0.7413) == 7413
    assert to_basis_points(0.63) == 6300
    assert to_basis_points(1.0) == 10000
    assert to_basis_points(-0.0283) == -283


# --- unicode -------------------------------------------------------------
def test_nfc_normalisation_makes_equivalent_strings_identical():
    composed, decomposed = "Mateusz Morawiecki spotkał się", "Mateusz Morawiecki spotkał się"
    e_composed = "café"          # e-acute as one codepoint
    e_decomposed = "café"       # e + combining acute
    assert e_composed != e_decomposed
    assert nfc(e_composed) == nfc(e_decomposed)
    assert canonicalize({"t": e_composed}) == canonicalize({"t": e_decomposed})
    assert composed == decomposed


def test_non_ascii_is_not_escaped():
    assert "ł" in canonicalize({"t": "spotkał"})


def test_unicode_keys_are_normalised_too():
    assert canonicalize({"café": 1}) == canonicalize({"café": 1})


# --- URLs ----------------------------------------------------------------
def test_scheme_and_host_lowercased_path_preserved():
    assert normalise_url("HTTPS://Example.COM/Path/To/Page") == \
        "https://example.com/Path/To/Page"


def test_fragment_stripped():
    assert normalise_url("https://a.example/p#section") == "https://a.example/p"


def test_tracking_params_stripped():
    """Observed live: Wikimedia's API returns URLs carrying utm_* params."""
    u = ("https://upload.wikimedia.org/x.jpg?utm_source=commons.wikimedia.org"
         "&utm_campaign=imageinfo&width=800&fbclid=abc&igshid=z")
    assert normalise_url(u) == "https://upload.wikimedia.org/x.jpg?width=800"


def test_meaningful_query_params_survive():
    assert normalise_url("https://a.example/s?q=cat&page=2") == \
        "https://a.example/s?q=cat&page=2"


def test_default_ports_removed_nondefault_kept():
    assert normalise_url("https://a.example:443/p") == "https://a.example/p"
    assert normalise_url("https://a.example:8443/p") == "https://a.example:8443/p"


def test_domain_strips_www():
    assert domain_of("https://www.Forbes.com/sites/x") == "forbes.com"
    assert domain_of("https://en.wikipedia.org/wiki/X") == "en.wikipedia.org"


def test_url_change_changes_the_hash():
    """The tamper demo depends on this: edit one character, hash must move."""
    a = {"source_url": normalise_url("https://example.com/a")}
    b = {"source_url": normalise_url("https://example.com/b")}
    assert evidence_hash(a) != evidence_hash(b)


# --- timestamps ----------------------------------------------------------
def test_rfc3339_format_and_utc():
    d = dt.datetime(2026, 9, 5, 14, 22, 7, 123456, tzinfo=dt.timezone.utc)
    assert rfc3339(d) == "2026-09-05T14:22:07Z"


def test_rfc3339_converts_from_other_timezone():
    tz = dt.timezone(dt.timedelta(hours=5, minutes=30))
    d = dt.datetime(2026, 9, 5, 19, 52, 7, tzinfo=tz)
    assert rfc3339(d) == "2026-09-05T14:22:07Z"


def test_rfc3339_accepts_epoch_seconds():
    assert rfc3339(1788273407).endswith("Z")


def test_subsecond_precision_is_dropped():
    a = dt.datetime(2026, 9, 5, 14, 22, 7, 1, tzinfo=dt.timezone.utc)
    b = dt.datetime(2026, 9, 5, 14, 22, 7, 999999, tzinfo=dt.timezone.utc)
    assert rfc3339(a) == rfc3339(b)


# --- rejection of unsupported types --------------------------------------
def test_unsupported_type_rejected():
    with pytest.raises(CanonicalizationError):
        canonicalize({"when": dt.datetime.now()})


def test_non_string_key_rejected():
    with pytest.raises(CanonicalizationError):
        canonicalize({1: "a"})


# --- the verifier must agree with the pipeline ---------------------------
def test_standalone_verifier_matches_pipeline_canonicalization():
    """verify.py re-implements the SS6 rules independently, so that a bug in the
    pipeline's canonicalizer cannot hide inside its own verifier. They must
    nonetheless agree byte-for-byte, or a valid bundle would fail to verify."""
    import verify as v

    cases = [
        {"schema": "x", "a": 1, "b": "café", "c": [1, 2, 3]},
        {"z": {"y": {"x": "spotkał się"}}, "a": None, "ok": True},
        {"source_url": "https://example.com/A/b?q=1", "n": -283},
        {"nested": [{"b": 2, "a": 1}, {"d": [], "c": {}}]},
    ]
    for ev in cases:
        assert v.canonicalize(ev) == canonical_bytes(ev)
        assert v.compute_hash(ev) == evidence_hash(ev)


def test_verifier_also_rejects_floats():
    import verify as v
    with pytest.raises(ValueError):
        v.canonicalize({"similarity": 0.74})


def test_canonical_output_contains_no_newline():
    """Guards the on-disk bytes against end-of-line translation.

    canonical.json is hashed verbatim and the digest goes on chain. If the
    canonical form ever gained a newline, a clone on a platform with
    core.autocrlf=true would rewrite it to CRLF and the published hash would
    stop reproducing. .gitattributes marks the file -text as well; this test
    keeps the property true at the source.
    """
    ev = {"a": "x", "nested": {"b": [1, 2]}, "t": "café", "u": "spotkał"}
    blob = canonical_bytes(ev)
    assert b"\n" not in blob
    assert b"\r" not in blob


def test_canonical_bytes_survive_a_crlf_roundtrip():
    """A CRLF round-trip must not change the hash, because there is nothing to
    translate. If this ever fails, the evidence hash is platform-dependent."""
    ev = {"schema": "hhgoa2026.task3.evidence.v1", "face_similarity_bp": 8503}
    blob = canonical_bytes(ev)
    assert blob.replace(b"\n", b"\r\n") == blob
    assert evidence_hash(ev) == hashlib.sha256(blob).hexdigest()


def test_verifier_tolerates_a_utf8_bom(tmp_path):
    """Notepad and PowerShell 5.1's Set-Content -Encoding utf8 both write a BOM.

    A reviewer hand-editing bundle.json to try the tamper demo would otherwise
    be told 'not valid JSON' and reasonably conclude the tool was broken,
    instead of seeing TAMPER DETECTED.
    """
    import json

    import verify as v

    bundle = {"run_id": "t", "evidence": {"a": 1}, "evidence_sha256": "0" * 64}
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle), encoding="utf-8-sig")   # with BOM
    assert p.read_bytes().startswith(b"\xef\xbb\xbf")

    loaded = v.load_bundle(p)
    assert loaded["evidence"] == {"a": 1}


def test_verifier_still_reads_a_bomless_bundle(tmp_path):
    import json

    import verify as v

    bundle = {"run_id": "t", "evidence": {"a": 1}, "evidence_sha256": "0" * 64}
    p = tmp_path / "bundle.json"
    p.write_text(json.dumps(bundle), encoding="utf-8")
    assert not p.read_bytes().startswith(b"\xef\xbb\xbf")
    assert v.load_bundle(p)["evidence"] == {"a": 1}
