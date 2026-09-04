"""Claims made in the docs, enforced as tests.

Two of these guard mistakes that already happened once in this project:

  * the README walkthrough promised a hash that the shipped file did not
    produce, so a reviewer following the instructions would have seen a
    mismatch and concluded the evidence was fabricated;
  * calibration artifacts containing withdrawn figures sat in the repo with
    nothing marking them as withdrawn, while the docs said no such figure was
    quoted anywhere.

Prose cannot be relied on to stay true as files change. These can.
"""
import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Every committed run a reviewer is invited to check. Adding a third means
# adding it here; its walkthrough is then held to the same standard.
SAMPLE_RUNS = [ROOT / "sample_run", ROOT / "sample_run_2"]


def _read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


# --- the sample runs a judge is told to check -------------------------------
@pytest.mark.parametrize("sample", SAMPLE_RUNS, ids=lambda p: p.name)
def test_shipped_canonical_json_hashes_to_the_bundles_claim(sample):
    """The bundle's claimed digest must match the bytes actually shipped."""
    raw = (sample / "canonical.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _read_json(sample / "bundle.json")["evidence_sha256"]


@pytest.mark.parametrize("sample", SAMPLE_RUNS, ids=lambda p: p.name)
def test_walkthrough_documents_the_hash_the_shipped_file_produces(sample):
    """Each run's README prints an expected digest beside a copy-pasteable
    command. If a run is refreshed without updating that line, a reviewer
    following it sees a mismatch -- which looks exactly like tampering. That
    already happened once here."""
    actual = hashlib.sha256((sample / "canonical.json").read_bytes()).hexdigest()
    doc = (sample / "README.md").read_text(encoding="utf-8")
    assert actual in doc, (
        f"{sample.name}/README.md omits the digest its canonical.json produces "
        f"({actual}). The walkthrough would fail for a reviewer."
    )


def test_top_level_readme_documents_the_run_it_walks_through():
    actual = hashlib.sha256((ROOT / "sample_run" / "canonical.json").read_bytes()).hexdigest()
    assert actual in (ROOT / "README.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("sample", SAMPLE_RUNS, ids=lambda p: p.name)
def test_shipped_canonical_json_has_no_line_endings_to_translate(sample):
    raw = (sample / "canonical.json").read_bytes()
    assert b"\n" not in raw and b"\r" not in raw


@pytest.mark.parametrize("sample", SAMPLE_RUNS, ids=lambda p: p.name)
def test_each_run_notarised_a_distinct_digest(sample):
    """Two runs sharing an evidence hash would be one run committed twice, and
    the second would prove nothing the first did not."""
    mine = _read_json(sample / "bundle.json")["evidence_sha256"]
    for other in (s for s in SAMPLE_RUNS if s != sample):
        assert mine != _read_json(other / "bundle.json")["evidence_sha256"]


@pytest.mark.parametrize("sample", SAMPLE_RUNS, ids=lambda p: p.name)
def test_no_run_ships_the_live_api_key_or_a_private_key(sample):
    """The raw provider responses are committed verbatim. Verbatim is the point
    -- and also the risk."""
    for f in sample.rglob("*.json"):
        text = f.read_text(encoding="utf-8-sig", errors="replace")
        assert "api_key=" not in text, f"{f} carries an api_key parameter"
        assert "BEGIN PRIVATE KEY" not in text


# --- withdrawn measurements must say so, in the file itself -----------------
# Kept deliberately rather than deleted: the record of what was tried and
# discarded is part of the argument. But a reader who opens only the JSON must
# not mistake a discarded number for a result.
RETRACTED = ["demographics.json", "demographic_far.json"]
LIVE = ["results.json", "in_domain.json", "per_identity.json"]


@pytest.mark.parametrize("name", RETRACTED)
def test_withdrawn_artifacts_are_marked_withdrawn_in_band(name):
    d = _read_json(ROOT / "calibration" / name)
    assert d.get("RETRACTED") is True, f"{name} carries withdrawn figures but is not marked"
    assert d.get("DO_NOT_CITE"), f"{name} must say why it cannot be cited"
    assert d.get("retraction_reason"), f"{name} must record what defeated the measurement"


@pytest.mark.parametrize("name", LIVE)
def test_live_artifacts_are_not_marked_withdrawn(name):
    """If everything were marked, the marker would carry no information."""
    assert _read_json(ROOT / "calibration" / name).get("RETRACTED") is not True


def test_every_calibration_artifact_is_classified():
    """A new artifact must be sorted into one list or the other, so a withdrawn
    figure cannot enter the repo unmarked simply by being new."""
    on_disk = {p.name for p in (ROOT / "calibration").glob("*.json")}
    assert on_disk == set(RETRACTED) | set(LIVE), (
        f"unclassified calibration artifacts: {on_disk - set(RETRACTED) - set(LIVE)}"
    )


# --- the shooting script is read aloud on camera ----------------------------
# Errors here are spoken as fact and cannot be edited afterwards. Three were
# found by hand in one pass: a false-accept rate quoted as 7.8e-4 when the
# calibration says 6.8e-4, a frozen run described as matching forbes.com when
# it matched manofmany.com, and a tamper demo whose replacement string appeared
# in no committed bundle -- PowerShell's -replace fails silently, so that demo
# would have printed VERIFIED on camera in the middle of proving tampering.
SCRIPT = ROOT / "SHOOTING_SCRIPT.md"


def _script() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_tamper_demo_string_occurs_in_the_bundle_it_targets():
    """The demo copies sample_run_2 and replaces a substring. If that substring
    is absent the edit silently does nothing and the verifier reports VERIFIED,
    which is the opposite of what the scene claims to show."""
    s = _script()
    assert "Copy-Item -Recurse sample_run_2 tampered" in s, "tamper demo no longer targets sample_run_2"
    assert "'businessinsider','bus1nessinsider'" in s, "replacement pair changed"
    bundle = (ROOT / "sample_run_2" / "bundle.json").read_text(encoding="utf-8-sig")
    assert "businessinsider" in bundle, (
        "the tamper demo replaces 'businessinsider', which no longer appears in "
        "sample_run_2/bundle.json -- the demo would silently do nothing"
    )


def test_script_quotes_the_real_far():
    """calibration/results.json is the only source for this number."""
    far = _read_json(ROOT / "calibration" / "results.json")["far"]
    assert f"{far:.1e}".startswith("6.8"), "calibration changed; update the script's spoken FAR"
    assert "6.8 times ten-to-the-minus-four" in _script()
    assert "7.8 times ten-to-the-minus-four" not in _script()


@pytest.mark.parametrize("sample", SAMPLE_RUNS, ids=lambda p: p.name)
def test_script_names_the_domain_each_frozen_run_actually_matched(sample):
    domain = _read_json(sample / "evidence.json")["source_domain"]
    assert domain in _script(), (
        f"{sample.name} matched {domain}, which the shooting script never mentions"
    )


def test_script_quotes_the_second_runs_real_evidence_hash():
    h = _read_json(ROOT / "sample_run_2" / "bundle.json")["evidence_sha256"]
    assert h in _script(), "the tamper scene prints an expected 'claimed' hash; it is stale"


def test_script_contains_no_control_characters():
    """A backslash-escape slip once wrote a literal backspace into a path in
    this file, turning sample_run\bundle.json into sample_runundle.json."""
    s = _script()
    bad = [c for c in s if ord(c) < 32 and c not in "\n\r\t"]
    assert not bad, f"control characters in the script: {[hex(ord(c)) for c in bad[:5]]}"
