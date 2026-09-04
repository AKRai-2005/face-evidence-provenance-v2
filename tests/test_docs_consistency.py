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
SAMPLE = ROOT / "sample_run"


def _read_json(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


# --- the sample run a judge is told to check --------------------------------
def test_shipped_canonical_json_hashes_to_the_bundles_claim():
    """The bundle's claimed digest must match the bytes actually shipped."""
    raw = (SAMPLE / "canonical.json").read_bytes()
    assert hashlib.sha256(raw).hexdigest() == _read_json(SAMPLE / "bundle.json")["evidence_sha256"]


def test_readme_documents_the_hash_the_shipped_file_actually_produces():
    """The README prints an expected digest next to a copy-pasteable command.
    If the sample run is ever refreshed without updating that line, a reviewer
    following it sees a mismatch -- which looks exactly like tampering."""
    actual = hashlib.sha256((SAMPLE / "canonical.json").read_bytes()).hexdigest()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert actual in readme, (
        f"README does not mention the digest the shipped canonical.json produces "
        f"({actual}). The walkthrough would fail for a reviewer."
    )


def test_shipped_canonical_json_has_no_line_endings_to_translate():
    raw = (SAMPLE / "canonical.json").read_bytes()
    assert b"\n" not in raw and b"\r" not in raw


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
