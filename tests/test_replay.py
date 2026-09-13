"""--replay must never modify the run it replays.

It used to. A replay wrote its bundle, evidence, canonical bytes and log back
into the captured run's own directory, replacing a notarised bundle with a
freshly timestamped one that had never been recorded. The only field that
changed was retrieved_at, which was enough: the original stopped verifying
and reported NOT REGISTERED. That happened to two real runs while their
replay fallback was being checked before a live demo -- replay being the
thing reached for precisely when a demo has already gone wrong.

Both runs were recovered exactly (the original timestamp was found by
reproducing the on-chain hash), but the property belongs in a test, not in
anyone's memory.

This builds its own capture from committed fixtures, so it runs on a clean
clone: runs/ is gitignored and cannot be relied on.
"""
import hashlib
import json
import pathlib
import shutil

import pytest

import app.main as pipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
SOURCE_ID = "20260101T000000Z_capt01"

# Stands in for a bundle that is already on chain. If a replay rewrites it,
# the real equivalent is a notarised run that no longer verifies.
NOTARISED = {"schema": "sentinel", "evidence_sha256": "0" * 64,
             "chain": {"tx_hash": "0x" + "ab" * 32}}


def _capture(runs: pathlib.Path) -> pathlib.Path:
    d = runs / SOURCE_ID
    for sub in ("input", "raw", "candidates"):
        (d / sub).mkdir(parents=True)

    probe = ROOT / "data" / "input.jpg"                      # derived crop of Nadella
    raw = probe.read_bytes()
    shutil.copyfile(probe, d / "input" / "input.jpg")
    # a different photograph of the same person, as a candidate would be
    shutil.copyfile(ROOT / "data" / "fixtures" / "B1_nadella_2017.jpg",
                    d / "candidates" / "cand01_fixture.jpg")

    (d / "run.json").write_text(json.dumps({
        "run_id": SOURCE_ID, "started_at": "2026-01-01T00:00:00Z",
        "input_name": "input.jpg",
        "input_sha256": hashlib.sha256(raw).hexdigest(),
        "input_width": 460, "input_height": 306}), encoding="utf-8")
    (d / "search.json").write_text(json.dumps({
        "provider": "serpapi_google_lens",
        "query_image_sha256": hashlib.sha256(raw).hexdigest(),
        "exact_match_count": 0,
        "candidates": [{
            "position": 1, "title": "fixture", "page_url": "https://example.org/a",
            "image_url": "https://example.org/a.jpg", "source": "Example",
            "provider": "serpapi_google_lens", "thumbnail_url": ""}]}),
        encoding="utf-8")
    (d / "raw" / "serpapi_lens.json").write_text("{}", encoding="utf-8")
    for name in ("bundle.json", "evidence.json"):
        (d / name).write_text(json.dumps(NOTARISED), encoding="utf-8")
    (d / "canonical.json").write_bytes(b'{"sentinel":true}')
    (d / "run.log").write_text("[RUN] original log line\n", encoding="utf-8")
    return d


def _snapshot(d: pathlib.Path) -> dict:
    return {str(p.relative_to(d)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(d.rglob("*")) if p.is_file()}


@pytest.fixture()
def runs(tmp_path, monkeypatch):
    r = tmp_path / "runs"
    r.mkdir()
    monkeypatch.setattr(pipeline, "RUNS", r)
    monkeypatch.setattr(pipeline, "CONSENT", tmp_path / ".consent")
    return r


def test_replay_leaves_the_captured_run_byte_identical(runs):
    src = _capture(runs)
    before = _snapshot(src)

    for _ in range(2):                          # a demo gets replayed more than once
        code = pipeline.main(["--replay", SOURCE_ID, "--yes", "--no-chain"])
        assert code == 0, f"replay did not complete (exit {code})"

    assert _snapshot(src) == before, (
        "--replay modified the run it replayed. On a real notarised run this "
        "replaces the recorded bundle and the original stops verifying.")


def test_each_replay_writes_a_run_directory_of_its_own(runs):
    _capture(runs)
    for _ in range(2):
        assert pipeline.main(["--replay", SOURCE_ID, "--yes", "--no-chain"]) == 0

    outputs = sorted(p for p in runs.iterdir() if p.name != SOURCE_ID)
    assert len(outputs) == 2, f"expected two replay outputs, found {[p.name for p in outputs]}"
    for out in outputs:
        bundle = json.loads((out / "bundle.json").read_text(encoding="utf-8"))
        assert bundle["run_id"] == out.name != SOURCE_ID
        assert f"replay of {SOURCE_ID}" in bundle["notes"]
        assert "not notarised" in bundle["notes"]
        assert (out / "canonical.json").exists()
        assert hashlib.sha256((out / "canonical.json").read_bytes()).hexdigest() \
            == bundle["evidence_sha256"]


def test_a_replay_output_is_not_itself_replayable(runs):
    """It holds no capture -- no run.json, no candidates -- so replaying it must
    be refused rather than half-work from an empty directory."""
    _capture(runs)
    assert pipeline.main(["--replay", SOURCE_ID, "--yes", "--no-chain"]) == 0
    out = next(p for p in runs.iterdir() if p.name != SOURCE_ID)
    assert not (out / "run.json").exists()
    assert pipeline.main(["--replay", out.name, "--yes", "--no-chain"]) \
        == pipeline.EXIT_BAD_INPUT
