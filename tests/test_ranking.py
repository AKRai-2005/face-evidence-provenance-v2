"""The three-way verdict and evidence-strength ordering.

These encode the project's central claim, so they are tested against the actual
numbers measured in Phase 0 rather than invented ones.
"""
import pytest

from app.face.matcher import Thresholds, Verdict, classify
from app.face.ranking import ScoredCandidate, rank, score_spread, select_best

TH = Thresholds(similarity=0.30, far=1e-3, tar=0.99, pairs=1000,
                same_photo_phash=20, source="test")

INPUT_SHA = "aa" * 32
INPUT_PH = "a5a152873b5c2e67"


def _classify(sim, cand_sha=None, cand_ph="ffffffffffffffff"):
    return classify(similarity=sim, input_sha256=INPUT_SHA,
                    candidate_sha256=cand_sha or "bb" * 32,
                    input_face_phash=INPUT_PH, candidate_face_phash=cand_ph,
                    thresholds=TH)


def _cand(pos, sim, verdict, dist):
    return ScoredCandidate(
        position=pos, page_url=f"https://e{pos}.example/p", image_url="",
        source=f"src{pos}", sha256=f"{pos:02x}" * 32,
        face_phash="0" * 16, similarity=sim,
        verdict=verdict, phash_distance=dist, faces_in_candidate=1,
        matched_face_index=0, matched_face_bbox=(0, 0, 10, 10),
        image_size=(100, 100))


# --- verdict ------------------------------------------------------------
def test_identical_bytes_is_exact_duplicate_however_high_the_score():
    """A same-SHA match is weak evidence even at cosine 1.0 -- it proves file
    identity, not face identification."""
    v, d = _classify(0.99, cand_sha=INPUT_SHA)
    assert v is Verdict.EXACT_DUPLICATE
    assert d == 0


def test_below_threshold_is_no_match():
    assert _classify(0.29)[0] is Verdict.NO_MATCH


def test_at_threshold_is_a_match():
    assert _classify(0.30)[0] is not Verdict.NO_MATCH


def test_low_phash_distance_means_same_photograph_republished():
    """Phase 0: republications measured 4-14 on the face-region hash."""
    for d in (4, 6, 8, 14):
        ph = _shift(INPUT_PH, d)
        v, dist = _classify(0.98, cand_ph=ph)
        assert dist == d
        assert v is Verdict.SAME_PHOTO


def test_high_phash_distance_means_distinct_photograph():
    """Phase 0: genuinely different photographs measured 26-40."""
    for d in (26, 28, 34, 40):
        v, dist = _classify(0.70, cand_ph=_shift(INPUT_PH, d))
        assert dist == d
        assert v is Verdict.DISTINCT_PHOTO


def test_missing_phash_falls_back_to_similarity_only():
    v, d = _classify(0.70, cand_ph="")
    assert d is None
    assert v is Verdict.DISTINCT_PHOTO


def _shift(hex_hash: str, bits: int) -> str:
    """Return a hash differing from hex_hash in exactly `bits` positions."""
    n = int(hex_hash, 16)
    for i in range(bits):
        n ^= (1 << i)
    return f"{n:016x}"


# --- ordering -----------------------------------------------------------
def test_distinct_photo_outranks_a_higher_scoring_republication():
    """The core inversion. Measured in Phase 0: republications scored
    0.968-0.985 and distinct photographs 0.637-0.841, so ranking by cosine
    returns the WEAKEST evidence with the highest number."""
    cands = [
        _cand(1, 0.9850, Verdict.SAME_PHOTO, 8),
        _cand(2, 0.9787, Verdict.SAME_PHOTO, 6),
        _cand(3, 0.7328, Verdict.DISTINCT_PHOTO, 34),
        _cand(4, 0.6368, Verdict.DISTINCT_PHOTO, 40),
    ]
    ranked = rank(cands)
    assert ranked[0].position == 3
    assert ranked[0].verdict is Verdict.DISTINCT_PHOTO
    assert select_best(cands).position == 3


def test_within_tier_higher_cosine_wins():
    cands = [_cand(1, 0.70, Verdict.DISTINCT_PHOTO, 30),
             _cand(2, 0.84, Verdict.DISTINCT_PHOTO, 28)]
    assert rank(cands)[0].position == 2


def test_exact_duplicate_ranks_below_republication():
    cands = [_cand(1, 1.0, Verdict.EXACT_DUPLICATE, 0),
             _cand(2, 0.97, Verdict.SAME_PHOTO, 6)]
    assert rank(cands)[0].position == 2


def test_no_match_ranks_last_and_is_never_selected():
    cands = [_cand(1, 0.20, Verdict.NO_MATCH, 50)]
    assert rank(cands)[0].verdict is Verdict.NO_MATCH
    assert select_best(cands) is None


def test_select_best_returns_none_when_nothing_clears_threshold():
    """A null result is a valid outcome, reported honestly rather than fixed by
    lowering the threshold."""
    assert select_best([_cand(i, 0.1, Verdict.NO_MATCH, 60) for i in range(5)]) is None


# --- spread -------------------------------------------------------------
def test_score_spread_reports_margin():
    s = score_spread([_cand(1, 0.90, Verdict.DISTINCT_PHOTO, 30),
                      _cand(2, 0.40, Verdict.NO_MATCH, 40)])
    assert s["best"] == 0.9
    assert s["runner_up"] == 0.4
    assert s["margin"] == pytest.approx(0.5)


def test_score_spread_handles_single_and_empty():
    assert score_spread([])["n"] == 0
    s = score_spread([_cand(1, 0.9, Verdict.DISTINCT_PHOTO, 30)])
    assert s["runner_up"] is None and s["margin"] is None


# --- threshold loading --------------------------------------------------
def test_threshold_is_loaded_from_calibration_not_hardcoded():
    """SS13: the threshold must come from calibration/results.json."""
    t = Thresholds.load()
    assert 0.0 < t.similarity < 1.0
    assert t.pairs > 1000
    assert t.same_photo_phash > 0
