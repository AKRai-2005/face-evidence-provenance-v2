"""Score candidates against the input face, and rank by EVIDENCE STRENGTH.

The ordering here is the least obvious and most important decision in the
project, so it is spelled out.

A naive pipeline ranks by cosine and takes the top hit. Measured in Phase 0 on
real Lens results, that returns the wrong answer: the highest scorers were
0.968-0.985, and every one of them was the SAME PRESS PHOTOGRAPH republished by
a different outlet at a different size. The genuinely different photographs of
the same person scored 0.637-0.841 -- lower, and far stronger evidence.

So we rank by verdict tier first and cosine only within a tier. The headline
result is the best DISTINCT-PHOTOGRAPH candidate, because that is the only
outcome that cannot be produced by file- or near-duplicate matching.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from .detector import DetectedFace, FaceEngine, phash_distance
from .encoder import cosine
from .matcher import Thresholds, Verdict, classify

# Strongest first. Used to order candidates across tiers.
TIER_RANK = {
    Verdict.DISTINCT_PHOTO: 0,
    Verdict.SAME_PHOTO: 1,
    Verdict.EXACT_DUPLICATE: 2,
    # UNCERTAIN ranks below every settled verdict but above NO_MATCH: it is a
    # candidate we decline to call either way, not one we rejected.
    Verdict.UNCERTAIN: 3,
    Verdict.NO_MATCH: 4,
}


@dataclasses.dataclass
class ScoredCandidate:
    position: int
    page_url: str
    image_url: str
    source: str
    sha256: str
    face_phash: str
    similarity: float          # median over test-time augmentations
    similarity_lo: float
    similarity_hi: float
    strength: str                  # "weak" when below the genuine p25 floor
    verdict: Verdict
    phash_distance: int | None
    faces_in_candidate: int
    matched_face_index: int
    matched_face_bbox: tuple[int, int, int, int]
    image_size: tuple[int, int]
    note: str = ""

    @property
    def tier(self) -> int:
        return TIER_RANK[self.verdict]

    def as_record(self) -> dict:
        return {
            "position": self.position,
            "source": self.source,
            "page_url": self.page_url,
            "image_url": self.image_url,
            "sha256": self.sha256,
            "face_phash": self.face_phash,
            "similarity": round(self.similarity, 4),
            "strength": self.strength,
            "similarity_lo": round(self.similarity_lo, 4),
            "similarity_hi": round(self.similarity_hi, 4),
            "verdict": self.verdict.name,
            "phash_distance": self.phash_distance,
            "faces_in_candidate": self.faces_in_candidate,
            "matched_face_index": self.matched_face_index,
            "matched_face_bbox": list(self.matched_face_bbox),
            "image_width": self.image_size[0],
            "image_height": self.image_size[1],
        }


def score_candidate(
    *,
    engine: FaceEngine,
    input_face: DetectedFace,
    input_sha256: str,
    fetched,
    thresholds: Thresholds,
    input_embeddings: list[np.ndarray] | None = None,
) -> ScoredCandidate | None:
    """Score one downloaded candidate, or None if it has no usable face.

    `input_embeddings` are the test-time-augmented embeddings of the input face.
    Each produces its own score against this candidate; we report the MEDIAN and
    the range. Using the single un-augmented embedding is quietly optimistic --
    measured on the demo input, it landed at the top of its own range.
    """
    import cv2

    img = cv2.imdecode(np.frombuffer(fetched.content, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return None

    faces = engine.detect(img)
    if not faces:
        return None

    probes = list(input_embeddings) if input_embeddings else [input_face.embedding]

    # Max over ALL faces, never the largest. Phase 0: in a two-person photo the
    # largest face was the wrong person, scoring a true match at 0.0123.
    per_view = [max(cosine(p, f.embedding) for f in faces) for p in probes]
    med = float(np.median(per_view))
    lo, hi = float(min(per_view)), float(max(per_view))

    # The matched face is the one the primary (un-augmented) embedding picks,
    # so the recorded bbox corresponds to a reproducible view of the input.
    best_i, best_cos = 0, -1.0
    for f in faces:
        c = cosine(input_face.embedding, f.embedding)
        if c > best_cos:
            best_i, best_cos = f.index, c
    best = faces[best_i]

    verdict, dist = classify(
        similarity=med,
        similarity_lo=lo,
        similarity_hi=hi,
        input_sha256=input_sha256,
        candidate_sha256=fetched.sha256,
        input_face_phash=input_face.face_phash,
        candidate_face_phash=best.face_phash,
        thresholds=thresholds,
    )

    h, w = img.shape[:2]
    return ScoredCandidate(
        position=fetched.position,
        page_url=fetched.page_url,
        image_url=fetched.image_url,
        source=fetched.source,
        sha256=fetched.sha256,
        face_phash=best.face_phash,
        similarity=med,
        similarity_lo=lo,
        similarity_hi=hi,
        strength=thresholds.strength_of(med),
        verdict=verdict,
        phash_distance=dist,
        faces_in_candidate=len(faces),
        matched_face_index=best.index,
        matched_face_bbox=best.bbox,
        image_size=(w, h),
    )


def rank(scored: list[ScoredCandidate]) -> list[ScoredCandidate]:
    """Strongest evidence first: verdict tier, then cosine within the tier."""
    return sorted(scored,
                  key=lambda s: (s.tier, 0 if s.strength == "strong" else 1,
                                 -s.similarity))


def select_best(scored: list[ScoredCandidate]) -> ScoredCandidate | None:
    """The headline result, or None when nothing clears the threshold.

    'No candidate above threshold' is a valid, reportable outcome. It is never
    resolved by lowering the threshold.
    """
    ranked = rank(scored)
    for s in ranked:
        if s.verdict is not Verdict.NO_MATCH:
            return s
    return None


def score_spread(scored: list[ScoredCandidate]) -> dict:
    """Summary stats. The gap between the best match and the runners-up is
    itself evidence: a lone high score against a low field is far more
    convincing than a cluster of similar scores."""
    sims = sorted((s.similarity for s in scored), reverse=True)
    if not sims:
        return {"n": 0}
    return {
        "n": len(sims),
        "best": round(sims[0], 4),
        "runner_up": round(sims[1], 4) if len(sims) > 1 else None,
        "margin": round(sims[0] - sims[1], 4) if len(sims) > 1 else None,
        "median": round(float(np.median(sims)), 4),
        "min": round(sims[-1], 4),
    }
