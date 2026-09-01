"""Similarity scoring and the three-way match verdict.

The central claim of this project is that we performed FACE identification, not
file lookup. That claim needs three outcomes, not two -- see Verdict.
"""
from __future__ import annotations

import dataclasses
import enum
import json
import pathlib

from .detector import phash_distance

CALIBRATION = pathlib.Path(__file__).resolve().parent.parent.parent / "calibration" / "results.json"


class CalibrationMissing(RuntimeError):
    pass


class Verdict(str, enum.Enum):
    """Ordered weakest -> strongest."""

    NO_MATCH = "NO MATCH ABOVE THRESHOLD"
    EXACT_DUPLICATE = "EXACT-DUPLICATE MATCH (weak evidence)"
    SAME_PHOTO = "SAME-PHOTOGRAPH REPUBLICATION (moderate evidence)"
    DISTINCT_PHOTO = "DISTINCT-PHOTOGRAPH SAME-SUBJECT CANDIDATE (strong evidence)"


@dataclasses.dataclass(frozen=True)
class Thresholds:
    similarity: float          # cosine, from ROC at a stated FAR
    far: float
    tar: float
    pairs: int
    same_photo_phash: int      # face-region pHash distance below which it is the same photo
    source: str

    @classmethod
    def load(cls, path: pathlib.Path = CALIBRATION) -> "Thresholds":
        if not path.exists():
            raise CalibrationMissing(
                f"No calibration file at {path}.\n"
                "  The decision threshold must come from measured data, not a\n"
                "  hardcoded constant. Generate it:\n"
                "      python scripts/calibrate_threshold.py"
            )
        d = json.loads(path.read_text(encoding="utf-8"))
        try:
            return cls(
                similarity=float(d["threshold_cosine"]),
                far=float(d["far"]),
                tar=float(d["tar"]),
                pairs=int(d["n_pairs"]),
                same_photo_phash=int(d["same_photo_phash_max"]),
                source=str(d.get("method", "calibration/METHOD.md")),
            )
        except KeyError as e:
            raise CalibrationMissing(f"{path} is missing required key {e}") from None

    def describe(self) -> str:
        return (
            f"cosine {self.similarity:.4f} selected at FAR {self.far:g} on "
            f"{self.pairs} pairs (TAR {self.tar:.3f})"
        )


def classify(
    *,
    similarity: float,
    input_sha256: str,
    candidate_sha256: str,
    input_face_phash: str,
    candidate_face_phash: str,
    thresholds: Thresholds,
) -> tuple[Verdict, int | None]:
    """Return (verdict, face_phash_distance).

    Order matters. Byte equality is checked first because it is decisive and
    cheap; a same-SHA match is weak evidence no matter how high the cosine is.
    """
    if input_sha256 == candidate_sha256:
        return Verdict.EXACT_DUPLICATE, 0

    dist = None
    if input_face_phash and candidate_face_phash:
        dist = phash_distance(input_face_phash, candidate_face_phash)

    if similarity < thresholds.similarity:
        return Verdict.NO_MATCH, dist

    # Different file, above threshold. Is it a different PHOTOGRAPH, or the same
    # photograph republished at another size/crop? Whole-image pHash cannot tell
    # (cropping randomises it); the face-region hash can.
    if dist is not None and dist <= thresholds.same_photo_phash:
        return Verdict.SAME_PHOTO, dist
    return Verdict.DISTINCT_PHOTO, dist
