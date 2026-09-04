"""Face detection + embedding via InsightFace buffalo_l (RetinaFace + ArcFace).

Model note: buffalo_l's recognition net is w600k_r50 (ArcFace, ResNet-50, 512-d),
NOT r100. The build brief's schema string said r100; using it would have written
a false model claim into an immutable on-chain record. Verified by inspecting the
loaded ONNX graph at runtime -- see FaceEngine.model_id().
"""
from __future__ import annotations

import dataclasses
import io
import logging
import pathlib

import cv2
import imagehash
import numpy as np
from PIL import Image

MODEL_NAME = "buffalo_l"

# Retry padding when the first detection pass finds nothing. 0.30 sits well
# inside the range measured to work (10% upward) with headroom for tighter crops.
PAD_RETRY_FRACTION = 0.30


class NoFaceDetected(RuntimeError):
    pass


class FaceTooSmall(RuntimeError):
    pass


class FaceTooBlurry(RuntimeError):
    pass


@dataclasses.dataclass(frozen=True)
class DetectedFace:
    index: int
    bbox: tuple[int, int, int, int]     # x1, y1, x2, y2
    det_score: float
    embedding: np.ndarray               # 512-d, L2-normalised
    face_px: int                        # shorter side of the box
    sharpness: float                    # scale-normalised Laplacian variance
    face_phash: str                     # pHash of the normalised face region

    def as_record(self) -> dict:
        return {
            "index": self.index,
            "bbox": list(self.bbox),
            "det_score": round(float(self.det_score), 4),
            "face_px": self.face_px,
            "sharpness": round(float(self.sharpness), 1),
            "face_phash": self.face_phash,
        }


def face_region_phash(img_bgr: np.ndarray, bbox, *, size: int = 128, hash_size: int = 8):
    """pHash over the face box expanded 30% and resampled to a fixed square.

    Whole-image pHash is useless for our purpose: a crop randomises it (measured
    110-138 of 256 across every candidate alike in Phase 0). Normalising position
    and scale to the face makes the hash comparable across different crops of the
    same photograph, which is what lets us tell 'same photo republished' from
    'genuinely different photograph'.
    """
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox]
    pad = int(0.30 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return imagehash.phash(Image.fromarray(crop[:, :, ::-1]), hash_size=hash_size)


def face_sharpness(img_bgr: np.ndarray, bbox, *, size: int = 112) -> float:
    """Variance of the Laplacian over the face box, resampled to a fixed size.

    Resampling first is essential: raw Laplacian variance scales with resolution,
    so a big soft face could outscore a small sharp one. Measured in Phase 0 on
    real fixtures -- out-of-focus audience faces land at 5-6, every usable face
    at 995-4219. Detection score does NOT catch this: those blurred faces scored
    0.74-0.80 and passed a size+confidence gate while embedding to pure noise.
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    crop = img_bgr[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return 0.0
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def phash_distance(a: str, b: str) -> int:
    """Hamming distance between two hex pHash strings.

    int() is load-bearing: imagehash returns a numpy int64, which json.dumps
    refuses to serialise, and the evidence bundle is JSON.
    """
    return int(imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b))


class FaceEngine:
    """Lazily-initialised InsightFace wrapper. Loading costs ~2s, so share one."""

    _instance: "FaceEngine | None" = None

    def __init__(self, *, det_size: int = 640, logger: logging.Logger | None = None):
        from insightface.app import FaceAnalysis

        self._log = logger or logging.getLogger("task3")
        self._app = FaceAnalysis(name=MODEL_NAME, providers=["CPUExecutionProvider"])
        self._app.prepare(ctx_id=-1, det_size=(det_size, det_size))
        self._rec_file = self._resolve_recognition_model()

    @classmethod
    def shared(cls, **kw) -> "FaceEngine":
        if cls._instance is None:
            cls._instance = cls(**kw)
        return cls._instance

    def _resolve_recognition_model(self) -> str:
        """Read the actual recognition net name rather than trusting a constant."""
        try:
            m = self._app.models.get("recognition")
            return pathlib.Path(getattr(m, "model_file", "") or "").stem or "unknown"
        except Exception:
            return "unknown"

    def model_id(self) -> str:
        """Stable identifier written into the evidence bundle."""
        return f"insightface_{MODEL_NAME}_arcface_{self._rec_file}"

    def _detect_on(self, img_bgr: np.ndarray, dx: int = 0, dy: int = 0
                   ) -> list[DetectedFace]:
        """Detect on `img_bgr`, reporting bboxes shifted back by (dx, dy).

        The shift exists so a padded retry still reports coordinates in the
        ORIGINAL image's frame -- matched_face_bbox goes into the evidence
        object, and a bbox expressed in a temporary padded frame would be wrong.
        """
        out = []
        for i, f in enumerate(sorted(self._app.get(img_bgr), key=lambda x: -x.det_score)):
            x1, y1, x2, y2 = [int(v) for v in f.bbox]
            # pHash and sharpness are measured on the image actually detected on,
            # where the face pixels are identical; only the coordinates shift.
            ph = face_region_phash(img_bgr, (x1, y1, x2, y2))
            sharp = face_sharpness(img_bgr, (x1, y1, x2, y2))
            out.append(
                DetectedFace(
                    index=i,
                    bbox=(x1 - dx, y1 - dy, x2 - dx, y2 - dy),
                    det_score=float(f.det_score),
                    embedding=np.asarray(f.normed_embedding, dtype=np.float32),
                    face_px=min(x2 - x1, y2 - y1),
                    sharpness=sharp,
                    face_phash=str(ph) if ph is not None else "",
                )
            )
        return out

    def detect(self, img_bgr: np.ndarray, *, allow_pad_retry: bool = True
               ) -> list[DetectedFace]:
        """All faces, ordered by detection confidence (highest first).

        Falls back to a padded retry when nothing is found. RetinaFace misses
        faces that fill too much of the frame -- measured on a 1024x1024
        head-and-shoulders portrait, detection returns NOTHING at 0% padding and
        succeeds at det_score 0.82-0.89 from 10% padding upward. That failure
        mode matters more than it sounds: a cropped headshot or profile picture
        is the most natural thing a user would feed this tool, and without the
        retry it is rejected outright as "no face detected".

        BORDER_REPLICATE is used rather than a solid colour so the padding does
        not introduce a hard edge the detector could latch onto.
        """
        faces = self._detect_on(img_bgr)
        if faces or not allow_pad_retry:
            return faces

        h, w = img_bgr.shape[:2]
        px, py = int(w * PAD_RETRY_FRACTION), int(h * PAD_RETRY_FRACTION)
        if px < 1 or py < 1:
            return faces
        padded = cv2.copyMakeBorder(img_bgr, py, py, px, px, cv2.BORDER_REPLICATE)
        return self._detect_on(padded, dx=px, dy=py)

    def detect_primary(
        self, img_bgr: np.ndarray, *, min_face_px: int, min_det_score: float,
        min_sharpness: float = 100.0,
    ) -> tuple[DetectedFace, list[DetectedFace]]:
        """Pick the input image's subject face, enforcing the quality gate.

        Selection is by detection confidence, NOT by box area. Phase 0 showed
        largest-face selection picks the wrong person in group photos (it chose
        Morawiecki over Pichai and scored a same-person pair at 0.0123).
        """
        faces = self.detect(img_bgr)
        if not faces:
            raise NoFaceDetected(
                "No face detected in the input image.\n"
                "  The image must contain a clearly visible, front-facing face."
            )

        big = [f for f in faces if f.det_score >= min_det_score and f.face_px >= min_face_px]
        if not big:
            best = faces[0]
            raise FaceTooSmall(
                f"Detected {len(faces)} face(s), but none meet the size/confidence "
                f"gate (need >= {min_face_px}px and det_score >= {min_det_score:.2f}).\n"
                f"  Best was {best.face_px}px at det_score {best.det_score:.3f}.\n"
                "  Use a higher-resolution image with a larger, front-facing face."
            )

        usable = [f for f in big if f.sharpness >= min_sharpness]
        if not usable:
            best = max(big, key=lambda f: f.sharpness)
            raise FaceTooBlurry(
                f"Detected {len(faces)} face(s) of adequate size, but all are too "
                f"blurred to embed reliably (need sharpness >= {min_sharpness:.0f}).\n"
                f"  Sharpest was {best.sharpness:.1f} at {best.face_px}px.\n"
                "  Out-of-focus faces embed to noise and score ~0.0 against every "
                "identity, which would produce a meaningless similarity number."
            )

        if len(usable) > 1:
            self._log.warning(
                "%d usable faces found; selecting highest-confidence "
                "(idx %d, det %.3f). All faces are recorded.",
                len(usable), usable[0].index, usable[0].det_score,
                extra={"stage": "FACE"},
            )
        return usable[0], faces


# Test-time augmentation. The input is a DERIVED artifact -- a crop we chose --
# so the honest question is how much the score depends on that arbitrary choice.
# Measured on the demo input: the un-augmented score sat at the TOP of its own
# augmentation range (0.8503 max vs 0.8429 median), i.e. reporting the single
# point estimate was quietly optimistic. Cost is ~5 extra embeddings of one
# small image, about 1.5s.
TTA_TRANSFORMS = ("identity", "hflip", "crop4", "crop8", "scale085", "scale125")


def augmented_views(img_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Return (name, image) pairs for test-time augmentation of the input."""
    h, w = img_bgr.shape[:2]
    out = [("identity", img_bgr), ("hflip", cv2.flip(img_bgr, 1))]
    for frac, name in ((0.04, "crop4"), (0.08, "crop8")):
        out.append((name, img_bgr[int(h * frac):int(h * (1 - frac)),
                                  int(w * frac):int(w * (1 - frac))]))
    for scale, name in ((0.85, "scale085"), (1.25, "scale125")):
        out.append((name, cv2.resize(img_bgr, (max(32, int(w * scale)),
                                               max(32, int(h * scale))),
                                     interpolation=cv2.INTER_AREA)))
    return out
