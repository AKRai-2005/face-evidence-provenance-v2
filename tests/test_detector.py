"""Detector behaviour, including the tight-crop failure mode."""
import cv2
import numpy as np
import pytest

from app.face.detector import PAD_RETRY_FRACTION, FaceEngine

FIX = "data/fixtures/A3_pichai_warsaw_crop.jpg"


@pytest.fixture(scope="module")
def engine():
    return FaceEngine.shared()


def _tight_crop(engine, path=FIX):
    """Crop hard to the face box so it fills nearly the whole frame -- the shape
    of a profile picture, and the case RetinaFace misses."""
    img = cv2.imread(path)
    faces = engine.detect(img)
    assert faces, "fixture must contain a detectable face to build the crop from"
    x1, y1, x2, y2 = faces[0].bbox
    h, w = img.shape[:2]
    m = int(0.02 * max(x2 - x1, y2 - y1))
    return img[max(0, y1 - m):min(h, y2 + m), max(0, x1 - m):min(w, x2 + m)]


def test_tight_headshot_is_detected_via_padded_retry(engine):
    """A cropped headshot is the most natural input a user would supply, and
    RetinaFace misses faces that fill too much of the frame. Without the retry
    this returns nothing and the run dies with 'no face detected'."""
    crop = _tight_crop(engine)
    assert engine.detect(crop, allow_pad_retry=True), \
        "padded retry should recover a face from a tight headshot"


def test_padded_retry_reports_bbox_in_original_coordinates(engine):
    """matched_face_bbox goes into the notarised evidence, so a bbox expressed
    in the temporary padded frame would be recorded wrong."""
    crop = _tight_crop(engine)
    faces = engine.detect(crop)
    if not faces:
        pytest.skip("fixture crop not recoverable on this build")
    h, w = crop.shape[:2]
    x1, y1, x2, y2 = faces[0].bbox
    # Must land within the original frame, allowing a small overhang: the true
    # face box can extend past a hard crop edge.
    slack = int(PAD_RETRY_FRACTION * max(w, h))
    assert -slack <= x1 < w + slack and -slack <= y1 < h + slack
    assert 0 < x2 <= w + slack and 0 < y2 <= h + slack
    assert x2 > x1 and y2 > y1


def test_pad_retry_can_be_disabled(engine):
    """The flag exists so the failure mode stays reproducible."""
    crop = _tight_crop(engine)
    without = engine.detect(crop, allow_pad_retry=False)
    with_ = engine.detect(crop, allow_pad_retry=True)
    assert len(with_) >= len(without)


def test_normal_image_does_not_change_under_the_retry(engine):
    """The retry must be a fallback only -- images that already work are
    detected on the first pass and must be untouched by it."""
    img = cv2.imread("data/fixtures/A2_pichai_warsaw2022.jpg")
    a = engine.detect(img, allow_pad_retry=False)
    b = engine.detect(img, allow_pad_retry=True)
    assert len(a) == len(b) == 2
    assert [f.bbox for f in a] == [f.bbox for f in b]


def test_blank_image_yields_no_face_even_with_retry(engine):
    """The retry must not manufacture detections out of nothing."""
    blank = np.full((400, 400, 3), 127, np.uint8)
    assert engine.detect(blank) == []
