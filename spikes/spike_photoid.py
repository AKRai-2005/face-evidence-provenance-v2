r"""Decisive Phase 0 experiment: can we tell "a DIFFERENT PHOTOGRAPH of the same
person" apart from "the SAME photograph republished at another size/crop"?

Whole-image pHash cannot: our query is a crop, and cropping randomises a global
hash (observed distances 110-138 out of 256 for every candidate alike).

Hypothesis: pHash computed over the ALIGNED FACE REGION is crop- and
scale-invariant, so it stays low for the same photograph and high for a
different one -- giving us a real three-way verdict.
"""
import hashlib, io, json, pathlib
import numpy as np, requests
from PIL import Image
import imagehash
from _spike_common import OUT, UA, check_interpreter

ROOT = pathlib.Path(__file__).resolve().parent.parent
CACHE = OUT / "candidates"; CACHE.mkdir(parents=True, exist_ok=True)
MAX_BYTES = 10 * 1024 * 1024
N = 14
OK_TYPES = ("image/jpeg", "image/png", "image/webp")


def face_phash(img_bgr, face, size=128):
    """pHash of the face box expanded 30% and resampled to a fixed square.
    Normalising position and scale is what makes this comparable across crops."""
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in face.bbox]
    pad = int(0.30 * max(x2 - x1, y2 - y1))
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    crop = img_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    import cv2
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return imagehash.phash(Image.fromarray(crop[:, :, ::-1]), hash_size=8)  # 64-bit


def main() -> None:
    check_interpreter()
    import cv2
    from insightface.app import FaceAnalysis

    qb = (OUT / "derived_query.jpg").read_bytes()
    vm = json.loads((OUT / "serpapi_lens_raw.json").read_text(encoding="utf-8"))["visual_matches"]

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    qimg = cv2.imdecode(np.frombuffer(qb, np.uint8), cv2.IMREAD_COLOR)
    qf = sorted(app.get(qimg), key=lambda f: -f.det_score)[0]
    qe, q_fph = qf.normed_embedding, face_phash(qimg, qf)
    q_wph = imagehash.phash(Image.open(io.BytesIO(qb)).convert("RGB"), hash_size=8)
    print(f"\n  query face-pHash: {q_fph}   whole-image pHash: {q_wph}\n")

    rows = []
    for m in vm[:N]:
        url, src = (m.get("image") or m.get("thumbnail")), str(m.get("source", ""))[:20]
        if not url:
            continue
        cf = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".bin")
        try:
            if cf.exists():
                b = cf.read_bytes()
            else:
                r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
                ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
                if r.status_code != 200 or ct not in OK_TYPES or len(r.content) > MAX_BYTES:
                    rows.append((src, None, None, None, f"HTTP {r.status_code} {ct or '?'}")); continue
                b = r.content; cf.write_bytes(b)
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            faces = app.get(img)
            if not faces:
                rows.append((src, None, None, None, "no face")); continue
            best = max(faces, key=lambda f: float(np.dot(qe, f.normed_embedding)))
            cos = float(np.dot(qe, best.normed_embedding))
            fph = face_phash(img, best)
            wph = imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"), hash_size=8)
            rows.append((src, cos, (q_fph - fph) if fph else None, q_wph - wph, ""))
        except Exception as e:
            rows.append((src, None, None, None, type(e).__name__))

    print(f"  {'source':22s} {'cosine':9s} {'faceHashD':10s} {'wholeHashD':11s} verdict")
    print("  " + "-" * 78)
    for src, cos, fd, wd, note in sorted(rows, key=lambda r: -(r[1] or -9)):
        if cos is None:
            print(f"  {src:22s} {'-':9s} {'-':10s} {'-':11s} {note}"); continue
        # face-region hash is the discriminator; cosine alone cannot separate
        # "same photo" from "same person" reliably.
        if fd is not None and fd <= 8:
            v = "SAME PHOTOGRAPH (republished)"
        elif cos >= 0.63:
            v = "DIFFERENT PHOTO, SAME SUBJECT  <<<"
        else:
            v = "below threshold"
        print(f"  {src:22s} {cos:+.4f}   {str(fd):10s} {str(wd):11s} {v}")

    print("\n  face-pHash distance is over a 64-bit hash (max 64).")


if __name__ == "__main__":
    main()
