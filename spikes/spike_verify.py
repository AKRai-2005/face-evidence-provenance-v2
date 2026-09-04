r"""Phase 0 follow-up: the experiment that decides how we frame the whole task.

Takes the DERIVED query artifact and the real visual_matches captured from
SerpApi, downloads the candidate images, and for each computes:
    sha256           -- is it a different file?
    pHash distance   -- is it a different PHOTOGRAPH, or just a re-crop?
    face cosine      -- is it the same person?

The interesting cell is: different sha256 AND high pHash distance AND high
cosine. That combination cannot be produced by near-duplicate/file matching,
and it is the only combination that proves face identification.
"""
import hashlib, io, json, pathlib, sys
import numpy as np, requests
from PIL import Image
import imagehash
from _spike_common import OUT, UA, check_interpreter

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAX_BYTES = 10 * 1024 * 1024
N = 14
OK_TYPES = ("image/jpeg", "image/png", "image/webp")


def phash(b: bytes):
    return imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"), hash_size=16)


def main() -> None:
    check_interpreter()
    import cv2
    from insightface.app import FaceAnalysis

    q = (OUT / "derived_query.jpg")
    if not q.exists():
        print("  Run spike_lens.py first (need derived_query.jpg)."); sys.exit(2)
    qb = q.read_bytes()
    q_sha, q_ph = hashlib.sha256(qb).hexdigest(), phash(qb)

    raw = OUT / "serpapi_lens_raw.json"
    vm = json.loads(raw.read_text(encoding="utf-8")).get("visual_matches", [])
    print(f"\n  query : {q.name} sha={q_sha[:12]}.. phash={q_ph}")
    print(f"  pool  : {len(vm)} visual matches; examining first {N}\n")

    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    qimg = cv2.imdecode(np.frombuffer(qb, np.uint8), cv2.IMREAD_COLOR)
    qf = sorted(app.get(qimg), key=lambda f: -f.det_score)
    if not qf:
        print("  No face in query image."); sys.exit(1)
    qe = qf[0].normed_embedding
    print(f"  query face det={qf[0].det_score:.3f}\n")

    rows = []
    for m in vm[:N]:
        url = m.get("image") or m.get("thumbnail")
        src = str(m.get("source", ""))[:20]
        if not url:
            continue
        try:
            r = requests.get(url, headers={"User-Agent": UA}, timeout=15, stream=True)
            ct = r.headers.get("content-type", "").split(";")[0].strip().lower()
            if r.status_code != 200 or ct not in OK_TYPES:
                rows.append((src, None, None, None, f"HTTP {r.status_code} {ct or '?'}")); continue
            b = r.raw.read(MAX_BYTES + 1, decode_content=True)
            if len(b) > MAX_BYTES:
                rows.append((src, None, None, None, "oversized")); continue
        except Exception as e:
            rows.append((src, None, None, None, type(e).__name__)); continue

        try:
            c_sha = hashlib.sha256(b).hexdigest()
            d_ph = q_ph - phash(b)
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                rows.append((src, None, None, None, "decode fail")); continue
            faces = app.get(img)
            if not faces:
                rows.append((src, c_sha, d_ph, None, f"no face ({img.shape[1]}x{img.shape[0]})")); continue
            cos = max(float(np.dot(qe, f.normed_embedding)) for f in faces)
            rows.append((src, c_sha, d_ph, cos, f"{len(faces)} face(s) {img.shape[1]}x{img.shape[0]}"))
        except Exception as e:
            rows.append((src, None, None, None, f"{type(e).__name__}"))

    print(f"  {'source':22s} {'sha256':10s} {'same?':6s} {'pHashD':7s} {'cosine':8s} note")
    print("  " + "-" * 84)
    for src, sha, d, cos, note in rows:
        s_sha = (sha[:10] if sha else "-")
        same = ("YES" if sha == q_sha else "no") if sha else "-"
        s_d = (f"{d}" if d is not None else "-")
        s_c = (f"{cos:+.4f}" if cos is not None else "-")
        print(f"  {src:22s} {s_sha:10s} {same:6s} {s_d:7s} {s_c:8s} {note}")

    scored = [r for r in rows if r[3] is not None]
    if scored:
        best = max(scored, key=lambda r: r[3])
        print(f"\n  BEST: {best[0]}  cosine={best[3]:+.4f}  pHashDistance={best[2]}  sha differs={best[1]!=q_sha}")
        strong = [r for r in scored if r[3] >= 0.63 and r[2] is not None and r[2] >= 25]
        print("  Candidates that are a DIFFERENT PHOTOGRAPH of the same person")
        print(f"  (cosine >= 0.63 AND pHash distance >= 25): {len(strong)}")
        for r in sorted(strong, key=lambda r: -r[3])[:5]:
            print(f"     {r[0]:22s} cos={r[3]:+.4f} pHashD={r[2]}")


if __name__ == "__main__":
    main()
