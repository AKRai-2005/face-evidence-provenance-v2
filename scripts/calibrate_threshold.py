r"""Choose the decision threshold from measured data, not from a constant.

    .venv\Scripts\python.exe scripts/calibrate_threshold.py

Produces calibration/results.json, calibration/roc.png and the numbers quoted in
calibration/METHOD.md. app/face/matcher.py loads the threshold from that file and
refuses to run if it is absent.

TWO thresholds are calibrated, because the pipeline makes two decisions.

1. similarity threshold -- "is this the same person?"
   ROC over LFW pairs, threshold taken at a target FAR.

   Note on sample size: FAR = 1e-3 means one false accept in a thousand, so it
   cannot be OBSERVED with fewer than ~1000 negative pairs. LFW's standard test
   split has 500, which would make a 1e-3 figure an extrapolation rather than a
   measurement. Embeddings are computed once and pairing is free, so we build
   pairs combinatorially from identities instead and can afford tens of
   thousands of negatives.

2. face-pHash threshold -- "is this a different PHOTOGRAPH, or the same one
   republished?" Calibrated from two distributions:
     same photograph  : an image against realistic republication transforms of
                        itself (rescale, re-crop, re-compress) -- what outlets
                        actually do to a press photo
     different photo  : LFW positive pairs, which are by construction different
                        photographs of the same person
"""
from __future__ import annotations

import argparse
import io
import json
import pathlib
import random
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "calibration"
TARGET_FAR = 1e-3


def _republication_variants(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Transforms a news outlet plausibly applies when republishing a photo."""
    import cv2

    h, w = img_bgr.shape[:2]
    out = []

    for scale in (0.6, 1.4):
        nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
        out.append(cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA))

    # modest re-crop (outlets crop to their aspect ratio)
    out.append(img_bgr[int(h * 0.06):int(h * 0.94), int(w * 0.08):int(w * 0.92)])

    # aggressive JPEG re-compression
    ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 45])
    if ok:
        out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", type=int, default=280,
                    help="LFW identities to use (each with >=3 images)")
    ap.add_argument("--max-negatives", type=int, default=60000)
    ap.add_argument("--phash-subjects", type=int, default=60)
    ap.add_argument("--far", type=float, default=TARGET_FAR)
    ap.add_argument("--seed", type=int, default=20260901)
    args = ap.parse_args()

    import cv2
    from sklearn.datasets import fetch_lfw_people
    from sklearn.metrics import roc_curve

    from app.face.detector import FaceEngine, phash_distance
    from app.face.encoder import cosine

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("  loading LFW (funneled, colour, full size) ...")
    people = fetch_lfw_people(min_faces_per_person=3, color=True, resize=1.0,
                              slice_=None, funneled=True)
    images, labels, names = people.images, people.target, people.target_names
    print(f"  {images.shape[0]} images, {len(names)} identities")

    by_id: dict[int, list[int]] = {}
    for i, y in enumerate(labels):
        by_id.setdefault(int(y), []).append(i)
    chosen = sorted(by_id, key=lambda k: -len(by_id[k]))[: args.identities]
    idxs = [i for k in chosen for i in by_id[k]]
    print(f"  using {len(chosen)} identities, {len(idxs)} images")

    engine = FaceEngine.shared()

    # ---- embed each image exactly once -------------------------------------
    emb: dict[int, np.ndarray] = {}
    ph: dict[int, str] = {}
    t0, rejected = time.time(), 0
    for n, i in enumerate(idxs, 1):
        arr = images[i]
        bgr = cv2.cvtColor((arr * 255).astype(np.uint8) if arr.max() <= 1.0
                           else arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        faces = engine.detect(bgr)
        if not faces:
            rejected += 1
            continue
        f = max(faces, key=lambda x: x.det_score)
        emb[i] = f.embedding
        ph[i] = f.face_phash
        if n % 100 == 0:
            print(f"    embedded {n}/{len(idxs)}  ({time.time()-t0:.0f}s, "
                  f"{rejected} without a detectable face)")
    print(f"  embedded {len(emb)} images in {time.time()-t0:.0f}s "
          f"({rejected} rejected)")

    usable: dict[int, list[int]] = {}
    for k in chosen:
        got = [i for i in by_id[k] if i in emb]
        if len(got) >= 2:
            usable[k] = got

    # ---- positive pairs: every within-identity combination ------------------
    pos = [(a, b) for got in usable.values()
           for x, a in enumerate(got) for b in got[x + 1:]]
    # ---- negative pairs: sampled across identities --------------------------
    keys = list(usable)
    neg = set()
    target_neg = min(args.max_negatives, len(keys) * (len(keys) - 1) // 2 * 4)
    while len(neg) < target_neg:
        k1, k2 = rng.sample(keys, 2)
        neg.add((rng.choice(usable[k1]), rng.choice(usable[k2])))
    neg = list(neg)
    print(f"  pairs: {len(pos)} positive, {len(neg)} negative")

    if len(neg) < 1 / args.far:
        print(f"  WARNING: {len(neg)} negatives cannot resolve FAR={args.far:g}")

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.array([cosine(emb[a], emb[b]) for a, b in pos] +
                      [cosine(emb[a], emb[b]) for a, b in neg])

    fpr, tpr, thr = roc_curve(y, scores)
    k = int(np.argmin(np.abs(fpr - args.far)))
    threshold, far_at, tar_at = float(thr[k]), float(fpr[k]), float(tpr[k])
    auc = float(np.trapezoid(tpr, fpr))
    print(f"\n  threshold {threshold:.4f} at FAR {far_at:.2e} -> TAR {tar_at:.4f}  (AUC {auc:.5f})")

    # ---- face-pHash: same photograph vs different photograph ---------------
    print("\n  calibrating face-pHash separation ...")
    same_photo, diff_photo = [], []
    for k in list(usable)[: args.phash_subjects]:
        got = usable[k]
        arr = images[got[0]]
        bgr = cv2.cvtColor((arr * 255).astype(np.uint8) if arr.max() <= 1.0
                           else arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
        base = engine.detect(bgr)
        if not base:
            continue
        b_ph = max(base, key=lambda x: x.det_score).face_phash
        for var in _republication_variants(bgr):
            vf = engine.detect(var)
            if vf:
                d = phash_distance(b_ph, max(vf, key=lambda x: x.det_score).face_phash)
                same_photo.append(d)
        for x, a in enumerate(got):
            for b in got[x + 1:]:
                if a in ph and b in ph:
                    diff_photo.append(phash_distance(ph[a], ph[b]))

    sp, dp = np.array(same_photo), np.array(diff_photo)
    # Separate at the same-photo 95th percentile: republication variants must
    # almost always fall below it, and it stays well under the different-photo
    # median so genuine matches are not misfiled as republications.
    phash_max = int(np.percentile(sp, 95)) if len(sp) else 20
    print(f"  same photograph      n={len(sp):5d} median {np.median(sp) if len(sp) else -1:.0f} "
          f"p95 {np.percentile(sp,95) if len(sp) else -1:.0f}")
    print(f"  different photograph n={len(dp):5d} median {np.median(dp) if len(dp) else -1:.0f} "
          f"p05 {np.percentile(dp,5) if len(dp) else -1:.0f}")
    print(f"  -> same_photo_phash_max = {phash_max}")

    # ---- outputs ------------------------------------------------------------
    results = {
        "schema": "hhgoa2026.task3.calibration.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "LFW (funneled, colour, full-size), min_faces_per_person=3",
        "model": engine.model_id(),
        "metric": "cosine_l2normed",
        "n_identities": len(usable),
        "n_images_embedded": len(emb),
        "n_images_rejected": rejected,
        "n_pairs": len(pos) + len(neg),
        "n_positive_pairs": len(pos),
        "n_negative_pairs": len(neg),
        "threshold_cosine": round(threshold, 4),
        "far": far_at,
        "far_target": args.far,
        "tar": round(tar_at, 4),
        "auc": round(auc, 5),
        "same_photo_phash_max": phash_max,
        "phash_same_photo_median": float(np.median(sp)) if len(sp) else None,
        "phash_same_photo_p95": float(np.percentile(sp, 95)) if len(sp) else None,
        "phash_diff_photo_median": float(np.median(dp)) if len(dp) else None,
        "phash_diff_photo_p05": float(np.percentile(dp, 5)) if len(dp) else None,
        "seed": args.seed,
        "method": "calibration/METHOD.md",
    }
    (OUT_DIR / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(fpr, tpr, lw=2, label=f"ArcFace w600k_r50 (AUC {auc:.4f})")
    ax1.scatter([far_at], [tar_at], color="crimson", zorder=5,
                label=f"threshold {threshold:.4f}\nFAR {far_at:.1e}, TAR {tar_at:.3f}")
    ax1.set_xscale("log")
    ax1.set_xlim(max(1e-5, fpr[fpr > 0].min() if (fpr > 0).any() else 1e-5), 1)
    ax1.set_xlabel("False accept rate (log)")
    ax1.set_ylabel("True accept rate")
    ax1.set_title(f"ROC -- {len(pos)} positive / {len(neg)} negative pairs")
    ax1.grid(alpha=.3); ax1.legend(loc="lower right", fontsize=8)

    if len(sp) and len(dp):
        bins = np.arange(0, 65, 2)
        ax2.hist(sp, bins=bins, alpha=.65, label=f"same photograph (n={len(sp)})")
        ax2.hist(dp, bins=bins, alpha=.65, label=f"different photograph (n={len(dp)})")
        ax2.axvline(phash_max, color="crimson", ls="--",
                    label=f"same_photo_phash_max = {phash_max}")
        ax2.set_xlabel("face-region pHash Hamming distance (64-bit)")
        ax2.set_ylabel("pairs")
        ax2.set_title("Same photograph vs different photograph")
        ax2.legend(fontsize=8); ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc.png", dpi=140)

    print(f"\n  wrote {OUT_DIR/'results.json'}")
    print(f"  wrote {OUT_DIR/'roc.png'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
