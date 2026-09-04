r"""Choose the decision thresholds from measured data, not from constants.

    .venv\Scripts\python.exe scripts/calibrate_threshold.py

Writes calibration/results.json and calibration/roc.png. app/face/matcher.py
loads the threshold from results.json and REFUSES TO RUN if it is absent --
there is no hardcoded fallback anywhere in the pipeline.

Two thresholds, because the pipeline makes two decisions. See METHOD.md.
"""
from __future__ import annotations

import argparse
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

# The twelve real Google Lens candidates measured in Phase 0, labelled by eye.
# Synthetic transforms only approximate republication; these are ground truth.
REAL_REPUBLICATION = [4, 6, 8, 8, 8, 14]
REAL_DISTINCT = [26, 26, 28, 28, 34, 40]


def _lfw_root() -> pathlib.Path | None:
    """Locate the funneled LFW images sklearn already downloaded."""
    for base in (pathlib.Path.home() / "scikit_learn_data",
                 pathlib.Path.home() / "scikit-learn-data"):
        p = base / "lfw_home" / "lfw_funneled"
        if p.is_dir():
            return p
    return None


def _republication_variants(img_bgr: np.ndarray) -> list[np.ndarray]:
    """Transforms a news outlet plausibly applies when republishing a photo.

    An earlier, gentler set produced a same-photo p95 of 6, while REAL
    republications measured 4-14 against live Lens results in Phase 0.
    Calibrating on it gave same_photo_phash_max=6, which then mislabelled a
    genuine republication (distance 12) as a DISTINCT PHOTOGRAPH in a live run
    -- inflating the project's strongest claim, in the one direction we must
    never err. This set models real republication far more closely.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    out = []

    for scale in (0.35, 0.6, 1.4):
        out.append(cv2.resize(img_bgr, (max(32, int(w * scale)), max(32, int(h * scale))),
                              interpolation=cv2.INTER_AREA))

    out.append(img_bgr[int(h * .06):int(h * .94), int(w * .08):int(w * .92)])
    out.append(img_bgr[int(h * .14):int(h * .90), int(w * .18):int(w * .86)])

    for q in (45, 25):
        ok, enc = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            out.append(cv2.imdecode(enc, cv2.IMREAD_COLOR))

    blur = cv2.GaussianBlur(img_bgr, (0, 0), 2.0)
    out.append(cv2.addWeighted(img_bgr, 1.6, blur, -0.6, 0))          # unsharp
    out.append(cv2.convertScaleAbs(img_bgr, alpha=1.18, beta=12))     # regrade up
    out.append(cv2.convertScaleAbs(img_bgr, alpha=0.85, beta=-10))    # regrade down
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 2.0, 1.0)
    out.append(cv2.warpAffine(img_bgr, M, (w, h), borderMode=cv2.BORDER_REPLICATE))
    out.append(cv2.cvtColor(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY),
                            cv2.COLOR_GRAY2BGR))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identities", type=int, default=450)
    ap.add_argument("--max-images-per-identity", type=int, default=6,
                    help="cap per identity; LFW is severely imbalanced")
    ap.add_argument("--max-pairs-per-identity", type=int, default=15)
    ap.add_argument("--max-negatives", type=int, default=60000)
    ap.add_argument("--phash-subjects", type=int, default=90)
    ap.add_argument("--far", type=float, default=TARGET_FAR)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    import cv2
    from sklearn.metrics import roc_curve

    from app.face.detector import FaceEngine, phash_distance
    from app.face.encoder import cosine

    rng = random.Random(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Read the LFW JPEGs straight off disk rather than via
    # sklearn.datasets.fetch_lfw_people. That helper materialises EVERY image
    # into a single float32 array -- ~3GB at full-size colour -- which pushed
    # this machine into swap and slowed embedding from 0.35s to ~2.4s/image.
    # Walking the directory keeps memory flat. sklearn is still used for the ROC.
    lfw = _lfw_root()
    if lfw is None:
        print("\n  LFW images not found on disk. Fetch once with:\n"
              '    python -c "from sklearn.datasets import fetch_lfw_people;'
              ' fetch_lfw_people(min_faces_per_person=3)"\n')
        return 2
    print(f"  LFW: {lfw}")

    people = []
    for d in sorted(lfw.iterdir()):
        if d.is_dir():
            imgs = sorted(d.glob("*.jpg"))
            if len(imgs) >= 3:
                people.append((d.name, imgs))
    print(f"  {len(people)} identities with >=3 images")

    # LFW is severely imbalanced: George W. Bush alone has 530 images. Taking
    # every within-identity combination let him contribute C(530,2) ~ 140k of
    # 236k positive pairs -- 59% of the positive set -- so the reported TAR was
    # effectively "TAR on one man", and AUC came out at 0.983 where ArcFace on
    # LFW should reach ~0.999. Cap images per identity so no one dominates.
    people.sort(key=lambda kv: -len(kv[1]))
    capped = {name: imgs[: args.max_images_per_identity]
              for name, imgs in people[: args.identities]}
    paths = [p for imgs in capped.values() for p in imgs]
    print(f"  using {len(capped)} identities, {len(paths)} images "
          f"(capped at {args.max_images_per_identity}/identity)")

    engine = FaceEngine.shared()

    # ---- embed each image exactly once, with an on-disk cache ---------------
    cache = OUT_DIR / "_embeddings_cache.npz"
    emb: dict[str, np.ndarray] = {}
    ph: dict[str, str] = {}
    if cache.exists() and not args.no_cache:
        z = np.load(cache, allow_pickle=True)
        emb = {str(k): z["emb"][i] for i, k in enumerate(z["key"])}
        ph = {str(k): str(z["ph"][i]) for i, k in enumerate(z["key"])}
        print(f"  loaded {len(emb)} cached embeddings (--no-cache to redo)")

    todo = [p for p in paths if str(p) not in emb]
    t0, rejected = time.time(), 0
    for n, fp in enumerate(todo, 1):
        img = cv2.imread(str(fp))
        if img is None:
            rejected += 1
            continue
        faces = engine.detect(img)
        if not faces:
            rejected += 1
            continue
        f = max(faces, key=lambda x: x.det_score)
        emb[str(fp)] = f.embedding
        ph[str(fp)] = f.face_phash
        if n % 200 == 0:
            rate = n / max(1e-9, time.time() - t0)
            print(f"    {n}/{len(todo)}  ({time.time()-t0:.0f}s, {rate:.1f} img/s, "
                  f"{rejected} rejected)")
    if todo:
        print(f"  embedded {len(todo)-rejected} images in {time.time()-t0:.0f}s "
              f"({rejected} rejected)")
        keys = sorted(emb)
        np.savez_compressed(cache, key=np.array(keys),
                            emb=np.stack([emb[k] for k in keys]),
                            ph=np.array([ph[k] for k in keys]))
        print(f"  cached {len(keys)} embeddings -> {cache.name}")

    usable = {name: [str(p) for p in imgs if str(p) in emb]
              for name, imgs in capped.items()}
    usable = {k: v for k, v in usable.items() if len(v) >= 2}

    # ---- pairs, capped per identity so no subject dominates -----------------
    pos = []
    for got in usable.values():
        allp = [(a, b) for i, a in enumerate(got) for b in got[i + 1:]]
        rng.shuffle(allp)
        pos.extend(allp[: args.max_pairs_per_identity])

    keys = list(usable)
    neg = set()
    while len(neg) < args.max_negatives:
        k1, k2 = rng.sample(keys, 2)
        neg.add((rng.choice(usable[k1]), rng.choice(usable[k2])))
    neg = list(neg)
    print(f"  pairs: {len(pos)} positive, {len(neg)} negative")
    if len(neg) < 1 / args.far:
        print(f"  WARNING: {len(neg)} negatives cannot resolve FAR={args.far:g}")

    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    scores = np.array([cosine(emb[a], emb[b]) for a, b in pos]
                      + [cosine(emb[a], emb[b]) for a, b in neg])

    # Percentiles of the GENUINE distribution. The threshold answers "is this
    # the same person at all"; these answer "is this a TYPICAL same-person
    # score". Both are needed: a synthetic face once scored 0.4866 against a
    # different synthetic face -- comfortably over the threshold, yet in the
    # bottom 5% of genuine scores -- and was reported as strong evidence. The
    # p25 floor is what stops a barely-plausible score being sold as a
    # confident one.
    gen = np.array([cosine(emb[a], emb[b]) for a, b in pos])
    strong_floor = float(np.percentile(gen, 25))

    fpr, tpr, thr = roc_curve(y, scores)
    k = int(np.argmin(np.abs(fpr - args.far)))
    threshold, far_at, tar_at = float(thr[k]), float(fpr[k]), float(tpr[k])
    auc = float(np.trapezoid(tpr, fpr))
    print(f"\n  threshold {threshold:.4f} at FAR {far_at:.2e} -> TAR {tar_at:.4f} "
          f"(AUC {auc:.5f})")

    # ---- face-pHash: same photograph vs different photograph ---------------
    print("\n  calibrating face-pHash separation ...")
    same_photo, diff_photo = [], []
    for name in list(usable)[: args.phash_subjects]:
        got = usable[name]
        img = cv2.imread(got[0])
        if img is None:
            continue
        base = engine.detect(img)
        if not base:
            continue
        b_ph = max(base, key=lambda x: x.det_score).face_phash
        for var in _republication_variants(img):
            vf = engine.detect(var)
            if vf:
                same_photo.append(
                    phash_distance(b_ph, max(vf, key=lambda x: x.det_score).face_phash))
        for i, a in enumerate(got):
            for b in got[i + 1:]:
                diff_photo.append(phash_distance(ph[a], ph[b]))

    sp, dp = np.array(same_photo), np.array(diff_photo)

    # Sweep for the cut that best separates BOTH distributions, rather than
    # taking a percentile of one. A percentile ignores where the other sits: the
    # same-photo p95 gave 6, below the 4-14 of real republications. Ties break
    # HIGH, because calling a republication a distinct photograph overstates the
    # evidence, while the reverse merely understates it.
    phash_max, best = 20, -1.0
    if len(sp) and len(dp):
        for t in range(65):
            s = (sp <= t).mean() + (dp > t).mean()
            if s >= best:
                best, phash_max = s, t
        print(f"  same photograph      n={len(sp):6d} median {np.median(sp):.0f} "
              f"p95 {np.percentile(sp,95):.0f} p99 {np.percentile(sp,99):.0f} max {sp.max()}")
        print(f"  different photograph n={len(dp):6d} median {np.median(dp):.0f} "
              f"p05 {np.percentile(dp,5):.0f} p01 {np.percentile(dp,1):.0f} min {dp.min()}")
        print(f"  -> same_photo_phash_max = {phash_max} "
              f"(captures {100*(sp<=phash_max).mean():.1f}% of republications, "
              f"{100*(dp>phash_max).mean():.1f}% of distinct photos)")

    ok_rep = sum(d <= phash_max for d in REAL_REPUBLICATION)
    ok_dis = sum(d > phash_max for d in REAL_DISTINCT)
    print(f"  real-world check (Phase 0 Lens candidates): "
          f"republications {ok_rep}/{len(REAL_REPUBLICATION)}, "
          f"distinct {ok_dis}/{len(REAL_DISTINCT)} correct")
    if ok_rep < len(REAL_REPUBLICATION) or ok_dis < len(REAL_DISTINCT):
        print("  WARNING: the calibrated cut misclassifies real observed cases.")

    results = {
        "schema": "hhgoa2026.task3.calibration.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "LFW funneled, >=3 images/identity, read from disk",
        "model": engine.model_id(),
        "metric": "cosine_l2normed",
        "n_identities": len(usable),
        "n_images_embedded": len(emb),
        "n_images_rejected": rejected,
        "max_images_per_identity": args.max_images_per_identity,
        "max_pairs_per_identity": args.max_pairs_per_identity,
        "n_pairs": len(pos) + len(neg),
        "n_positive_pairs": len(pos),
        "n_negative_pairs": len(neg),
        "threshold_cosine": round(threshold, 4),
        "far": far_at,
        "far_target": args.far,
        "tar": round(tar_at, 4),
        "auc": round(auc, 5),
        "same_photo_phash_max": int(phash_max),
        "strong_floor_cosine": round(strong_floor, 4),
        "genuine_p05": round(float(np.percentile(gen, 5)), 4),
        "genuine_p25": round(strong_floor, 4),
        "genuine_median": round(float(np.median(gen)), 4),
        "phash_same_photo_median": float(np.median(sp)) if len(sp) else None,
        "phash_same_photo_p95": float(np.percentile(sp, 95)) if len(sp) else None,
        "phash_diff_photo_median": float(np.median(dp)) if len(dp) else None,
        "phash_diff_photo_p05": float(np.percentile(dp, 5)) if len(dp) else None,
        "real_check_republication": f"{ok_rep}/{len(REAL_REPUBLICATION)}",
        "real_check_distinct": f"{ok_dis}/{len(REAL_DISTINCT)}",
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
    nz = fpr[fpr > 0]
    ax1.set_xlim(max(1e-5, nz.min() if nz.size else 1e-5), 1)
    ax1.set_xlabel("False accept rate (log)")
    ax1.set_ylabel("True accept rate")
    ax1.set_title(f"ROC -- {len(pos)} positive / {len(neg)} negative pairs")
    ax1.grid(alpha=.3)
    ax1.legend(loc="lower right", fontsize=8)

    if len(sp) and len(dp):
        bins = np.arange(0, 65, 2)
        ax2.hist(sp, bins=bins, alpha=.65, density=True,
                 label=f"same photograph (n={len(sp)})")
        ax2.hist(dp, bins=bins, alpha=.65, density=True,
                 label=f"different photograph (n={len(dp)})")
        ax2.axvline(phash_max, color="crimson", ls="--",
                    label=f"same_photo_phash_max = {phash_max}")
        ax2.set_xlabel("face-region pHash Hamming distance (64-bit)")
        ax2.set_ylabel("density")
        ax2.set_title("Same photograph vs different photograph")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "roc.png", dpi=140)

    print(f"\n  wrote {OUT_DIR/'results.json'}")
    print(f"  wrote {OUT_DIR/'roc.png'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
