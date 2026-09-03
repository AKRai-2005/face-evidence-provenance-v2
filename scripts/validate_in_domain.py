r"""In-domain validation: is the LFW-calibrated threshold valid on the OPEN WEB?

    .venv\Scripts\python.exe scripts/validate_in_domain.py --run <dir> --run <dir> ...

WHY THIS EXISTS

The decision threshold is calibrated on LFW: funneled, frontal, 250x250 press
photography, one curated distribution. This pipeline is then pointed at whatever
Google Lens returns -- arbitrary resolution, pose, crop, compression and
provenance. "FAR 7.83e-04" is therefore a claim about LFW, not a claim about
what the tool actually does.

That gap between calibration domain and deployment domain is where face
recognition systems quietly fail in the real world, and quoting a benchmark
number as if it were an operating guarantee is the standard way to be wrong in
public. So we measure the operating point on the deployment distribution itself.

METHOD

Each run directory is a real search for ONE subject, so its downloaded candidate
images are web-sourced photographs of that person. For each candidate we take
the face that best matches that run's own input -- exactly the face the pipeline
would have scored -- giving a set of web-domain "identity faces" per subject.

    genuine pairs   = within-subject pairs of those faces
    impostor pairs  = across-subject pairs

Both drawn entirely from images the pipeline actually downloaded. No benchmark,
no curation, no re-touching.

Requires two or more runs on DIFFERENT subjects. Uses only files already on
disk: no network, no API quota, no gas.
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _subject_label(run_dir: pathlib.Path) -> str:
    meta = run_dir / "run.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text(encoding="utf-8"))["input_name"]
        except Exception:
            pass
    return run_dir.name


def main() -> int:
    import cv2

    from app.face.detector import FaceEngine
    from app.face.encoder import cosine
    from app.face.matcher import CalibrationMissing, Thresholds

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, type=pathlib.Path,
                    help="run directory (repeat; needs >=2 on different subjects)")
    ap.add_argument("--json-out", type=pathlib.Path,
                    default=ROOT / "calibration" / "in_domain.json")
    args = ap.parse_args()

    if len(args.run) < 2:
        print("\n  Need at least two --run directories on DIFFERENT subjects.\n")
        return 2

    try:
        th = Thresholds.load()
    except CalibrationMissing as e:
        print(f"\nCALIBRATION MISSING\n{e}\n")
        return 2

    engine = FaceEngine.shared()

    subjects: dict[str, list[tuple[str, np.ndarray]]] = {}
    for run_dir in args.run:
        cand_dir = run_dir / "candidates"
        inp = next((run_dir / "input").glob("*"), None) if (run_dir / "input").is_dir() else None
        if not cand_dir.is_dir() or inp is None:
            print(f"  SKIP {run_dir} (no candidates/ or input/)")
            continue

        img = cv2.imread(str(inp))
        faces = engine.detect(img) if img is not None else []
        if not faces:
            print(f"  SKIP {run_dir} (no face in input)")
            continue
        ref = max(faces, key=lambda f: f.det_score).embedding

        label = _subject_label(run_dir)
        got = []
        for p in sorted(cand_dir.glob("*")):
            cimg = cv2.imread(str(p))
            if cimg is None:
                continue
            cfaces = engine.detect(cimg)
            if not cfaces:
                continue
            # the face the pipeline itself would have scored
            best = max(cfaces, key=lambda f: cosine(ref, f.embedding))
            got.append((p.name, best.embedding))
        if got:
            # EXTEND, never assign. Two runs on the same subject share an input
            # filename, and assigning would silently discard one run's faces --
            # the analysis would then quietly rest on less data than it claims.
            subjects.setdefault(label, []).extend(got)
            print(f"  {label:28s} +{len(got):2d} web-sourced identity faces "
                  f"from {run_dir.name}  (subject total {len(subjects[label])})")

    if len(subjects) < 2:
        print("\n  Need two subjects with usable faces.\n")
        return 2

    genuine, impostor = [], []
    for label, faces in subjects.items():
        for (_, a), (_, b) in itertools.combinations(faces, 2):
            genuine.append(cosine(a, b))
    for l1, l2 in itertools.combinations(subjects, 2):
        for (_, a) in subjects[l1]:
            for (_, b) in subjects[l2]:
                impostor.append(cosine(a, b))

    g, i = np.array(genuine), np.array(impostor)
    t = th.similarity

    fa = int((i >= t).sum())
    fr = int((g < t).sum())
    far = fa / len(i)
    frr = fr / len(g)

    print()
    print("  IN-DOMAIN VALIDATION -- web-sourced faces, not a benchmark")
    print("  " + "=" * 68)
    print(f"  subjects        : {len(subjects)}")
    print(f"  genuine pairs   : {len(g):6d}   "
          f"min {g.min():+.4f}  median {np.median(g):+.4f}  max {g.max():+.4f}")
    print(f"  impostor pairs  : {len(i):6d}   "
          f"min {i.min():+.4f}  median {np.median(i):+.4f}  max {i.max():+.4f}")
    print()
    print(f"  LFW-calibrated threshold : {t:+.4f}")
    print(f"  in-domain false accepts  : {fa} / {len(i)}   (FAR {far:.2e})")
    print(f"  in-domain false rejects  : {fr} / {len(g)}   (FRR {frr:.2e})")
    print()

    margin = g.min() - i.max()
    print(f"  worst genuine  : {g.min():+.4f}")
    print(f"  worst impostor : {i.max():+.4f}")
    print(f"  separation     : {margin:+.4f}")

    if margin > 0:
        lo, hi = i.max(), g.min()
        print(f"  Every genuine pair outscores every impostor pair. Any threshold")
        print(f"  in ({lo:.4f}, {hi:.4f}] separates them perfectly on this data;")
        print(f"  the calibrated {t:.4f} sits "
              f"{'INSIDE' if lo < t <= hi else 'OUTSIDE'} that window.")
    else:
        print("  Distributions OVERLAP in this domain -- the benchmark threshold")
        print("  does not transfer cleanly. Report this rather than the LFW figure.")

    out = {
        "schema": "hhgoa2026.task3.in_domain.v1",
        "subjects": {k: len(v) for k, v in subjects.items()},
        "n_genuine_pairs": len(g),
        "n_impostor_pairs": len(i),
        "genuine_min": round(float(g.min()), 4),
        "genuine_median": round(float(np.median(g)), 4),
        "impostor_max": round(float(i.max()), 4),
        "impostor_median": round(float(np.median(i)), 4),
        "separation": round(float(margin), 4),
        "threshold_cosine": t,
        "in_domain_false_accepts": fa,
        "in_domain_false_rejects": fr,
        "in_domain_far": far,
        "in_domain_frr": frr,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote {args.json_out.relative_to(ROOT)}\n")
    return 0 if fa == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
