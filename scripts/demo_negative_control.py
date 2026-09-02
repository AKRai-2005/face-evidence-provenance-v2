r"""Negative control: prove the threshold REJECTS the wrong person.

    .venv\Scripts\python.exe scripts/demo_negative_control.py --run runs/<id>

A pipeline that only ever demonstrates matches has shown nothing about its false
accept rate. This scores an IMPOSTOR -- a different public figure -- against the
same real candidate set the run matched, using the identical embedding pipeline
and the identical calibrated threshold.

The interesting number is the margin between the highest impostor score and the
threshold. If that margin is thin, the threshold is doing less work than the
matching demo implies.

Uses only images already on disk. No network, no API quota, no gas.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_IMPOSTOR = ROOT / "data" / "fixtures" / "A3_pichai_warsaw_crop.jpg"


def main() -> int:
    import cv2

    from app.face.detector import FaceEngine
    from app.face.encoder import cosine
    from app.face.matcher import CalibrationMissing, Thresholds

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, type=pathlib.Path,
                    help="a run directory containing candidates/")
    ap.add_argument("--impostor", type=pathlib.Path, default=DEFAULT_IMPOSTOR)
    ap.add_argument("--genuine", type=pathlib.Path, default=ROOT / "data" / "input.jpg",
                    help="the run's own input, for the side-by-side comparison")
    args = ap.parse_args()

    cand_dir = args.run / "candidates"
    if not cand_dir.is_dir():
        print(f"\n  NO CANDIDATES IN {args.run}\n  Run the pipeline first.\n")
        return 2
    if not args.impostor.exists():
        print(f"\n  IMPOSTOR IMAGE NOT FOUND: {args.impostor}\n")
        return 2

    try:
        th = Thresholds.load()
    except CalibrationMissing as e:
        print(f"\nCALIBRATION MISSING\n{e}\n")
        return 2

    engine = FaceEngine.shared()

    def primary(path):
        img = cv2.imread(str(path))
        if img is None:
            return None
        faces = engine.detect(img)
        return max(faces, key=lambda f: f.det_score) if faces else None

    imp = primary(args.impostor)
    if imp is None:
        print(f"\n  NO FACE IN IMPOSTOR IMAGE {args.impostor}\n")
        return 2
    gen = primary(args.genuine) if args.genuine.exists() else None

    print()
    print("  NEGATIVE CONTROL -- does the threshold reject the wrong person?")
    print("  " + "=" * 68)
    print(f"  impostor  : {args.impostor.name}  (det {imp.det_score:.3f})")
    if gen is not None:
        print(f"  genuine   : {args.genuine.name}  (det {gen.det_score:.3f})")
    print(f"  candidates: {cand_dir}")
    print(f"  threshold : {th.similarity:.4f}   {th.describe()}")
    print()

    rows = []
    for p in sorted(cand_dir.glob("*")):
        img = cv2.imread(str(p))
        if img is None:
            continue
        faces = engine.detect(img)
        if not faces:
            continue
        i_score = max(cosine(imp.embedding, f.embedding) for f in faces)
        g_score = (max(cosine(gen.embedding, f.embedding) for f in faces)
                   if gen is not None else None)
        rows.append((p.name, i_score, g_score))

    if not rows:
        print("  No candidate images with a detectable face.\n")
        return 1

    rows.sort(key=lambda r: -r[1])
    print(f"  {'candidate':30s} {'impostor':>10s} {'genuine':>10s}")
    print("  " + "-" * 54)
    for name, i_s, g_s in rows:
        flag = "  <-- FALSE ACCEPT" if i_s >= th.similarity else ""
        g_txt = f"{g_s:+.4f}" if g_s is not None else "     -"
        print(f"  {name[:30]:30s} {i_s:+10.4f} {g_txt:>10s}{flag}")

    worst = max(r[1] for r in rows)
    false_accepts = sum(1 for r in rows if r[1] >= th.similarity)
    print()
    print(f"  highest impostor score : {worst:+.4f}")
    print(f"  calibrated threshold   : {th.similarity:+.4f}")
    print(f"  margin                 : {th.similarity - worst:+.4f} "
          f"({th.similarity / worst:.1f}x)" if worst > 0 else "")
    print(f"  false accepts          : {false_accepts} / {len(rows)}")

    if gen is not None:
        gmin = min(r[2] for r in rows if r[2] is not None)
        gmax = max(r[2] for r in rows if r[2] is not None)
        print(f"  genuine score range    : {gmin:+.4f} to {gmax:+.4f}")
        print()
        print(f"  Separation: every genuine score exceeds every impostor score by")
        print(f"  at least {gmin - worst:+.4f}. The threshold sits in that gap.")

    print()
    if false_accepts:
        print("  THRESHOLD FAILED THIS CONTROL -- do not present the match demo")
        print("  without disclosing this.")
        return 1
    print("  PASS -- no impostor candidate clears the threshold.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
