r"""Does the false-accept rate hold across demographic groups?

    .venv\Scripts\python.exe scripts/audit_demographic_far.py

WHY THIS IS POSSIBLE WHEN A FULL AUDIT IS NOT
---------------------------------------------
A complete fairness audit needs BOTH error directions, and false rejects require
identity-labelled pairs (multiple photographs of the same person, labelled by
group). Those datasets -- RFW, BUPT-Balancedface -- are access-gated behind
signed agreements and are not obtainable here.

False ACCEPTS need only different-person pairs with group labels, and that is
obtainable: FairFace is freely available, and every image is a distinct person.
So this measures the half that is measurable, and says plainly that it is a half.

It is also the safety-critical half. A false reject inconveniences someone; a
false accept attaches a stranger's face to someone else's evidence record.

ON THE LABELS
-------------
Race, gender and age labels come from FairFace itself. Nothing here infers them
from a face -- that inference is the capability this project argues against, and
using it to make a fairness claim would be self-defeating. FairFace's categories
are annotation conventions, not biological facts, and are reported as given.

METHOD
------
* Sample N images per race group from the FairFace validation split, via the
  HuggingFace datasets-server rows API (no 236MB parquet download).
* Embed each with the exact pipeline used at run time.
* Form WITHIN-GROUP impostor pairs -- different people of the same labelled
  group. These are the hard case: within-group false accepts are where
  demographic disparity in face recognition is known to show up.
* Report FAR at the SHIPPED threshold, per group. Nothing is re-tuned here.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import itertools
import json
import pathlib
import sys
import time

import numpy as np
import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "calibration"
CACHE = ROOT / "data" / "fairface" / "images"
ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "HuggingFaceM4/FairFace", "config": "1.25", "split": "validation"}
RACES = ["East Asian", "Indian", "Black", "White",
         "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
GENDERS = ["Male", "Female"]


def fetch_rows(offset: int, length: int) -> list[dict]:
    r = requests.get(ROWS_API, params={**DATASET, "offset": offset, "length": length},
                     timeout=120)
    r.raise_for_status()
    return [x["row"] for x in r.json().get("rows", [])]


def fetch_image(url: str) -> bytes | None:
    p = CACHE / (hashlib.sha256(url.encode()).hexdigest()[:16] + ".jpg")
    if p.exists():
        return p.read_bytes()
    try:
        r = requests.get(url, timeout=45)
        if r.status_code != 200 or len(r.content) < 512:
            return None
    except requests.RequestException:
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    p.write_bytes(r.content)
    return r.content


def main() -> int:
    import cv2

    from app.face.detector import FaceEngine
    from app.face.encoder import cosine
    from app.face.matcher import CalibrationMissing, Thresholds

    ap = argparse.ArgumentParser()
    ap.add_argument("--per-group", type=int, default=55,
                    help="images to embed per race group")
    ap.add_argument("--scan", type=int, default=2600,
                    help="rows to scan for a balanced sample")
    ap.add_argument("--max-pairs-per-group", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20260904)
    args = ap.parse_args()

    try:
        th = Thresholds.load()
    except CalibrationMissing as e:
        print(f"\nCALIBRATION MISSING\n{e}\n")
        return 2

    rng = np.random.default_rng(args.seed)
    engine = FaceEngine.shared()

    print(f"\n  threshold under test : {th.similarity:.4f}  (shipped, not re-tuned)")
    print(f"  target               : {args.per_group} images per race group\n")

    # ---- collect a balanced sample -------------------------------------
    wanted = {r: [] for r in RACES}
    offset, page = 0, 100
    while offset < args.scan and any(len(v) < args.per_group for v in wanted.values()):
        try:
            rows = fetch_rows(offset, page)
        except requests.RequestException as e:
            print(f"  rows API failed at offset {offset}: {e}")
            break
        if not rows:
            break
        for row in rows:
            race = RACES[int(row["race"])]
            if len(wanted[race]) >= args.per_group:
                continue
            src = (row.get("image") or {}).get("src")
            if src:
                wanted[race].append((src, GENDERS[int(row["gender"])]))
        offset += page
    print(f"  scanned {offset} rows")

    # ---- embed ----------------------------------------------------------
    embs: dict[str, list] = {r: [] for r in RACES}
    genders: dict[str, list] = {r: [] for r in RACES}
    t0, no_face = time.time(), 0
    for race, items in wanted.items():
        for src, gender in items:
            b = fetch_image(src)
            if b is None:
                continue
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            faces = engine.detect(img)
            if not faces:
                no_face += 1
                continue
            embs[race].append(max(faces, key=lambda f: f.det_score).embedding)
            genders[race].append(gender)
        print(f"    {race:18s} {len(embs[race]):3d} embedded")
    total = sum(len(v) for v in embs.values())
    print(f"  {total} faces in {time.time()-t0:.0f}s ({no_face} with no detectable face)\n")

    if total < 100:
        print("  Too few faces to draw a conclusion.\n")
        return 1

    # ---- within-group impostor pairs -------------------------------------
    print("  WITHIN-GROUP IMPOSTOR PAIRS  (different people, same labelled group)")
    print(f"  {'group':18s} {'n':>4s} {'pairs':>7s} {'max':>9s} {'p99.9':>9s} {'FAR':>9s}")
    print("  " + "-" * 62)
    per_group, all_scores = {}, []
    for race in RACES:
        v = embs[race]
        if len(v) < 8:
            continue
        pairs = list(itertools.combinations(range(len(v)), 2))
        if len(pairs) > args.max_pairs_per_group:
            idx = rng.choice(len(pairs), args.max_pairs_per_group, replace=False)
            pairs = [pairs[i] for i in idx]
        s = np.array([cosine(v[a], v[b]) for a, b in pairs])
        far = float((s >= th.similarity).mean())
        per_group[race] = dict(n=len(v), pairs=int(s.size), max=float(s.max()),
                               p999=float(np.percentile(s, 99.9)), far=far)
        all_scores.append(s)
        flag = "  <-- above threshold" if s.max() >= th.similarity else ""
        print(f"  {race:18s} {len(v):4d} {s.size:7d} {s.max():+9.4f} "
              f"{np.percentile(s,99.9):+9.4f} {far:9.5f}{flag}")

    A = np.concatenate(all_scores)
    overall = float((A >= th.similarity).mean())
    fars = np.array([g["far"] for g in per_group.values()])
    maxes = np.array([g["max"] for g in per_group.values()])

    print("  " + "-" * 62)
    print(f"  {'ALL GROUPS':18s} {'':4s} {A.size:7d} {A.max():+9.4f} "
          f"{np.percentile(A,99.9):+9.4f} {overall:9.5f}")
    print()
    print("  DISPARITY")
    print(f"    per-group FAR   min {fars.min():.5f}   max {fars.max():.5f}")
    print(f"    per-group max   min {maxes.min():+.4f}  max {maxes.max():+.4f}")
    print(f"    worst group is {'the same as' if fars.max()==fars.min() else 'worse than'} "
          f"the best by {fars.max()-fars.min():.5f} FAR")
    print(f"    headroom to threshold from the worst score: "
          f"{th.similarity - maxes.max():+.4f}")

    # ---- gender cut ------------------------------------------------------
    print("\n  BY GENDER (labels from the dataset, not inferred)")
    for g in GENDERS:
        vs = [e for race in RACES for e, gg in zip(embs[race], genders[race]) if gg == g]
        if len(vs) < 8:
            continue
        pairs = list(itertools.combinations(range(len(vs)), 2))
        if len(pairs) > args.max_pairs_per_group * 2:
            idx = rng.choice(len(pairs), args.max_pairs_per_group * 2, replace=False)
            pairs = [pairs[i] for i in idx]
        s = np.array([cosine(vs[a], vs[b]) for a, b in pairs])
        print(f"    {g:8s} n={len(vs):4d}  pairs={s.size:6d}  max {s.max():+.4f}  "
              f"FAR {float((s >= th.similarity).mean()):.5f}")

    res = {
        "schema": "hhgoa2026.task3.demographic_far.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "FairFace 1.25 validation (labels supplied by the dataset)",
        "measures": "false accept rate only; false rejects need identity-paired "
                    "data (RFW/BUPT), which is access-gated and not used here",
        "threshold_cosine": th.similarity,
        "n_faces": total,
        "n_impostor_pairs": int(A.size),
        "overall_far": round(overall, 6),
        "overall_max_impostor": round(float(A.max()), 4),
        "per_group": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in per_group.items()},
        "far_spread": round(float(fars.max() - fars.min()), 6),
        "caveat": ("FairFace race/gender labels are annotation conventions, not "
                   "biological facts, and no attribute is inferred from any face "
                   "here. Within-group impostor pairs assume FairFace images are "
                   "distinct individuals, which is how the dataset is built."),
    }
    (OUT / "demographic_far.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT / 'demographic_far.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
