r"""Attempted demographic false-accept audit -- and why its number is NOT published.

    .venv\Scripts\python.exe scripts/audit_demographic_far.py

READ THIS BEFORE QUOTING ANY FIGURE THIS SCRIPT PRINTS.

The goal was a demographic false-accept rate. False rejects need identity-paired
data labelled by group (RFW, BUPT-Balancedface), which is access-gated; false
accepts need only different-person pairs with group labels, which FairFace
provides freely. False accepts are also the safety-critical direction: a false
reject inconveniences someone, a false accept attaches a stranger's face to
someone else's evidence record.

Four attempts were made. Each was defeated by a different confound, and the
measurement is reported here as UNRESOLVED rather than dressed up as a result.

1. WITHIN-GROUP PAIRS, no deduplication.
   FAR 0.00726 with an apparent 15x spread between groups. Inspecting the
   highest-scoring "impostor" pairs showed the same woman photographed twice at
   one event, the same child in the same hat and jacket, and that woman twice
   more. FairFace's validation split repeats individuals, so those were genuine
   same-person pairs. Discarded.

2. CROSS-GROUP PAIRS, on the assumption that different race labels imply
   different people.
   FAR 0.00131. Then the gender cut -- same gender, DIFFERENT race -- still
   reported a max of 0.8295, which is the duplicate child pair. FairFace assigns
   the same individual different race labels across their images, so the
   assumption is false. Discarded.

3. CROSS-GROUP PAIRS with image-level deduplication at pHash <= 22.
   Rejected 76% of the sample and finished with 94 faces. The threshold was
   borrowed from the FACE-region pHash scale and is far too loose for a
   whole-image hash: measured across 378 sampled images the pair distribution
   runs min 10, p1 20, median 30 on a 64-bit hash, so 22 flags 4.1% of
   legitimately distinct pairs. Discarded.

4. CROSS-GROUP PAIRS with deduplication at a measured pHash <= 10.
   FAR 0.00226, 19 false accepts in 8,400 pairs, max impostor +0.6400 -- a score
   the pipeline would call a STRONG match. That figure cannot be trusted either.
   Image-level dedup catches the same shot twice; it cannot catch the same
   person photographed at a different event, which is exactly what FairFace's
   repeats look like. Deduplicating on FACE similarity instead would be
   circular: removing high-scoring pairs is removing the false accepts you set
   out to count.

THE TERMINAL FINDING. FairFace cannot support a reliable false-accept
measurement for this purpose. It repeats individuals, labels those repeats
inconsistently, and the only dedup signal strong enough to catch the repeats is
the very quantity under measurement. A trustworthy answer needs a dataset with
identity labels -- which is precisely the access-gated data the exercise was
trying to work around.

WHAT THIS PROJECT CLAIMS INSTEAD. The negative control in
scripts/demo_negative_control.py measures false accepts on real web candidates
with KNOWN identities: 0 of 11 at a 3x threshold margin. That is trustworthy and
it is not demographic. The demographic question stays open, and ETHICS.md says so.

The script still runs, and prints its numbers with the caveats attached, because
the attempt and its failure modes are the useful artefact.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import pathlib
import sys
import time

import imagehash
import numpy as np
import requests
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "calibration"
CACHE = ROOT / "data" / "fairface" / "images"
ROWS_API = "https://datasets-server.huggingface.co/rows"
DATASET = {"dataset": "HuggingFaceM4/FairFace", "config": "1.25", "split": "validation"}
RACES = ["East Asian", "Indian", "Black", "White",
         "Middle Eastern", "Latino_Hispanic", "Southeast Asian"]
GENDERS = ["Male", "Female"]

# Whole-image pHash distance below which two images are treated as the same
# shoot. Measured, not guessed: across 378 sampled FairFace images the pair
# distance distribution runs min 10, p1 20, median 30, max 50 on a 64-bit hash,
# and identical files score 0. A cutoff of 10 flags one pair; 22 flagged 4.1% of
# them and starved the sample -- 22 was borrowed from the FACE-region pHash
# scale, which is a different measurement on a different crop.
IMAGE_DUP_MAX = 10


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
    # Over-sample. Dedup happens during embedding, so collecting exactly
    # per_group URLs per group guarantees finishing short of target -- an
    # earlier version did exactly that and ended with 94 usable faces.
    target_urls = args.per_group * 3
    wanted = {r: [] for r in RACES}
    offset, page = 0, 100
    while offset < args.scan and any(len(v) < target_urls for v in wanted.values()):
        try:
            rows = fetch_rows(offset, page)
        except requests.RequestException as e:
            print(f"  rows API failed at offset {offset}: {e}")
            break
        if not rows:
            break
        for row in rows:
            race = RACES[int(row["race"])]
            if len(wanted[race]) >= target_urls:
                continue
            src = (row.get("image") or {}).get("src")
            if src:
                wanted[race].append((src, GENDERS[int(row["gender"])]))
        offset += page
    print(f"  scanned {offset} rows")

    # ---- embed ----------------------------------------------------------
    embs: dict[str, list] = {r: [] for r in RACES}
    genders: dict[str, list] = {r: [] for r in RACES}
    t0, no_face, dup_file, dup_shoot = time.time(), 0, 0, 0
    seen_files: set[str] = set()
    seen_images: list = []
    for race, items in wanted.items():
        for src, gender in items:
            if len(embs[race]) >= args.per_group:
                break
            b = fetch_image(src)
            if b is None:
                continue
            img = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # DEDUPLICATION, and it is not optional.
            # Two confounds, both of which inflated earlier versions of this
            # measurement and neither of which involves the face model, so
            # filtering on them is not circular:
            #   1. the same FILE arriving twice -- HuggingFace's cached-assets
            #      URLs rotate between API calls, so one image can be fetched
            #      under two names. Caught by SHA-256.
            #   2. the same SHOOT appearing twice -- FairFace's validation split
            #      contains repeated individuals (verified by eye: the same
            #      woman at one event, the same child in the same hat). Caught
            #      by whole-image pHash, which is a property of the pixels, not
            #      of ArcFace.
            # Filtering on face similarity instead would be circular: dropping
            # high-scoring pairs trivially lowers the false accept rate.
            digest = hashlib.sha256(b).hexdigest()
            if digest in seen_files:
                dup_file += 1
                continue
            ih = imagehash.phash(Image.open(io.BytesIO(b)).convert("RGB"))
            if any((ih - prev) <= IMAGE_DUP_MAX for prev in seen_images):
                dup_shoot += 1
                continue

            faces = engine.detect(img)
            if not faces:
                no_face += 1
                continue
            seen_files.add(digest)
            seen_images.append(ih)
            embs[race].append(max(faces, key=lambda f: f.det_score).embedding)
            genders[race].append(gender)
        print(f"    {race:18s} {len(embs[race]):3d} embedded")
    total = sum(len(v) for v in embs.values())
    print(f"  {total} faces in {time.time()-t0:.0f}s ({no_face} with no detectable face)\n")

    if total < 100:
        print("  Too few faces to draw a conclusion.\n")
        return 1

    # ---- CROSS-group impostor pairs -------------------------------------
    # Different race labels => almost certainly different people, which is what
    # makes these usable when within-group pairs are contaminated by repeats.
    print("  CROSS-GROUP IMPOSTOR PAIRS  (different labelled groups => different people)")
    print(f"  {'group':18s} {'n':>4s} {'pairs':>7s} {'max':>9s} {'p99.9':>9s} {'FAR':>9s}")
    print("  " + "-" * 62)
    per_group, all_scores = {}, []
    for race in RACES:
        v = embs[race]
        if len(v) < 8:
            continue
        others = [(o, e) for o in RACES if o != race for e in embs[o]]
        if not others:
            continue
        pairs = [(i, j) for i in range(len(v)) for j in range(len(others))]
        if len(pairs) > args.max_pairs_per_group:
            idx = rng.choice(len(pairs), args.max_pairs_per_group, replace=False)
            pairs = [pairs[i] for i in idx]
        s = np.array([cosine(v[i], others[j][1]) for i, j in pairs])
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
    # Is the per-group spread real, or Poisson noise on a handful of events?
    # These are counts of rare events: 11 false accepts across seven groups.
    # Under a constant rate the expected count per group is ~1.6, and the
    # Poisson spread around that covers 0 to ~5 -- which is every value observed.
    # Running this script twice flipped the group ordering outright. Reporting a
    # per-group table from that would be fiction dressed as measurement.
    events = np.array([round(g["far"] * g["pairs"]) for g in per_group.values()])
    expected = overall * float(np.mean([g["pairs"] for g in per_group.values()]))
    resolvable = expected >= 25          # ~5 sigma before a 2x difference shows

    print("  IS THE PER-GROUP SPREAD REAL?")
    print(f"    false accepts per group : {', '.join(str(int(e)) for e in sorted(events))}")
    print(f"    expected under a constant rate: {expected:.1f} per group")
    if not resolvable:
        need = int(25 / max(overall, 1e-9))
        print("    NOT RESOLVABLE at this sample size. Distinguishing a genuine")
        print(f"    2x difference at FAR {overall:.5f} needs roughly {need:,} pairs")
        print(f"    per group; this has {int(np.mean([g['pairs'] for g in per_group.values()])):,}.")
        print( "    The per-group numbers above are reported for completeness and")
        print( "    should NOT be read as a bias ranking.")
    else:
        print(f"    per-group FAR   min {fars.min():.5f}   max {fars.max():.5f}")
    print(f"    headroom to threshold from the worst score: "
          f"{th.similarity - maxes.max():+.4f}")

    # ---- gender cut ------------------------------------------------------
    # Same gender, DIFFERENT race -- same reasoning as the race cut above.
    # Pooling all same-gender pairs re-admits the repeated individuals, and did:
    # an earlier version reported a max of 0.8295, which is one of the duplicate
    # pairs rather than a false accept.
    print("\n  BY GENDER (same gender, different race, so still different people)")
    by_gender: dict[str, list] = {g: [] for g in GENDERS}
    for race in RACES:
        for emb, gen in zip(embs[race], genders[race]):
            by_gender[gen].append((race, emb))
    for g in GENDERS:
        vs = by_gender[g]
        if len(vs) < 8:
            continue
        pairs = [(i, j) for i in range(len(vs)) for j in range(i + 1, len(vs))
                 if vs[i][0] != vs[j][0]]
        if not pairs:
            continue
        if len(pairs) > args.max_pairs_per_group * 2:
            idx = rng.choice(len(pairs), args.max_pairs_per_group * 2, replace=False)
            pairs = [pairs[i] for i in idx]
        s = np.array([cosine(vs[a][1], vs[b][1]) for a, b in pairs])
        print(f"    {g:8s} n={len(vs):4d}  pairs={s.size:6d}  max {s.max():+.4f}  "
              f"FAR {float((s >= th.similarity).mean()):.5f}")

    res = {
        "schema": "hhgoa2026.task3.demographic_far.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "FairFace 1.25 validation (labels supplied by the dataset)",
        # Marker names match calibration/demographics.json and are asserted by
        # tests/test_docs_consistency.py, so re-running this script cannot
        # quietly produce a file that reads as a result.
        "RETRACTED": True,
        "DO_NOT_CITE": (
            "This FAR was withdrawn from the README and ETHICS.md. It is kept "
            "as the record of a discarded attempt, not as a result."),
        "PUBLISHABLE": False,
        "retraction_reason": (
            "FairFace repeats individuals and labels those repeats "
            "inconsistently across their images. Image-level dedup cannot catch "
            "a repeat photographed at a different event, and face-level dedup "
            "would be circular. Four attempts, each defeated by a different "
            "confound -- see the script docstring. Use "
            "scripts/demo_negative_control.py for a trustworthy false-accept "
            "figure on known identities."),
        "measures": ("cross-group false accept rate only. False rejects need "
                     "identity-paired data (RFW/BUPT), which is access-gated. "
                     "WITHIN-group FAR is not reported: FairFace repeats "
                     "individuals, so same-group pairs are contaminated by "
                     "genuine same-person pairs -- verified by inspecting the "
                     "top-scoring pairs, which are visibly the same people."),
        "threshold_cosine": th.similarity,
        "n_faces": total,
        "n_impostor_pairs": int(A.size),
        "overall_far": round(overall, 6),
        "overall_max_impostor": round(float(A.max()), 4),
        "per_group": {k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                          for kk, vv in v.items()} for k, v in per_group.items()},
        "far_spread": round(float(fars.max() - fars.min()), 6),
        "per_group_spread_is_noise": True,
        "per_group_spread_note": (
            "11 false accepts across 7 groups (0,0,1,2,2,2,4). Expected ~1.6 per "
            "group under a constant rate; Poisson variation covers every observed "
            "value, and re-running flipped the ordering. Do NOT read the "
            "per_group FAR values as a bias ranking -- the aggregate is the only "
            "figure this sample supports."),
        "caveat": ("FairFace race/gender labels are annotation conventions, not "
                   "biological facts, and no attribute is inferred from any face "
                   "here. Cross-group pairs are the EASY case; this understates "
                   "the disparity that within-group pairs would reveal, and that "
                   "measurement is not available from this dataset."),
    }
    (OUT / "demographic_far.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\n  wrote {OUT / 'demographic_far.json'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
