r"""Is the error rate spread evenly across people, or concentrated on a few?

    .venv\Scripts\python.exe scripts/validate_per_identity.py

WHY THIS EXISTS
---------------
Every other accuracy figure in this project is an AGGREGATE. An aggregate false
reject rate of 2.4% is compatible with two very different worlds: every person
failing 2.4% of the time, or 94% of people never failing while a few fail half
the time. Those have completely different consequences for whoever is unlucky
enough to be in the second group, and only one of them is visible in a headline
number. This measures which world we are in.

WHY LFW AND NOT A DIVERSE WEB-SOURCED SET
-----------------------------------------
The first attempt built a demographically varied subject set from Wikimedia
Commons. It failed, twice, for the same reason: the identity labels could not be
trusted.

  * Commons FREE-TEXT SEARCH returns any file whose description merely MENTIONS
    the name. One such file -- a photograph of an entirely different person --
    produced all four of Satya Nadella's worst genuine pairs, scoring -0.128
    against himself.
  * Commons CATEGORIES were no better for most subjects. Politicians' categories
    are full of event photographs containing other attendees, so several
    subjects came back with a MEDIAN self-similarity near zero, which is only
    possible if the set is mostly other people.

Publishing a fairness number computed on contaminated labels would have been
worse than publishing none, so it was abandoned rather than massaged. LFW's
labels are reliable -- the directory IS the identity -- which is exactly the
property the claim needs.

WHAT THIS CAN AND CANNOT CONCLUDE
---------------------------------
It can show whether error concentrates on particular identities. It CANNOT
support a demographic claim: LFW is well documented as predominantly male and
predominantly light-skinned, so a per-identity spread measured on it says
nothing reliable about performance across demographic groups. That limitation is
the one stated in ETHICS.md and this does not remove it.

No race or ethnicity is inferred from any face here.
"""
from __future__ import annotations

import argparse
import collections
import itertools
import json
import pathlib
import sys
import time

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "calibration"
CACHE = OUT / "_embeddings_cache.npz"


def main() -> int:
    from app.face.encoder import cosine
    from app.face.matcher import CalibrationMissing, Thresholds

    ap = argparse.ArgumentParser()
    ap.add_argument("--min-images", type=int, default=3)
    args = ap.parse_args()

    if not CACHE.exists():
        print(f"\n  No embedding cache at {CACHE}.\n"
              "  Run scripts/calibrate_threshold.py first.\n")
        return 2
    try:
        th = Thresholds.load()
    except CalibrationMissing as e:
        print(f"\nCALIBRATION MISSING\n{e}\n")
        return 2

    z = np.load(CACHE, allow_pickle=True)
    by: dict[str, list] = collections.defaultdict(list)
    for k, e in zip(z["key"], z["emb"]):
        by[pathlib.Path(str(k)).parent.name].append(e)
    by = {k: v for k, v in by.items() if len(v) >= args.min_images}

    print(f"\n  identities            : {len(by)} (>= {args.min_images} images each)")
    print(f"  threshold under test  : {th.similarity:.4f}")

    per, allg = {}, []
    for ident, vs in by.items():
        g = [cosine(a, b) for a, b in itertools.combinations(vs, 2)]
        per[ident] = g
        allg += g
    G = np.array(allg)
    overall = float((G < th.similarity).mean())

    rows = sorted(
        ((float(np.mean(np.array(g) < th.similarity)), ident, float(np.median(g)), len(g))
         for ident, g in per.items()),
        reverse=True,
    )
    frrs = np.array([r[0] for r in rows])
    zero = int((frrs == 0).sum())

    print(f"  genuine pairs         : {G.size}")
    print(f"  OVERALL FRR           : {overall:.4f}")
    print()
    print("  THE AGGREGATE HIDES THE SHAPE")
    print(f"    identities with ZERO false rejects : {zero}/{len(by)}  ({zero/len(by):.1%})")
    print(f"    per-identity FRR  median {np.median(frrs):.4f}   "
          f"p90 {np.percentile(frrs, 90):.4f}   max {frrs.max():.4f}")
    print(f"    of {G.size} genuine pairs, {int((G < th.similarity).sum())} are false rejects,")
    print(f"    and they fall on just {len(by) - zero} of {len(by)} identities.")
    print()
    print("  WORST 12 IDENTITIES")
    print(f"    {'identity':30s} {'pairs':>6s} {'FRR':>7s} {'median':>9s}")
    print("    " + "-" * 56)
    for f, ident, med, n in rows[:12]:
        print(f"    {ident[:30]:30s} {n:6d} {f:7.3f} {med:+9.4f}")

    res = {
        "schema": "hhgoa2026.task3.per_identity.v1",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "dataset": "LFW funneled (directory = identity; labels reliable)",
        "threshold_cosine": th.similarity,
        "n_identities": len(by),
        "n_genuine_pairs": int(G.size),
        "overall_frr": round(overall, 5),
        "identities_with_zero_false_rejects": zero,
        "fraction_identities_clean": round(zero / len(by), 4),
        "per_identity_frr_median": round(float(np.median(frrs)), 5),
        "per_identity_frr_p90": round(float(np.percentile(frrs, 90)), 5),
        "per_identity_frr_max": round(float(frrs.max()), 5),
        "worst_identities": [
            {"identity": i, "pairs": n, "frr": round(f, 4), "median_genuine": round(m, 4)}
            for f, i, m, n in rows[:15]
        ],
        "caveat": ("LFW is predominantly male and light-skinned. A per-identity "
                   "spread measured on it does NOT support a demographic claim. "
                   "No race or ethnicity is inferred from any face."),
        "rejected_approach": ("A Wikimedia-sourced diverse subject set was built "
                              "and abandoned: neither free-text search nor "
                              "categories yield trustworthy identity labels, and "
                              "several subjects returned a median self-similarity "
                              "near zero. See this script's docstring."),
    }
    (OUT / "per_identity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(frrs, bins=np.linspace(0, 1, 21), color="#c0392b", alpha=.8)
    ax.set_yscale("log")
    ax.set_xlabel("per-identity false reject rate")
    ax.set_ylabel("identities (log)")
    ax.set_title(f"Error concentrates: {zero}/{len(by)} identities never fail, "
                 f"a few fail often\n(overall FRR {overall:.4f})")
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(OUT / "per_identity_frr.png", dpi=140)

    print(f"\n  wrote {OUT / 'per_identity.json'}")
    print(f"  wrote {OUT / 'per_identity_frr.png'}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
