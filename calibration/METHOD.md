# Threshold calibration method

Reproduce with:

```bash
python scripts/calibrate_threshold.py
```

Outputs `results.json` (machine-readable, loaded by the pipeline) and `roc.png`.
`app/face/matcher.py` reads the threshold from `results.json` and **refuses to
run if the file is absent** — there is no hardcoded fallback anywhere in the
pipeline, by design.

---

## Why two thresholds

The pipeline makes two separate decisions, so it needs two separate numbers.

1. **`threshold_cosine` — "is this the same person?"**
   Cosine similarity on L2-normalised ArcFace embeddings, cut at a target false
   accept rate.

2. **`same_photo_phash_max` — "is this a *different photograph*, or the same one
   republished?"**
   Hamming distance between pHashes of the *aligned face region*.

The second one is what makes the project's central claim defensible, and it is
not in any standard benchmark, so it is calibrated here from first principles.

---

## Data

**LFW**, funneled, colour, full 250×250 (no sklearn slicing or downscaling),
`min_faces_per_person=3`, fetched via `sklearn.datasets.fetch_lfw_people`.

Each image is embedded **once** with the exact pipeline used at run time
(`FaceEngine.detect` → highest-confidence face → `normed_embedding`). Embeddings
are cached to `_embeddings_cache.npz` so re-calibration does not repeat the
~15-minute embedding pass.

### Sample-size correction

FAR = 1e-3 means one false accept in a thousand. **It cannot be *observed* with
fewer than ~1000 negative pairs.** LFW's standard `pairs` test split provides
500 same and 500 different — so a "FAR 1e-3" figure quoted off that split is an
extrapolation, not a measurement.

Because embeddings are computed once and pairing is free, pairs are instead
constructed combinatorially from identities, which affords tens of thousands of
negatives and resolves FAR to ~2e-5.

### Class-balance correction

LFW is severely imbalanced — George W. Bush alone has 530 images. A first pass
that took *every* within-identity combination produced 236,225 positive pairs,
of which C(530,2) ≈ 140,000 (**59%**) came from that one person. The reported
TAR was effectively "TAR on George W. Bush", and AUC came out at 0.983 where
ArcFace on LFW should reach ~0.999.

The calibration therefore caps **images per identity** (`--max-images-per-identity`,
default 6) and **positive pairs per identity** (`--max-pairs-per-identity`,
default 15), so no single subject dominates the operating point.

---

## Similarity threshold

Standard ROC over the pooled positive and negative pair scores
(`sklearn.metrics.roc_curve`). The threshold is taken at the point whose FPR is
closest to the target FAR; the realised FAR is reported alongside it rather than
the target, because with discrete pairs they are not identical.

`results.json` records `threshold_cosine`, the **realised** `far`, `tar`, `auc`,
and the pair counts, so the operating point can be audited rather than trusted.

---

## Face-region pHash threshold

### Why not whole-image pHash

The build plan called for perceptual-hash distance between input and match, on
the assumption that a high distance indicates a different photograph. Measured
against real Google Lens results, **whole-image pHash cannot make that
distinction at all**: because the query is a *crop*, the global hash is
randomised, and every candidate — republications and genuinely different
photographs alike — landed at 110–138 out of 256.

Hashing the **aligned face region** (box expanded 30%, resampled to a fixed
128×128 square, 64-bit pHash) normalises position and scale, so the hash becomes
comparable across different crops of the same photograph. On the same real
candidates that separates cleanly:

| | cosine | face-pHash distance |
|---|---|---|
| same photograph, republished | 0.968–0.985 | **4–14** |
| different photograph, same subject | 0.637–0.841 | **26–40** |

### How the cut is chosen

Two distributions are built:

- **same photograph** — each subject image against realistic republication
  transforms of itself: rescaling (0.35×, 0.6×, 1.4×), two re-crops, JPEG
  re-compression at q45 and q25, unsharp masking, two tonal regrades, a 2°
  rotation, and a greyscale conversion.
- **different photograph** — LFW positive pairs, which are by construction
  different photographs of the same person.

The threshold sweeps 0–64 and maximises
`P(same ≤ t) + P(different > t)`, **breaking ties toward the higher cut.**

Two deliberate choices there:

*A sweep, not a percentile.* An earlier version took the same-photo 95th
percentile. That ignores where the other distribution sits, and with a gentler
transform set it produced `same_photo_phash_max = 6` — **below the 4–14 range of
real republications measured in Phase 0.** That cut would have promoted a
genuinely republished press photo to `DISTINCT-PHOTOGRAPH`, inflating the
project's strongest claim.

*Ties break high.* The asymmetry is intentional: labelling a republication as a
distinct photograph overstates the evidence, while the reverse merely
understates it. Only one of those is a claim we would have to retract.

### Real-world check

The calibration additionally re-classifies the twelve real Lens candidates from
Phase 0, whose labels were established by eye, and prints how many of each class
the chosen cut gets right. If the synthetic cut misclassifies them, it says so.
Synthetic transforms can only approximate republication; those twelve are the
closest thing to ground truth available here.

---

## Known limitations

- **LFW is not demographically representative.** It is predominantly male and
  predominantly light-skinned, and a single global threshold measured on it
  should not be assumed to deliver the same FAR for everyone. See ETHICS.md §6.
- **Republication transforms are synthetic.** They model what an outlet's CMS
  plausibly does, and are validated against only twelve real cases.
- **The similarity threshold is dataset-specific.** LFW is largely frontal
  press photography. Inputs far from that distribution — heavy pose, occlusion,
  low light — are not represented.
- **`same_photo_phash_max` assumes a detectable face in both images.** Where the
  face hash is unavailable the pipeline records `null` and falls back to the
  similarity verdict alone.
- **These numbers apply only to `buffalo_l` (w600k_r50).** Switching to the
  `requirements-fallback.txt` deepface path changes the embedding space, and the
  calibration must be re-run before any verdict is trusted.
