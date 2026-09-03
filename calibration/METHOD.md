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

**LFW**, funneled, full 250×250, identities with ≥3 images. The JPEGs are read
**directly from disk** (`~/scikit_learn_data/lfw_home/lfw_funneled/`), not through
`sklearn.datasets.fetch_lfw_people`.

That is a deliberate change, not a stylistic one. `fetch_lfw_people` materialises
every image into a single float32 array — roughly 3 GB at full-size colour —
which pushed the development machine into swap and slowed embedding from 0.38 s
to ~2.4 s per image, so a run that should take 16 minutes had not finished after
106 minutes of CPU. Walking the directory keeps memory flat (≈680 MB) and
restores the measured rate. `sklearn` is still used, for `roc_curve`.

Fetch the data once with:

```bash
python -c "from sklearn.datasets import fetch_lfw_people; fetch_lfw_people(min_faces_per_person=3)"
```

Each image is embedded **once** with the exact pipeline used at run time
(`FaceEngine.detect` → highest-confidence face → `normed_embedding`), and cached
to `_embeddings_cache.npz` so re-calibration does not repeat the embedding pass.

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
TAR was effectively "TAR on George W. Bush" rather than a property of the model.

The calibration therefore caps **images per identity** (`--max-images-per-identity`,
default 6) and **positive pairs per identity** (`--max-pairs-per-identity`,
default 15), so no single subject dominates the operating point.

### Measured results

```
450 identities, 2534 images (capped at 6/identity), 9 rejected
pairs: 5907 positive, 60000 negative
threshold 0.2149 at FAR 7.83e-04  ->  TAR 0.9756   (AUC 0.98683)
```

**On the AUC.** ArcFace is often quoted at ~99.8% on LFW, so 0.9868 warrants an
explanation rather than a shrug. Two things account for it, and neither is a bug:

1. That headline figure is *accuracy on LFW's curated 6,000-pair protocol*, at
   the best-performing threshold. This calibration instead builds pairs
   combinatorially across 450 identities, which includes far harder positives —
   the same person a decade apart, in different lighting and pose — that the
   standard protocol does not weight so heavily. A harder pair set yields a
   lower AUC for the same model.
2. It is *not* label noise from face selection. LFW images frequently contain
   background people — 17.4% of a 149-image sample had more than one detected
   face — but because funneled LFW centres the subject, the highest-confidence
   face is the centre face **98.7%** of the time. Measured mislabelling from
   picking top-confidence is ~1.3%.

The operating point is what matters here, and it is comfortable: the chosen
threshold of 0.2149 sits well above the highest cross-identity score observed on
real data during the risk spike (0.1284) and far below genuine same-subject
matches (0.637-0.985).

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

### Measured distributions

```
same photograph       n= 1061   median  2   p95 12   p99 24   max 32
different photograph  n= 1345   median 28   p05 18   p01 14   min  4
-> same_photo_phash_max = 15
   captures 96.7% of republications and 98.3% of distinct photographs
```

The distributions overlap in the tails — a heavily reprocessed republication can
reach 32, and an unusually similar pair of distinct photographs (same shoot, same
pose) can fall to 4. A single scalar cannot separate those perfectly, and this
one does not claim to. What it does is put the cut where total error is lowest
while erring toward under-claiming.

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

## Two things deliberately NOT built

### A quality gate on candidate faces

The input face is gated on size, detection score and sharpness. Candidate faces
are not. That asymmetry was questioned and then measured across 34 matched
candidate faces from three real runs:

```
face_px    min  64   p10  102   median  252   max  683
sharpness  min 293   p10  492   median 1137   max 3089
det_score  min 0.691             median 0.817

candidates that would fail the input gate : 0 of 34
quality vs score correlation : px r=-0.018,  sharpness r=-0.047
```

Real search results are press and editorial photography, which is already
well-lit and adequately resolved; nothing came close to the gate, and face
quality does not predict the score in this range at all. A gate here would be a
control with no measured failure to prevent, so it is not implemented.

The asymmetry is also principled rather than lazy. The input is the one image a
*user* supplies and can get wrong -- a blurred phone snap, a face 20px across --
and a bad input silently poisons every comparison. A bad candidate merely scores
low and drops out on its own.

### Symmetric test-time augmentation

Augmentation is applied to the input only, not to candidates. Augmenting both
sides would roughly double the embedding cost for a symmetric estimate of a
quantity whose spread is already only 0.02-0.03. The question TTA exists to
answer is "how much does this depend on the crop *we* chose?", and only the
input side is ours to have chosen.

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
