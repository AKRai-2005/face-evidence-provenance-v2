# Face-Derived Evidence Provenance

Take a face image. Genuinely discover a matching public web post. Prove the match
with face embeddings rather than assertion. Freeze a cryptographic fingerprint of
that discovery on a public blockchain. Let a stranger re-verify it **without
running any of this code**.

> **The chain notarises the claim. It does not validate it.**
> `EvidenceRegistry` proves an evidence object existed at a given time and has
> not changed since. It has no opinion on whether the match was correct.

---

## The Distinct-Artifact Proof

Most reverse-image pipelines prove nothing about faces: they find *the same JPEG*
somewhere else, which would work identically on a photo of a chair. This one is
built so that cannot be what happened.

**The image fed in is never the image discovered.** We prove they are different
files (different SHA-256), and we prove they contain the same person (cosine
similarity on ArcFace embeddings against a calibrated threshold).

```
DISTINCT-ARTIFACT PROOF
  input image   sha256: 1f6c56c41df11f00006220fb8776d6cf70a4a15d92487ff687dbd12d6864783a
                        460x306, local
  matched image sha256: 39acef1d441f80b4bec8ee77c507075c9b05dbce5f9ce0545b81ed5f687c1bb1
                        320x480, from forbes.com

  identical files: NO          <- reverse-image-hash matching is ruled out
  face cosine similarity: 0.8346   [0.8099 - 0.8421] over 6 augmented views
  calibrated threshold:   0.2149  (TAR 0.976 @ FAR 7.83e-04, see calibration/)
  face-region pHash distance: 34   (same-photo cutoff 15)

  verdict: DISTINCT-PHOTOGRAPH SAME-SUBJECT CANDIDATE (strong evidence)
```

This is the run frozen in [`sample_run/`](sample_run/README.md), reproducible by
anyone with `web3` alone. Evidence hash
`c5ae5ff891162c418fcef8f227e74faeb7cd229559d1ffd634f88a63e153b712`, notarised at
[`0xc866adad…`](https://sepolia.basescan.org/tx/0xc866adadf35a8b8150c99ef56c355ffa2b8744dcb10b2ccc0dd7b6d87db3c460).

**The similarity is an interval, not a point.** The input is a crop *we chose*,
so the honest question is how much the score depends on that choice. The pipeline
embeds the input under six views (identity, horizontal flip, two crops, two
scales) and reports the **median** with the full range. This was not cosmetic:
the un-augmented score sits at the *top* of its own range, so reporting a single
number was quietly flattering. When an interval straddles the threshold the
verdict is `UNCERTAIN` rather than a coin-flip presented as settled.

The ranked table from that run — note what is *not* selected:

```
 #  source                cosine   range (TTA)   pHashD  verdict
11  Forbes               +0.8346  0.810-0.842      34    DISTINCT PHOTO  <- selected
 8  Technology Magazine  +0.8345  0.813-0.841      28    DISTINCT PHOTO
 9  AI Magazine          +0.7634  0.751-0.770      36    DISTINCT PHOTO
 7  Fortune              +0.7605  0.750-0.772      28    DISTINCT PHOTO
 5  Wikipedia            +0.7599  0.743-0.764      34    DISTINCT PHOTO
 6  LinkedIn             +0.7412  0.732-0.752      30    DISTINCT PHOTO
 1  Axios                +0.9761  0.958-0.986      12    same photo
 2  eBay                 +0.9684  0.952-0.975      12    same photo
```

The two **highest-scoring** candidates are the same press photograph republished,
and are correctly demoted. A pipeline that ranked by cosine would have reported
Axios at 0.9761 and called a republished file "strong evidence".

> **Live results vary, and that is not a defect.** Google Lens returns a
> different candidate set from one day to the next — an earlier run of this exact
> input matched siliconangle.com at 0.8429 with a completely different field. The
> committed `sample_run/` is frozen so the documented numbers stay checkable; a
> live run will find its own match. What has held across every run is the
> *structure*: republications cluster near 0.97 with pHash distance ≤ 12, and
> distinct photographs sit lower with distance ≥ 26.

### Does it reject the wrong person?

A pipeline that only ever demonstrates matches has said nothing about its false
accept rate. `scripts/demo_negative_control.py` scores an **impostor** — a
different public figure — against the same real candidate set, with the same
threshold:

```
highest impostor score : +0.0629
calibrated threshold   : +0.2149      (3.4x margin)
genuine score range    : +0.5448 to +0.9837
false accepts          : 0 / 11
```

Every genuine score beats every impostor score by at least **0.4819**, and the
threshold sits inside that gap rather than near either distribution. This held on
two independent candidate sets captured on different days. It needs no network,
no API quota and no gas, so it can be demonstrated even with every external
service down.

### Who does it fail for?

An aggregate error rate can hide its own shape, so it is broken down per person
(`scripts/validate_per_identity.py`, on LFW where the directory *is* the label):

```
overall FRR                        0.0244
identities with ZERO false rejects 422/450  (93.8%)
per-identity FRR  median 0.0000   p90 0.0000   max 0.6000
144 false rejects fall on just 28 of 450 identities
```

**The error concentrates.** For 93.8% of identities the false-reject rate is
zero; for a few it exceeds 0.5. "2.4% FRR" is true and also misleading, and the
tail is what would matter to whoever is in it. See [ETHICS.md](ETHICS.md) —
including why an attempt to turn this into a *demographic* claim was abandoned
rather than published on untrustworthy labels.

### Three outcomes, not two

Measured on real Google Lens results during the risk spike, the *highest*-scoring
candidates (cosine 0.968–0.985) were all **the same press photograph republished**
by different outlets at different sizes. The genuinely different photographs of
the same person scored **lower** (0.637–0.841) and are **far stronger evidence**.

A pipeline that ranks by cosine therefore returns the weakest evidence with the
biggest number. So this one reports three outcomes and ranks by evidence
strength, not similarity:

| verdict | condition | strength |
|---|---|---|
| `EXACT-DUPLICATE` | identical SHA-256 | weak — proves file identity, not face identification |
| `SAME-PHOTOGRAPH REPUBLICATION` | different file, face-pHash distance ≤ cutoff | moderate |
| `DISTINCT-PHOTOGRAPH SAME-SUBJECT` | different file, face-pHash distance > cutoff | **strongest** — cannot be produced by file or near-duplicate matching |

Distinguishing the middle row from the bottom one is the hardest and most
load-bearing claim in the project, and it required discarding the obvious method
— see [Why whole-image pHash fails](#why-whole-image-phash-fails).

---

## Quickstart

```bash
git clone <repo> && cd task3
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
copy .env.example .env          # then fill in SERPAPI_KEY (see below)
.venv\Scripts\python.exe -m app.main --image data/input.jpg
```

On Windows, invoke `.venv\Scripts\python.exe` explicitly. A bare `python` can
resolve to the Microsoft Store alias even with the venv "activated", and the
project's dependencies are not installed there. The pipeline detects this and
exits with an explanation rather than a confusing `ImportError`.

**Minimum configuration** — only `SERPAPI_KEY` is needed to search
([free tier: 250/month](https://serpapi.com/manage-api-key)). To also write to
the chain you need `DEPLOYER_PRIVATE_KEY` (generate with
`python scripts/new_wallet.py`, fund at the
[CDP faucet](https://portal.cdp.coinbase.com/products/faucet)) and
`SUBJECT_COMMITMENT_SALT`. Run with `--no-chain` to skip the chain entirely.

First run prompts once for an ethics acknowledgement, recorded in `.consent`.

### Making your own derived input

```bash
.venv\Scripts\python.exe scripts\make_derived_input.py --source <photo> --out data/input.jpg
```

Crops, rescales and re-encodes, then prints both SHA-256s so you can confirm the
input is genuinely a new artifact. **Never feed in a pristine file that exists
byte-identical on the web** — that would make exact-file matching a viable
explanation for any hit, which is precisely what this project exists to rule out.

---

## How an input image causes a real network search

This is the question a sceptical reader should press hardest, so here is the
whole path with nothing elided:

1. `data/input.jpg` is read and SHA-256'd locally.
2. RetinaFace detects faces; the highest-confidence face passing the quality gate
   is embedded to a 512-d L2-normalised ArcFace vector.
3. The **image bytes** are POSTed as `multipart/form-data` to
   `https://serpapi.com/image`, which returns an `image_id`.
4. `GET https://serpapi.com/search?engine=google_lens&image_id=…` returns
   `visual_matches[]` — title, page link, image URL, source.
5. A second call with `type=exact_matches` asks a different question (see below).
6. Each candidate image is downloaded **from its own third-party host**, not from
   the search provider.
7. Every face in every candidate is embedded and scored against the input.

Nothing is cached, canned, or bundled. The **untouched provider JSON** is written
to `runs/<run_id>/raw/` on every run, and the downloaded candidate images to
`runs/<run_id>/candidates/`. Each response carries SerpApi's own
`search_metadata.id`, which resolves in their dashboard — so the search can be
confirmed to have happened from their records, not just ours.

The input never needs a public host on this path: SerpApi accepts the upload
directly. That is better than the design this project started with, and it is a
genuine privacy improvement — see [ETHICS.md §4](ETHICS.md).

---

## Architecture

```
input face image (a derived artifact)
    |
[FACE]     RetinaFace detect -> quality gate -> ArcFace 512-d embedding (L2-normed)
    |
[HOST]     (Bright Data path only) temp-upload to an expiring public host
    |
[SEARCH]   SerpApi google_lens -> visual_matches[]      (fallback: Bright Data)
    |
[FETCH]    download each candidate: 12 max, 8s timeout, 10MB cap, type allowlist
    |
[VERIFY]   detect ALL faces per candidate -> embed -> cosine vs input -> keep max
    |
           rank by verdict tier, then cosine; record runner-up spread
    |
[EVIDENCE] canonical JSON -> SHA-256 -> evidence_hash
    |
[CHAIN]    EvidenceRegistry.recordEvidence(hash, commitment, domain) on Base Sepolia
    |
[VERIFY]   verify.py: recompute -> read chain -> VERIFIED / TAMPERED / NOT REGISTERED
```

Every run writes a complete audit trail to `runs/<utc_timestamp>_<short_id>/`:
input image, raw provider responses, downloaded candidates, `evidence.json`,
`canonical.json`, `bundle.json`, `run.json` and `run.log`.

---

## Model and metric

```
detector    RetinaFace  (buffalo_l / det_10g)
embedder    ArcFace     (buffalo_l / w600k_r50)   512-d, L2-normalised
metric      cosine similarity on L2-normalised embeddings
recorded    insightface_buffalo_l_arcface_w600k_r50
runtime     onnxruntime 1.29.0, CPU, ~0.35 s/image
```

Not Euclidean distance, and not a "confidence percentage" — every number the
pipeline prints traces to a line of code that computed it.

**One correction worth flagging.** `buffalo_l`'s recognition network is
**`w600k_r50`** — ArcFace on a ResNet-50 — not the `r100` that the project's
original schema specified. Since that string is canonicalised, hashed and
notarised, shipping it unchecked would have written a false model claim into an
immutable public record. `FaceEngine.model_id()` therefore reads the identifier
from the **loaded ONNX graph at runtime** rather than trusting a constant.

### Quality gate

Faces are rejected before embedding if they are too small, too low-confidence,
or **too blurred** — the last of which detection score does not catch. Measured
on the committed fixtures, out-of-focus faces score 0.74–0.80 detection
confidence at 86–87 px, clearing any size-and-confidence gate, while embedding to
noise (~0.01 cosine against every identity). A scale-normalised Laplacian
variance separates them by three orders of magnitude:

```
blurred, unusable   5.3   6.4
usable            995 - 4219      threshold 100
```

`data/fixtures/A1_pichai_hanoi2015.jpg` is kept specifically to test this, and is
correctly rejected with exit code 11.

---

## Evidence canonicalization

The evidence hash must be reproducible by a stranger on a different OS, Python
build and CPU. Floats and timestamps are where naive implementations silently
break, so the rules are strict:

- UTF-8 throughout; **Unicode NFC** normalisation on every string
- keys sorted lexicographically, recursively
- `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
- **no floats anywhere** — similarity is stored in integer basis points
  (`0.7413 → 7413`), because float repr differs across platforms and would
  destroy hash reproducibility
- timestamps RFC 3339 UTC, second precision, trailing `Z`
- URLs: lowercase scheme and host, strip fragments, strip tracking params
  (`utm_*`, `fbclid`, `igshid`, …), preserve path case
- then SHA-256 over the UTF-8 bytes

The canonicalizer **rejects** a float rather than coercing it, and names the JSON
path where it found one.

### Reproduce the hash yourself

`canonical.json` holds the exact bytes that were hashed, so this needs nothing
but the standard library — no venv, no dependencies, none of our code:

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" runs/<run_id>/canonical.json
```

Compare that against the transaction input data on BaseScan.

---

## Verify the sample run — no keys, no models, no pipeline

A complete real run is committed at [`sample_run/`](sample_run/README.md) so the
central claim can be checked with nothing installed but `web3`:

```bash
# 1. reproduce the notarised hash with the standard library alone
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" sample_run/canonical.json
#    -> 9fe3f99176aeb9c2523d1b6363d6ea20788e7a87188dfd616bf821de11d43da6

# 2. check it against the chain
pip install web3
python verify.py --bundle sample_run/bundle.json
#    -> VERIFIED
```

It also ships the untouched provider responses, including the separate
`type=exact_matches` query, so the search can be confirmed to have happened
rather than taken on trust.

---

## Independent verification

`verify.py` depends on **`web3` and the standard library only**. No InsightFace,
no torch, no SerpApi, none of the pipeline:

```bash
python -m venv fresh && fresh\Scripts\pip install web3
fresh\Scripts\python verify.py --bundle runs/<run_id>/bundle.json
```

It deliberately **re-implements** canonicalization rather than importing ours. A
verifier that calls the code it is checking would share any bug in it and prove
correspondingly less. A test asserts the two agree byte-for-byte so they cannot
silently diverge.

Three unambiguous outcomes, with distinct exit codes:

| outcome | meaning | exit |
|---|---|---|
| `VERIFIED` | hash present on chain, bundle intact | 0 |
| `TAMPER DETECTED` | bundle modified after recording | 1 |
| `NOT REGISTERED` | hash absent from the registry | 2 |

---

## Smart contract

**Deployed and source-verified on Base Sepolia:**
[`0x38157D4652304BA251edCcffa435A0bC3F55305a`](https://sepolia.basescan.org/address/0x38157D4652304BA251edCcffa435A0bC3F55305a#code)

Three design decisions worth defending:

**Nothing biometric goes on chain.** Only `subjectCommitment = SHA-256(salt ||
int8-quantised embedding)`, with the salt held locally in `.env`. A public ledger
is immutable and world-readable, which makes it the worst possible place for a
face template — one published there could never be withdrawn, by us or by the
subject. The commitment still lets us later prove a record concerns a given
subject by revealing the salt. The embedding is quantised to int8 before hashing
because float embeddings do not hash reproducibly across platforms.

**Only the domain, not the full URL.** The full URL lives in the bundle and is
covered by `evidenceHash`, so it is provably fixed at record time without being
permanently broadcast.

**`AlreadyRecorded` reverts** on a duplicate, which gives first-seen timestamping
rather than last-writer-wins. Recording the same evidence twice is refused, and
the refusal cites the original timestamp.

```bash
.venv\Scripts\python.exe scripts\compile.py          # pinned solc 0.8.24, optimizer 200
.venv\Scripts\python.exe scripts\deploy.py           # deploys once, pins address into .env
.venv\Scripts\python.exe scripts\verify_contract.py  # publishes source on BaseScan
```

---

## Threshold calibration

**Threshold 0.2149 selected at FAR 7.83e-04 on 65,907 pairs from 450 LFW
identities; TAR 0.9756, AUC 0.98683.** The same-photograph cutoff is a
face-region pHash distance of 15, which captures 96.7% of republications and
98.3% of distinct photographs. Method and data in
[`calibration/`](calibration/METHOD.md).

```
450 identities, 2534 images (capped at 6/identity)
pairs: 5907 positive, 60000 negative
threshold 0.2149 at FAR 7.83e-04 -> TAR 0.9756   (AUC 0.98683)

same photograph       n=1061  median  2  p95 12  p99 24  max 32
different photograph  n=1345  median 28  p05 18  p01 14  min  4
-> same_photo_phash_max = 15
real-world check against the 12 Phase 0 Lens candidates: 6/6 and 6/6 correct
```

The pipeline **loads these from `calibration/results.json` and refuses to run if
the file is missing.** There is no hardcoded threshold anywhere.

Two corrections worth stating, because both changed the numbers materially:

*FAR 1e-3 cannot be observed with 500 negative pairs.* LFW's standard test split
has exactly that, so quoting 1e-3 off it would be extrapolation. Pairs are built
combinatorially instead, affording 60,000 negatives.

*LFW is severely imbalanced.* A first pass taking every within-identity
combination produced 236,225 positives, of which ~140,000 (59%) came from George
W. Bush alone — the reported TAR was effectively "TAR on one man". Images and
pairs are now capped per identity.

On the AUC: 0.9868 is below the ~99.8% ArcFace is usually quoted at on LFW,
because that figure is accuracy on LFW's curated 6,000-pair protocol while this
uses harder combinatorial pairs. It is not label noise — the highest-confidence
face is the centred subject face in 98.7% of a sampled 149 images.

Full method, including the corrections below, in [`calibration/METHOD.md`](calibration/METHOD.md).

### Does the benchmark threshold transfer to the open web?

Calibrating on LFW and then quoting "FAR 7.83e-04" while pointing the tool at
whatever Google Lens returns is a claim about *LFW*, not about what this pipeline
does. LFW is funneled, frontal, 250×250 press photography; real candidates are
arbitrary resolution, pose, crop and compression. That gap between calibration
domain and deployment domain is where face recognition quietly fails in public.

So `scripts/validate_in_domain.py` measures the operating point on the deployment
distribution itself, using only images the pipeline actually downloaded:

```
subjects        : 2 (both public figures, from real Lens searches)
genuine pairs   :  297   min +0.5316  median +0.7684  max +0.9988
impostor pairs  :  264   min -0.0681  median +0.0209  max +0.1185

LFW-calibrated threshold : +0.2149
in-domain false accepts  : 0 / 264      (FAR 0.00e+00)
in-domain false rejects  : 0 / 297      (FRR 0.00e+00)
separation               : +0.4131
```

Every genuine pair outscores every impostor pair. **Any** threshold in
(0.1185, 0.5316] would separate them perfectly on this data, and the
LFW-calibrated 0.2149 sits inside that window — so the benchmark number
transfers, with room to spare, on this distribution.

**The caveat matters more than the result.** Two subjects is a *narrow*
validation set, and both are male technology executives of South Asian descent.
This says the threshold transfers from curated benchmark photography to
uncurated web photography. It says **nothing** about demographic generalisation,
which remains the project's most consequential open limitation — see
[ETHICS.md §6](ETHICS.md). Zero false accepts out of 264 pairs also cannot
resolve a rate of 1e-3; it bounds it loosely, no more.

---

### Why whole-image pHash fails

The plan for distinguishing "different photograph" from "same photograph
republished" was perceptual-hash distance between input and match. Measured
against real Lens results, **whole-image pHash cannot make that distinction at
all**: because the query is a *crop*, the global hash is randomised, and every
candidate — republications and genuinely different photographs alike — landed at
110–138 out of 256.

Hashing the **aligned face region** instead (box expanded 30%, resampled to a
fixed 128×128 square, 64-bit pHash) normalises position and scale, so the hash
becomes comparable across different crops of the same photograph:

| | cosine | whole-image pHash | **face-region pHash** |
|---|---|---|---|
| same photograph, republished | 0.968–0.985 | 18–32 | **4–14** |
| different photograph, same subject | 0.637–0.841 | 20–32 | **26–40** |

Whole-image distance overlaps completely. Face-region distance separates cleanly.

---

## Provider landscape

Checked 2026-09-01, each confirmed with a real call.

| option | status | decision |
|---|---|---|
| **SerpApi `google_lens`** | live; free tier 250/month; accepts direct upload | **primary** |
| Bright Data SERP | live; 5,000/month free; reverse-image paths unreliable | fallback, see below |
| Bing Visual Search | **retired** Aug 2025, no new keys | not used |
| TinEye | live, but **exact/near-duplicate only** | **rejected** — cannot find a *different* photo of the same person, which is the entire point here |
| PimEyes | no API; ToS prohibits automation | **not touched** |
| Yandex | best raw recall, aggressive anti-automation | not used |

### Why only one provider actually works

The single-provider dependency is a researched conclusion, not an unexplored gap.
**There is no free, web-scale, reverse-image API besides the one we use.** The
category mostly does not exist; what looks like an alternative is usually solving
a different problem:

| candidate | why it does not substitute |
|---|---|
| Qdrant · Marqo · CLIP+pgvector | Self-hosted vector search **over your own images**. Web-scale search would require crawling the web first. Different problem. |
| Google Cloud Vision *Product Search* | Searches **your** product catalogue, not the public web. |
| TinEye API | From $200/month, **no free plan** — and near-duplicate only, so it cannot find a *different photograph* of the same person, which is this project's entire point. |
| Bing Visual Search | Retired Aug 2025; no new keys issuable. |
| Yandex | No commercial API, and face search there is a documented legal grey area. |

**One genuine candidate exists: Google Cloud Vision `WEB_DETECTION`.** It is truly
web-scale, returns `visuallySimilarImages` and `pagesWithMatchingImages`, and its
free tier is **1,000 units/month — four times SerpApi's 250**. It is not wired up
here because it requires a Google Cloud billing account (a card, even though
nothing is charged within quota), and shipping an *untested* provider would be a
worse failure than documenting a real one — see the fabricated-provider
anti-pattern this project is built against. It is the obvious next integration.

### Provider reliability (measured, not assumed)

Bright Data's SERP zone serves plain Google reliably. Its **reverse-image** paths
do not:

| request | result |
|---|---|
| plain Google, `format=raw` | 821 KB every attempt |
| plain Google, `brd_json=1` | 125 KB parsed JSON |
| Lens `uploadbyurl` + `brd_lens` + `brd_json` | **1 of 6** attempts non-empty |
| Lens `uploadbyurl` + `brd_json` only | 0 bytes every attempt |
| `www.google.com/searchbyimage` | 244 KB twice, then 0 bytes |

The single non-empty Lens response was 1.3 KB of tab metadata containing **no
visual matches**, against SerpApi's 86 KB and 59 matches for the same image.

So Bright Data is implemented as a **genuine but unreliable** fallback: it makes
real calls, retries with backoff, and raises `ProviderUnavailable` when it cannot
deliver. It never substitutes cached or fabricated results. In practice SerpApi
carries the pipeline, and this README says so rather than implying a redundancy
that does not exist.

### On `exact_matches`

The pipeline issues a **separate** `type=exact_matches` query. This matters: the
default `type=all` response contains no `exact_matches` key at all, so reading it
there and finding nothing would prove nothing — it would only mean we never
asked.

Worth stating plainly: Google's "exact match" is **perceptual near-duplicate
detection, not byte equality**. A derived crop of a widely-published photo still
returns hundreds of exact matches, because Google recognises its ancestor. We
therefore never claim "zero exact matches" as evidence. The byte-level claim
rests on SHA-256, which we compute ourselves.

---

## Failure handling

No traceback ever reaches the user. Every failure prints a labelled, actionable
message and exits with a distinct code:

| code | condition |
|---|---|
| 10 | no face detected |
| 11 | face fails the quality gate (too small, or too blurred) |
| 12 | input missing or not decodable |
| 13 | configuration error (names the missing key and where to get it) |
| 20 | all search providers failed |
| 21 | zero candidates, or none downloadable, or no faces in any |
| 22 | **no candidate above threshold** — a valid outcome, reported honestly |
| 40 | calibration missing |

Exit 22 deserves emphasis: *no match* is a legitimate result. The threshold is
never lowered to force one.

Also handled: multiple faces (highest-confidence chosen, all logged), provider
auth failure, rate limits and quota exhaustion, candidate 403/404, non-image
content types, download timeouts, oversized downloads (including a lying
`Content-Length`), RPC unreachable, insufficient gas (prints the faucet URL),
duplicate records, and bundle schema mismatch in the verifier.

---

## Replay mode

```bash
.venv\Scripts\python.exe -m app.main --replay <run_id>
```

Re-runs against **previously captured real responses** in `runs/`. It never
auto-activates, the flag is always explicit, and the banner is always visible. It
exists so a network failure during a recording costs a retake rather than the
submission. A run directory without a `run.json` manifest is refused rather than
faked.

---

## Tests

```bash
.venv\Scripts\python.exe -m pytest -q
```

Chain tests run against an **in-process EVM**, so they need no network, no funds
and no deployed contract.

Worth noting where that has limits: every chain test passed while a real
read-after-write bug sat in the code. `getEvidence()` immediately after a receipt
returned `exists=False` for a record that had just been written, because public
RPC endpoints are load-balanced and the call hit a node that had not yet applied
the block. An in-process EVM has instant consistency and **structurally cannot
reproduce this**. It was found by running against the real chain, and is now
handled by `wait_for_record()` with a stubbed regression test.

---

## Licensing

**InsightFace pretrained models — including `buffalo_l`, used here — are released
by their authors for non-commercial research purposes only.** This submission is
a non-commercial research artefact and falls within that; any commercial use
would require different models or a licence from the authors.

Fixture images are Wikimedia Commons photographs of public figures under their
respective licences — see [`data/fixtures/SOURCES.md`](data/fixtures/SOURCES.md).

Project code: MIT. Contract: MIT (`SPDX-License-Identifier` in source).

---

## Limitations

Read [ETHICS.md](ETHICS.md) — it is not boilerplate. The load-bearing points:

- Face-recognition error rates are **not uniform across demographic groups**, and
  the threshold here is a single global one calibrated on LFW, which is
  predominantly male and light-skinned.
- Search recall skews heavily toward the already-indexed. A null result is weak
  evidence of absence.
- Similarity above threshold is **evidence, not proof of identity**.
- The chain proves *when* a claim was made and that it has not changed. It does
  not prove the claim was *true*.
