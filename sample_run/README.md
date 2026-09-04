# Sample run — verify this without running the pipeline

A complete, real run, committed so a reviewer can check the central claim **with
no API keys, no models, and none of our code**.

```
run       20260904T113830Z_d33f18
input     input/input.jpg   460x306   sha256 1f6c56c4...   (a derived crop)
matched   manofmany.com    1200x900   sha256 1fc3fc3c...
cosine    0.9623  [0.9445 - 0.9679] over 6 augmented views of the input
          threshold 0.2149   face-pHash distance 28   same-photo cutoff 15
verdict   DISTINCT-PHOTOGRAPH SAME-SUBJECT CANDIDATE
strength  strong  (at or above the genuine-score 25th percentile, 0.6066)
tx        0xa9611ac746a6137ba6cdb0cb932c13ca33c5bc58b0950af29a27b5ebc4142e0a
```

## 1. Reproduce the evidence hash with the standard library alone

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" sample_run/canonical.json
```

Expected:

```
df7a4de56c024069923024643e81efe11bdbd772bb9d0a805485a9fb1df42a87
```

That is the value written to Base Sepolia. Compare it against the transaction
input data on
[BaseScan](https://sepolia.basescan.org/tx/0xa9611ac746a6137ba6cdb0cb932c13ca33c5bc58b0950af29a27b5ebc4142e0a).

`canonical.json` holds the exact bytes that were hashed — a single line with no
trailing newline, so no end-of-line translation can alter it on any platform.

## 2. Verify against the chain

```bash
pip install web3
python verify.py --bundle sample_run/bundle.json
```

Expect `VERIFIED` (exit 0). `verify.py` needs only `web3` and the standard
library — no InsightFace, no torch, no SerpApi — and it **re-implements**
canonicalization rather than importing ours, so it does not share our bugs.

## 3. See it fail, so you know the check is real

Change one character of `evidence.source_url` in `bundle.json` and run it again:

```
TAMPER DETECTED -- evidence modified
```

(Editing in Notepad is fine — the verifier tolerates the UTF-8 BOM that Notepad
and PowerShell add.)

## Reading the numbers

**Two thresholds, answering two different questions.** `threshold_bp` 2149 asks
*"is this the same person at all?"* — chosen by ROC at FAR 6.8e-04.
`evidence_strength` asks *"is this score typical of a genuine match?"*, against
the 25th percentile of genuine scores (0.6066). A score can clear the first and
fail the second, and when it does the run says **WEAK — treat as a lead, not a
finding**. That distinction exists because a synthetic face once scored 0.4866
against a different synthetic face and was reported as strong evidence. This run
scores 0.9623 and is comfortably strong.

**The similarity is an interval.** `face_similarity_bp` 9623 is the *median* over
six augmented views of the input; `_lo_bp` / `_hi_bp` give the range. The input
is a crop we chose, so the honest question is how much the score depends on that
choice — here about ±0.02. Stored as integer basis points because floats do not
hash reproducibly across platforms.

**`provider_exact_match_count` is not zero.** Google reports "exact" matches for
a crop that exists nowhere byte-identical, because its notion of an exact match
is perceptual near-duplicate detection rather than byte equality. That is exactly
why this project never cites that number as evidence of distinctness. The
byte-level claim rests on SHA-256, which we compute ourselves; the "different
photograph" claim rests on the face-region pHash distance of 28 against a
calibrated cutoff of 15.

## What the raw responses show

`raw/serpapi_lens.json` is the untouched provider response, including
`search_metadata.id`, which resolves in SerpApi's own dashboard — so the search
can be confirmed to have happened from their records, not just from ours.
`raw/serpapi_exact.json` is the separate `type=exact_matches` query. Candidate
images are omitted here to keep the repository small; a live run writes them to
`runs/<run_id>/candidates/`.

## A live run will not reproduce this exactly

Google Lens returns a different candidate set from one day to the next. This
input has matched manofmany.com, forbes.com and siliconangle.com on different
days, each time against a different field. That is a property of the web, not a
defect, and it is why this run is frozen here. What has held across every run is
the structure: republications cluster near 0.97 with face-pHash distance ≤ 12,
distinct photographs sit at ≥ 24, and an impostor never clears the threshold.
