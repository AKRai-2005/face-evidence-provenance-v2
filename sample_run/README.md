# Sample run — verify this without running the pipeline

A complete, real run, committed so a reviewer can check the central claim **with
no API keys, no models, and none of our code**.

```
run      20260903T014815Z_2fb013
input    input/input.jpg  460x306  sha256 1f6c56c4...   (a derived crop)
matched  forbes.com       320x480  sha256 39acef1d...
cosine   0.8346  [0.8099 - 0.8421] over 6 augmented views of the input
         threshold 0.2149    face-pHash distance 34    same-photo cutoff 15
verdict  DISTINCT-PHOTOGRAPH SAME-SUBJECT CANDIDATE
tx       0xc866adadf35a8b8150c99ef56c355ffa2b8744dcb10b2ccc0dd7b6d87db3c460
```

## 1. Reproduce the evidence hash with the standard library alone

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" sample_run/canonical.json
```

Expected:

```
c5ae5ff891162c418fcef8f227e74faeb7cd229559d1ffd634f88a63e153b712
```

That is the value written to Base Sepolia. Compare it against the transaction
input data on
[BaseScan](https://sepolia.basescan.org/tx/0xc866adadf35a8b8150c99ef56c355ffa2b8744dcb10b2ccc0dd7b6d87db3c460).

`canonical.json` holds the exact bytes that were hashed — a single line, no
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

## What the numbers mean

**The similarity is an interval.** `face_similarity_bp` 8346 is the *median*
over six augmented views of the input; `face_similarity_lo_bp` / `_hi_bp` give
the range. The input is a crop we chose, so the honest question is how much the
score depends on that choice — here, ±0.02. Stored as integer basis points
because floats do not hash reproducibly across platforms.

**`provider_exact_match_count` is 400, not 0.** Google reports 400 "exact"
matches for a crop that exists nowhere byte-identical, because its notion of an
exact match is perceptual near-duplicate detection rather than byte equality.
This is precisely why the project never cites that number as evidence of
distinctness. The byte-level claim rests on SHA-256, which we compute ourselves,
and the "different photograph" claim rests on the face-region pHash distance
of 34 against a calibrated cutoff of 15.

## What the raw responses show

`raw/serpapi_lens.json` is the untouched provider response, including
`search_metadata.id`, which resolves in SerpApi's own dashboard — so the search
can be confirmed to have happened from their records, not just from ours.
`raw/serpapi_exact.json` is the separate `type=exact_matches` query. Candidate
images are omitted here to keep the repository small; a live run writes them to
`runs/<run_id>/candidates/`.

## A live run will not reproduce this exactly

Google Lens returns a different candidate set from one day to the next. An
earlier run of this identical input matched siliconangle.com at 0.8429 against a
completely different field. That is a property of the web, not a defect, and it
is why this run is frozen here. What has held across every run is the structure:
republications cluster near 0.97 with pHash distance ≤ 12, distinct photographs
sit lower with distance ≥ 26, and an impostor never clears the threshold.
