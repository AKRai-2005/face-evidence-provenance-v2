# Sample run — verify this without running the pipeline

A complete, real run captured on 2026-09-02. It is committed so a reviewer can
check our central claim **with no API keys, no models, and none of our code**.

```
run      20260902T133815Z_f8d116
input    input/input.jpg  460x306  sha256 1f6c56c4...  (a derived crop)
matched  siliconangle.com 1024x780 sha256 bd093dde...
cosine   0.8503   threshold 0.2149   face-pHash distance 28   cutoff 15
verdict  DISTINCT-PHOTOGRAPH SAME-SUBJECT CANDIDATE
tx       0xac901c81786683551c6cee2d91fa46073bd3239214a5e17da7f505a8ae57bd71
```

## 1. Reproduce the evidence hash with the standard library alone

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" sample_run/canonical.json
```

Expected:

```
9fe3f99176aeb9c2523d1b6363d6ea20788e7a87188dfd616bf821de11d43da6
```

That is the value written to Base Sepolia. Compare it against the transaction
input data on
[BaseScan](https://sepolia.basescan.org/tx/0xac901c81786683551c6cee2d91fa46073bd3239214a5e17da7f505a8ae57bd71).

## 2. Verify against the chain

```bash
pip install web3
python verify.py --bundle sample_run/bundle.json
```

Expect `VERIFIED` (exit 0). `verify.py` needs only `web3` and the standard
library — no InsightFace, no torch, no SerpApi.

## 3. See it fail, so you know the check is real

Change one character of `evidence.source_url` in `bundle.json` and run it again:

```
TAMPER DETECTED -- evidence modified
```

## What the raw responses show

`raw/serpapi_lens.json` is the untouched provider response, including
`search_metadata.id`, which resolves in SerpApi's own dashboard — so the search
can be confirmed to have happened from their records, not just from ours. The
candidate images themselves are omitted here to keep the repository small; a
live run writes them to `runs/<run_id>/candidates/`.

`raw/serpapi_exact.json` is a separate `type=exact_matches` query. It returns
**10** results for an input that exists nowhere byte-identical, which is exactly
why this project never cites a "zero exact matches" figure as evidence — Google's
"exact match" is perceptual near-duplicate detection, not byte equality. The
byte-level claim rests on SHA-256, which we compute ourselves.
