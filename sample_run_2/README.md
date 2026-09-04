# Second sample run — a different subject, and the ranking inversion in the open

A second complete run, on a different person, committed for the same reason as
[`sample_run/`](../sample_run/README.md): so the central claim can be checked
**with no API keys, no models, and none of our code**.

```
run       20260904T163955Z_c56949
subject   Sundar Pichai (public figure — see ../ETHICS.md, demo-subject policy)
input     input/input_pichai_run2.jpg  339x478  sha256 36e29b27...  (a derived crop)
matched   businessinsider.com          700x525  sha256 853211fa...
cosine    0.8761  [0.8598 - 0.8767] over 6 augmented views of the input
          threshold 0.2149   face-pHash distance 22   same-photo cutoff 15
verdict   DISTINCT-PHOTOGRAPH SAME-SUBJECT CANDIDATE
strength  strong  (at or above the genuine-score 25th percentile, 0.6066)
tx        0xe3b94010676d6881429d40139fd667f11de4ba987d565da200d16c52602ca754
```

## Why this run is here as well as the first

The first sample run demonstrates the pipeline. This one demonstrates the
**ranking inversion**, which is the design decision hardest to believe without
seeing it happen.

Twelve candidates were examined. Ranked by raw similarity, the winner would have
been Wikimedia Commons at **0.9915** — the highest score in the run by a wide
margin. It was demoted to the bottom tier, and a Business Insider photograph
scoring **0.8761** was reported instead:

| source | cosine | face-pHash distance | verdict |
|---|---|---|---|
| Wikimedia Commons | **0.9915** | 2 | same photograph — *demoted* |
| NY Post | 0.7505 | 14 | same photograph — *demoted* |
| **Business Insider** | **0.8761** | 22 | **distinct photograph — reported** |
| CNBC | 0.8640 | 24 | distinct photograph |
| Wikipedia | 0.8205 | 18 | distinct photograph |
| … | | | |

Every row above is checkable: `run.log` in this directory is the run's own
scored output for all twelve candidates, including the two demoted ones.
(`bundle.json`'s `runner_up_scores` carries only the top five *after* the
winner, which by construction excludes the demoted tier — so the log is what
makes this table verifiable rather than asserted.)

The 0.9915 hit is the **source photograph itself**, republished. The input was
derived from a Wikimedia image, so the pipeline found the very file it came
from, scored it near-perfectly, and correctly refused to treat that as evidence
of anything except file provenance. A near-1.0 cosine against a re-encoded crop
of the same shot proves the crop was made — not that a face was recognised.

That is the whole argument for ranking by evidence strength rather than by
similarity, and here it is visible in one table.

## 1. Reproduce the evidence hash with the standard library alone

```bash
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" sample_run_2/canonical.json
```

Expected:

```
0c2529df728dc7705bf3bf36034749b0d1a5651e1b18953bea69d1a8c10d8745
```

That is the value written to Base Sepolia. Compare it against the transaction
input data on
[BaseScan](https://sepolia.basescan.org/tx/0xe3b94010676d6881429d40139fd667f11de4ba987d565da200d16c52602ca754).

## 2. Verify against the chain

```bash
pip install web3
python verify.py --bundle sample_run_2/bundle.json
```

Expect `VERIFIED` (exit 0), reporting first-seen `2026-09-04T16:40:40Z` and
source domain `businessinsider.com`.

## Reading the numbers

**`provider_exact_match_count` is 226.** Google reports two hundred and twenty-six
"exact" matches for bytes that exist nowhere on the web — the file was created
locally minutes before the search. Its notion of an exact match is perceptual
near-duplicate detection, not byte equality. The first run reported 400 for the
same reason. Neither number is cited anywhere as evidence: the byte-level claim
rests on SHA-256, computed here, and the "different photograph" claim rests on
the face-region pHash distance of 22 against a calibrated cutoff of 15.

**The interval is tighter than the first run's**, ±0.0085 against ±0.0117 across
six augmented views. We have no established cause for that and are not going to
invent one — the obvious guess, that a larger input face is disturbed less by
augmentation, is contradicted by the measurement: this input's face is *smaller*
(131px against the first run's 148px). Two runs cannot separate the effect of
face size from the effect of the particular photograph anyway.

**The score is lower and the find is better.** 0.8761 against the first run's
0.9623, and both are `strong`. What makes this the better demonstration is not
the score but the spread: the same input also produced a 0.9915 republication,
so both verdict tiers appear within a single run rather than having to be
compared across runs.

## Same caveat as the first run

Google Lens returns a different candidate set from one day to the next, so a
live run will not reproduce this exactly. What has held across every run is the
structure: republications cluster high with face-pHash distance ≤ 14, distinct
photographs sit at ≥ 18, and an impostor never clears the threshold.

Candidate images are omitted to keep the repository small and to avoid
redistributing third-party press photographs; a live run writes them to
`runs/<run_id>/candidates/`.
