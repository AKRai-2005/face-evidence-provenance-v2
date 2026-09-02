# Shooting script — Task 3 demo video

**Target 110–125 s.** Every number spoken below is real and reproducible from
this repository. Do not round them up, and do not say anything the code did not
compute.

Record on **Sept 6**, not Sept 7.

---

## Language discipline — read before you speak

This is a scoring criterion in disguise, and it is the easiest thing to lose on
camera when you are improvising.

| say | never say |
|---|---|
| "high face similarity" | "identity confirmed" |
| "same-subject candidate" | "verified person" |
| "cosine 0.8503 against a calibrated threshold of 0.2149" | "85% match", "99.9% confident" |
| "the chain notarises when the claim was made" | "the blockchain proves it's him" |

If you fumble a number, **stop and retake**. A wrong number on camera is worse
than a pause.

---

## Pre-flight (do all of this before you hit record)

```powershell
cd task3

# 1. quota — each live run costs 2 SerpApi searches. Check you have room.
#    https://serpapi.com/dashboard

# 2. wallet still funded (deploy is already done; NEVER deploy on camera)
.venv\Scripts\python.exe -c "from web3 import Web3; from eth_account import Account; from app.config import get_settings; c=get_settings(); w=Web3(Web3.HTTPProvider(c.base_sepolia_rpc)); a=Account.from_key(c.deployer_private_key).address; print(a, w.from_wei(w.eth.get_balance(a),'ether'), 'ETH')"

# 3. consent already acknowledged, so no prompt interrupts the take
type .consent

# 4. tests green
.venv\Scripts\python.exe -m pytest -q

# 5. CAPTURE A FALLBACK RUN NOW, so a network failure costs a retake, not the shoot
.venv\Scripts\python.exe -m app.main --image data/input.jpg --yes
#    note the run_id it prints -- that is your --replay safety net
```

**Windows terminal setup:** maximise, font ≥ 16 pt, and set the window to about
100 columns. The `rich` tables and the proof panel wrap badly below that — check
by running once before recording. Dark background, light text.

**Have these open in tabs, pre-loaded, before recording:**

1. Terminal in `task3/`
2. Browser: [the contract on BaseScan](https://sepolia.basescan.org/address/0x38157D4652304BA251edCcffa435A0bC3F55305a#code) — on the **Contract** tab showing the green verified checkmark
3. A second terminal already `cd`'d into the **web3-only venv** for the verifier

**One thing to prepare mentally:** the live run takes ~40–45 s and produces a lot
of scrolling output. You are narrating *over* it, not waiting for it. Scenes 2–5
are a single command.

---

## Scene 1 — 0:00–0:14 · The premise

**Screen:** run this, output stays on screen.

```powershell
.venv\Scripts\python.exe scripts\make_derived_input.py --source data\fixtures\B2_nadella_smiling.jpg --out data\input.jpg
```

**Expect:**

```
source : 1280x853  198KB  sha256 16f416cc...
output : 460x306    19KB  sha256 1f6c56c4...
identical  : NO
```

**Say:**

> "The input is a re-encoded crop — 460 by 306, a different SHA-256 from any
> published file. These exact bytes exist nowhere on the web, so exact-file
> matching cannot explain anything we find. Any match has to come from the face."

*Why this is first: it is the claim everything else rests on. Establish it before
anything can be doubted.*

---

## Scene 2 — 0:14–1:00 · The live run

**Screen:** one command. Let it run. Narrate over the scrolling output.

```powershell
.venv\Scripts\python.exe -m app.main --image data\input.jpg
```

**0:14–0:24 — as the FACE lines appear**

> "RetinaFace detection, then a 512-dimension ArcFace embedding. Detection
> confidence 0.84, face 148 pixels, sharpness gate passed."

**0:24–0:38 — as SEARCH and FETCH scroll**

> "This is a live Google Lens query through SerpApi. The image bytes are
> uploaded directly — it never goes to a public host. Twelve candidates come
> back, and the raw provider response is being written to the run directory
> right now. Nothing here is cached or bundled."

**0:38–1:00 — when the ranked table lands. SLOW DOWN. This is the differentiator.**

> "Now look at the ranking, because this is the part most people get wrong.
> The three *highest* scores — 0.9855, 0.9845, 0.9813 — are all the same press
> photograph, republished by Windows Central, Axios and Forbes. Their
> face-region perceptual hash distance is 6, 12 and 10: nearly identical images.
>
> The one we select scores *lower*, 0.8503, from SiliconANGLE — with a hash
> distance of 28. That is a genuinely different photograph of the same person,
> and it is much stronger evidence. Ranking by similarity would have returned
> the weakest result with the biggest number."

*This is the single most important 20 seconds in the video. If anything gets cut,
it is not this.*

---

## Scene 3 — 1:00–1:14 · Distinct-Artifact Proof + chain write

**Screen:** the proof panel, then the `[CHAIN]` lines, still the same command.

**Say:**

> "Different SHA-256. Perceptual hash distance 28 against a same-photo cutoff of
> 15. Cosine 0.8503 against a threshold of 0.2149, which was calibrated at a
> false-accept rate of 7.8 times ten-to-the-minus-four on sixty-five thousand
> LFW pairs — not a number we picked.
>
> The evidence is canonicalised, SHA-256'd, and that hash goes to Base Sepolia."

**Then cut to the BaseScan tab.** Show the contract with its verified source, and
paste the tx hash from the run into the search bar.

> "Here is the transaction, and the contract with its source published and
> verified."

---

## Scene 4 — 1:14–1:30 · Independent verification

**Screen:** the second terminal — the venv with **only** `web3` installed.

```powershell
# prove what is NOT in this environment -- this prints nothing at all
pip list | findstr /I "insightface torch opencv scikit imagehash matplotlib"

# and how small it is: 43 packages, every one of them web3's own dependency
(pip list | Measure-Object -Line).Lines

python verify.py --bundle runs\<run_id>\bundle.json
```

**Expect:** the filter prints **nothing**, the count is **43**, then `VERIFIED`.

**Say:**

> "Different environment. No InsightFace, no torch, no OpenCV — `web3` and the
> standard library only. It recomputes the hash from the bundle, reads the
> registry, and confirms it. None of the pipeline code runs here."

*Showing the count matters: an empty `findstr` result alone could read as a
command that silently failed. 43 packages, none of them ours, is unambiguous.*

---

## Scene 5 — 1:30–1:46 · Tamper

**Screen:** visible, single-character edit. Do not use an editor — this is
faster and more obviously honest. These are PowerShell, and every one was run
before being written here.

```powershell
Copy-Item -Recurse runs\<run_id> tampered
Select-String tampered\bundle.json -Pattern source_url | Select-Object -First 1
```

Then change exactly one character — `siliconangle` → `sil**1**conangle`:

```powershell
(Get-Content tampered\bundle.json -Raw) -replace 'siliconangle','sil1conangle' | Set-Content tampered\bundle.json -Encoding utf8
.\.venv\Scripts\python.exe verify.py --bundle tampered\bundle.json
```

**Expect:**

```
claimed   : 9fe3f991...
recomputed: b6cdb60b...
TAMPER DETECTED -- evidence modified
```

**Say:**

> "One character. An `i` becomes a `1` in the source URL. The recomputed hash no
> longer matches what is on chain, and the verifier rejects it."

*Do this demo. It is the most convincing eight seconds in the video and most
teams will not have it.*

---

## Scene 6 — 1:46–2:00 · First-seen semantics, and the honest close

```powershell
.venv\Scripts\python.exe scripts\demo_duplicate.py --bundle runs\<run_id>\bundle.json
```

**Expect:** `Evidence already recorded` citing the original timestamp.

**Say:**

> "And it records first sighting. Re-submitting the same evidence is refused,
> citing the original timestamp — nobody can overwrite it or backdate it.
>
> Real face matching, a real search, a genuinely different photograph, a real
> chain record, independently verifiable. And to be precise about what this
> proves: the chain notarises *when* the claim was made and that it hasn't
> changed. It does not prove the claim is true. That distinction is in the
> README and in ETHICS.md."

*Ending on the limitation is deliberate. Saying it before a judge says it to you
is the difference between a defensible submission and a defensive one.*

---

## If something breaks mid-take

| failure | do this |
|---|---|
| Search returns nothing / network drops | Re-shoot Scene 2 with `--replay <run_id>` from the pre-flight capture. The banner will be visible and you **must** say "this is replay mode against a previously captured real run." Never hide it. |
| SerpApi quota exhausted | Same — replay. Check quota in pre-flight so this cannot surprise you. |
| RPC unreachable on the chain write | The pipeline prints `CHAIN WRITE SKIPPED` and continues; re-shoot, or narrate Scenes 4–6 against the earlier run that is already on chain. |
| Verdict comes back `SAME_PHOTO` instead of `DISTINCT_PHOTO` | Lens result ordering shifts over time. Do not re-run hoping for a better answer — that is fishing. Narrate what is on screen honestly; a republication correctly *labelled* as a republication still demonstrates the three-way verdict working. |
| Wrong number spoken | Retake the scene. |

**Do not** lower the threshold, re-run until you get a nicer number, or edit
`calibration/results.json` to make the demo look better. The whole submission
rests on those numbers being measured rather than chosen.

---

## Post-recording checklist

- [ ] Every number spoken matches what is on screen
- [ ] The words "identity confirmed" and any invented percentage do **not** appear
- [ ] Tamper demo is in the cut
- [ ] Duplicate-refusal demo is in the cut
- [ ] BaseScan tab clearly shows **verified** source
- [ ] If replay was used, the banner is visible and it is said out loud
- [ ] Under 2:10
- [ ] Uploaded, link tested **in an incognito window**
- [ ] Private key never on screen — check the terminal scrollback frame by frame
      around the chain write

That last one matters. `.env` is never printed by the pipeline and the logger
redacts secrets, but check anyway: an exposed key in a video cannot be unpublished.
