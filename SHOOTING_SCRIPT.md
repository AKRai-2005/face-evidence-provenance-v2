# Shooting script — Task 3 demo video

**Target 110–125 s.** Every number spoken below is real and reproducible from
this repository. Do not round them up, and do not say anything the code did not
compute.

Scene 2b is roughly 15 seconds. It is written as a *substitute* for the ranking
narration inside Scene 2, not an addition — if the live run shows the inversion
clearly, keep Scene 2 and skip 2b; if it does not, cut Scene 2 short and run 2b
instead. Doing both fits only if you are running under 100 s elsewhere.

Record on **Sept 6**, not Sept 7.

---

## Language discipline — read before you speak

This is a scoring criterion in disguise, and it is the easiest thing to lose on
camera when you are improvising.

| say | never say |
|---|---|
| "high face similarity" | "identity confirmed" |
| "same-subject candidate" | "verified person" |
| "cosine 0.83 against a calibrated threshold of 0.2149" | "83% match", "99.9% confident" |
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

# 5. both frozen runs verify -- these are your fallbacks, prove them NOW
.venv\Scripts\python.exe verify.py --bundle sample_run\bundle.json
.venv\Scripts\python.exe verify.py --bundle sample_run_2\bundle.json
#    both must print VERIFIED. They read the live chain, so this also confirms
#    the RPC is up before you start recording.

# 6. CAPTURE A FALLBACK RUN NOW, so a network failure costs a retake, not the shoot
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
> The *highest* scores are the same press photograph republished by different
> outlets — their face-region perceptual hash distance is small, in single
> figures or low teens. Near-identical images.
>
> The one we select scores *lower*, with a hash distance above the cutoff of
> fifteen. That is a genuinely different photograph of the same person, and it
> is much stronger evidence. Ranking by similarity alone would have returned the
> weakest result with the biggest number.
>
> And the score is an interval, not a point. The input is a crop we chose, so we
> embed it six ways and report the median with its range."

**Read the numbers off the screen, not off this page.** Lens returns a different
candidate set from one day to the next — the frozen `sample_run/` matched
manofmany.com and `sample_run_2/` matched businessinsider.com. The *structure*
is stable (republications cluster at face-pHash distance <= 14, distinct
photographs at >= 18, against a calibrated cutoff of 15); the individual outlets
are not.

**If the live ranking does not show a clean inversion, do not improvise.** Say
"here it is unambiguously, in a run I committed" and cut to Scene 2b. A live run
can come back with no republication in it at all, and then the most important
claim in the video has nothing on screen behind it.

*This is the single most important 20 seconds in the video. If anything gets cut,
it is not this.*

---

## Scene 2b — the ranking inversion, guaranteed · *insurance, and the better shot*

**Use this whenever the live ranking is ambiguous — and consider using it even
when it isn't.** A live run shows whatever Lens returned that minute. This run
is committed, so it shows the same thing every time, and it shows it more
clearly than any live run has.

```powershell
Get-Content sample_run_2\run.log | Select-String "VERIFY" | ForEach-Object { ($_ -split "INFO\s+")[1] }
```

**On screen:** eleven scored candidates, including the two the pipeline
demoted. The `-split` strips the log timestamp — without it the verdict column
wraps onto a second line at 100 columns and the table stops being readable.

**Say:**

> "Here is the same thing in a run I committed, so you can check it yourself.
> Lens returned twelve candidates; eleven could be fetched and scored — one host
> served the wrong content type and was skipped, and the log says so.
>
> The *highest* score in the whole run is 0.9915 — Wikimedia Commons. That is
> the photograph my input was cropped from, republished. Its face-region hash
> distance is 2: it is the same picture.
>
> The pipeline demotes it, and reports a Business Insider photograph at 0.8761
> with a hash distance of 22 instead — a genuinely different photograph of the
> same person. A system that ranked by similarity would have handed you the
> weakest evidence with the biggest number, and called it the best match."

**Then, if you have the seconds, verify it:**

```powershell
.venv\Scripts\python.exe verify.py --bundle sample_run_2\bundle.json
```

> "And that run is notarised too — different subject, different transaction."

*Why this earns its place: the ranking inversion is the one design decision a
judge is most likely to doubt, and this is the only place in the video where
both verdict tiers appear side by side, from one input, in an artifact they can
re-run after the video ends.*

---

## Scene 3 — 1:00–1:14 · Distinct-Artifact Proof + chain write

**Screen:** the proof panel, then the `[CHAIN]` lines, still the same command.

**Say:**

> "Different SHA-256. Perceptual hash distance well above the same-photo cutoff
> of 15. The cosine is far above a threshold of 0.2149, which was calibrated at a
> false-accept rate of 6.8 times ten-to-the-minus-four on sixty-six thousand
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

**Use the frozen run, not your live one.** The matched domain in a live run is
whatever Lens returned that day, so a hard-coded replacement string may match
nothing — `-replace` then fails *silently*, the bundle stays valid, and the
verifier prints VERIFIED on camera in the middle of your tamper demo.
`sample_run_2` always contains `businessinsider`.

```powershell
Copy-Item -Recurse sample_run_2 tampered
Select-String tampered\bundle.json -Pattern source_url | Select-Object -First 1
```

Then change exactly one character — `businessinsider` → `bus**1**nessinsider`:

```powershell
(Get-Content tampered\bundle.json -Raw) -replace 'businessinsider','bus1nessinsider' | Set-Content tampered\bundle.json -Encoding utf8
.\.venv\Scripts\python.exe verify.py --bundle tampered\bundle.json
```

**Expect:**

```
claimed   : 0c2529df728dc7705bf3bf36034749b0d1a5651e1b18953bea69d1a8c10d8745
recomputed: <completely different>
TAMPER DETECTED -- evidence modified
```

**Say:**

> "One character. The `i` in businessinsider becomes a one. The recomputed hash
> no longer matches what is on chain, and the verifier rejects it."

*Do this demo. It is the most convincing eight seconds in the video and most
teams will not have it.*

---

## Scene 6 — 1:46–1:58 · The negative control

**Screen:**

```powershell
.venv\Scripts\python.exe scripts\demo_negative_control.py --run runs\<run_id>
```

**Expect:** `0 / 11` false accepts, roughly a 3x margin, `PASS`.

**Say:**

> "And the obvious question — does it ever say yes to the wrong person? Same
> candidates, same threshold, but a different public figure as the probe. The
> highest impostor score is 0.06 against a threshold of 0.21. Zero false
> accepts, and every genuine score beats every impostor by nearly half a point."

*This is the answer to the sharpest question a judge can ask, and it costs no
network, no quota and no gas — so it works even if everything else is down.
Twelve seconds well spent.*

---

## Scene 7 — 1:58–2:12 · First-seen semantics, and the honest close

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

## Hard questions, with the measured answers

Have these numbers ready. Every one is reproducible from a script in the repo.

**"Does it ever say yes to the wrong person?"**
`scripts/demo_negative_control.py` — highest impostor 0.0629 vs threshold
0.2149, a 3.4x margin, 0/11 false accepts. Passed on two independent candidate
sets from different days.

**"Your threshold is calibrated on LFW, but you point this at the open web."**
The best question anyone can ask, and it is measured:
`scripts/validate_in_domain.py` — 297 genuine and 264 impostor pairs built
entirely from images the pipeline downloaded. 0 false accepts, 0 false rejects,
separation 0.4131. Any threshold in (0.1185, 0.5316] separates perfectly; the
calibrated 0.2149 sits inside that window.

**"How confident is that number?"**
It is a median over six augmented views, not a point estimate. The two committed
runs report 0.9623 [0.9445-0.9679] and 0.8761 [0.8598-0.8767]. When that range
straddles the threshold the verdict is UNCERTAIN rather than a call. Reporting
the un-augmented number was quietly optimistic — it sits at the top of its own
range — which is why the median is reported instead.

**"Isn't 0.9868 AUC low for ArcFace on LFW?"**
The ~99.8% headline is accuracy on LFW's curated 6,000-pair protocol. This uses
harder combinatorial pairs across 450 identities. It is not label noise: the
highest-confidence face is the centred subject face in 98.7% of a 149-image
sample.

**"Why gate the input's face quality but not the candidates'?"**
Measured: 0 of 34 matched candidate faces would fail the input gate, and quality
correlates with score at r = -0.02. A gate there would prevent no observed
failure. The input is the one image a user can get wrong, and a bad input
poisons every comparison; a bad candidate just scores low.

**"What if the match is wrong?"**
Then the chain notarises a wrong claim, tamper-evidently. It records *when* a
claim was made and that it has not changed — never that it is true. Say this
before it is asked.

**"Could someone use this to find a private individual?"**
Recall depends entirely on what a search engine has already indexed. For a
well-photographed public figure it returns dozens of candidates; for someone
with no indexed images it returns nothing useful. That asymmetry is a property
of the index, not of our matching — and it is why a null result is weak evidence
of absence. See ETHICS.md.

---

## Post-recording checklist

- [ ] Every number spoken matches what is on screen
- [ ] The words "identity confirmed" and any invented percentage do **not** appear
- [ ] Tamper demo is in the cut, and it used `sample_run_2` (a live run's domain
      may not contain the string the command replaces, and `-replace` fails
      silently — you would show VERIFIED during your tamper demo)
- [ ] Duplicate-refusal demo is in the cut
- [ ] The ranking inversion is on screen somewhere — Scene 2 or Scene 2b. If it
      is in neither, the strongest claim in the project went unshown
- [ ] BaseScan tab clearly shows **verified** source
- [ ] If replay was used, the banner is visible and it is said out loud
- [ ] Under 2:10
- [ ] Uploaded, link tested **in an incognito window**
- [ ] Private key never on screen — check the terminal scrollback frame by frame
      around the chain write

That last one matters. `.env` is never printed by the pipeline and the logger
redacts secrets, but check anyway: an exposed key in a video cannot be unpublished.
