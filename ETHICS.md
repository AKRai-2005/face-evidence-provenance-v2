# Ethics, privacy and limitations

Face search is the most heavily regulated capability this project could have
touched. This document states what the tool does, what it deliberately does not
do, and where it is weak. Several of the limitations below are unflattering;
they are here because a reviewer who finds them unaided should not be finding
them for the first time.

---

## 1. Scope: what this is, and what it is not

This is a **provenance tool**. It takes one face image, searches the public web
for images containing the same face, verifies the strongest candidate with face
embeddings, and notarises a cryptographic fingerprint of that finding on a
public blockchain so a third party can confirm the finding has not been altered
since it was made.

It is **not** a people-search service. It does not:

- build or persist identity profiles — each run is self-contained,
- map a face to a name (it reports a URL and a similarity score; any name that
  appears does so because it was in the page title the search engine returned),
- maintain a face database or gallery, or index anyone over time,
- store embeddings anywhere durable — the 512-d vector exists in memory for the
  duration of a run and is never written to disk or to the chain.

The output of a run is a claim of the form *"an image at this URL contains a
face whose embedding is within a calibrated distance of the input's"*. That is
narrower than "this is a photograph of X", and the tool is careful never to say
the latter.

## 2. Demo subject policy

Demonstrations use **either the author's own face, or a public figure whose
images are already broadly indexed by the search engines involved** — never a
private individual.

The fixtures committed to this repository are Wikimedia Commons photographs of
Satya Nadella and Sundar Pichai, both of whom are extensively photographed
public figures. Their provenance is recorded in `data/fixtures/SOURCES.md`.

This is a real constraint, not a formality: running this pipeline against a
private individual would produce exactly the kind of capability the project is
framed against, and the consent gate (§5) states so before the first run.

## 3. No biometric data on the blockchain

**Only a salted commitment reaches the chain.** The contract stores:

```
evidenceHash       SHA-256 of the canonical evidence object
subjectCommitment  SHA-256(salt || int8-quantised embedding)
sourceDomain       the domain only, e.g. "wikipedia.org"
timestamp, recorder
```

The embedding itself is never published. The reasoning matters more than the
mechanism: **a public ledger is immutable and world-readable, which makes it the
worst possible place to put a face template.** A biometric template published to
Base Sepolia could not be deleted, corrected, or withdrawn — not by us, not by
the subject, not ever. "We can revoke it later" is not available as a mitigation
on a blockchain, so the only safe design is not to publish it at all.

The commitment still does useful work: revealing the salt later proves a given
record concerns a given subject, without ever having broadcast biometrics. The
salt lives only in `.env`, which is gitignored and checked by
`scripts/check_secrets.py`.

**Honest weakness:** the commitment is only as strong as the salt. If the salt
leaks, an adversary holding a gallery of candidate faces could brute-force the
commitment by embedding each face, quantising, and hashing. The salt is
therefore a secret, not a formality. A per-record random salt would be stronger
still than the single deployment-wide salt used here; that is a known
limitation, chosen for simplicity of verification and stated rather than hidden.

**Only the domain, not the full URL, is stored on chain.** The full URL lives in
the evidence bundle and is covered by `evidenceHash`, so it is provably fixed at
record time — but it is not permanently broadcast to everyone. This keeps a
specific page about a specific person off the permanent public record while
still making it tamper-evident.

## 4. Image handling and ephemeral hosting

The primary search path (**SerpApi**) uploads the query image directly to the
provider and searches by the returned `image_id`. **No third-party throwaway
host ever holds the face image on this path.** The original build plan assumed
public hosting was unavoidable; it is not, and avoiding it is a genuine privacy
improvement over the planned design.

The fallback path (**Bright Data**) reaches Google Lens via
`lens.google.com/uploadbyurl`, which requires a literal public URL. Only on that
path is the image hosted, and only then:

- the host is `litterbox.catbox.moe` with a **1-hour expiry**,
- the run logs the upload and what happened to it afterwards.

**Honest weakness:** litterbox exposes no delete API. `CatboxHost.delete()`
therefore returns `False` and the log says *"left to expire (host has no delete
API)"* rather than claiming a deletion we do not perform. The expiry is the
actual control. If a stronger guarantee is needed, use a signed-URL backend you
control; that is why `IMAGE_HOST_BACKEND` is configurable.

Downloaded candidate images are written into the run directory as an audit
trail. `runs/` is gitignored. Delete it when you are finished; nothing in the
pipeline depends on retaining it except `--replay`.

## 5. Consent gate

The first run prints a notice and records acknowledgement in `.consent`
(gitignored). It states the demo-subject policy, that a score above threshold is
evidence rather than proof, and that the chain records *when* a claim was made
rather than whether it was true. It is deliberately cheap — one prompt, once —
and exists to make the operator read those three facts before the tool runs.

It is an acknowledgement by the **operator**. It is emphatically not consent
from the person in the photograph, and it should not be mistaken for it.

## 6. Stated limitations

**Demographic bias.** Face-recognition error rates are not uniform across
demographic groups. NIST's FRVT evaluations have repeatedly found higher false
match rates for some groups than others, varying by algorithm. The threshold in
`calibration/results.json` is a **single global threshold** measured on LFW,
which is well known to be predominantly male and predominantly light-skinned.
A threshold calibrated on that distribution should not be assumed to deliver the
same false-accept rate for everyone, and this tool does not attempt per-group
calibration. This is the most consequential limitation in the project.

**Search recall is skewed toward the already-famous.** The pipeline can only
find what a search engine has already indexed. For a well-photographed public
figure it returns dozens of candidates; for a private individual with no
indexed images it will usually return nothing useful. That asymmetry is a
property of the underlying index, not of the face matching. It also means a
null result is weak evidence of absence.

**Similarity above threshold is evidence, not proof of identity.** A cosine of
0.74 against a threshold of X means the embeddings are closer than X% of
impostor pairs were during calibration. It does not establish identity. Twins,
close relatives, lookalikes, heavy occlusion, and adversarially perturbed images
can all move that number. The pipeline is worded accordingly and never prints
"identity confirmed" or a confidence percentage it did not compute.

**The chain notarises the claim; it does not validate it.** This is the single
most important sentence in this document. `EvidenceRegistry` proves that a
particular evidence object existed at a particular time and has not changed
since. It has no opinion on whether the match was correct. A confidently wrong
result, notarised, is still confidently wrong — it is merely now tamper-evident.
Immutability is not accuracy.

**Quality gating is a blunt instrument.** Faces below the size, detection-score
or sharpness gates are rejected outright, because they embed to noise (measured:
out-of-focus faces score ~0.01 against every identity). This prevents meaningless
scores but also means the tool silently declines to work on poor-quality inputs
rather than degrading gracefully.

**A single provider carries the demo.** Bright Data's reverse-image paths proved
unreliable in measurement (see the README's provider-reliability table), so in
practice SerpApi's Google Lens is the working path. The fallback is real but
should not be relied upon.

## 7. What we would need before this could be used for anything real

Nothing here is production-ready, and the list is not short: per-group threshold
calibration, per-record salts, a subject-facing mechanism to contest or annotate
a record, a retention policy for `runs/`, legal review against GDPR/BIPA and
their analogues (biometric templates are a special category of personal data
under several of these regimes), and an operator-accountability model stronger
than "whoever holds the private key". This is a hackathon submission
demonstrating a verification technique, and it should be read as one.

## 8. Licensing

InsightFace's pretrained models — including `buffalo_l`, used here — are
released by their authors **for non-commercial research purposes only**. This
submission is a non-commercial research/hackathon artefact and falls within
that, but any commercial use would require different models or a licence from
the authors. See the README's Licensing section.

Fixture images are Wikimedia Commons photographs under their respective licences;
see `data/fixtures/SOURCES.md`.
