r"""Phase 0 / Gate 0 spike: the riskiest unknown in the project.

Question it answers: when we feed Google Lens a DERIVED artifact (a re-encoded
crop that exists nowhere byte-identical on the web), does it return non-identical
images of the same person -- or nothing useful?

If this returns only exact duplicates or junk, the architecture must change, and
we need to know that on Sept 1.

Costs 1-2 SerpApi searches (free tier = 250/month).

    .venv\Scripts\python.exe spikes/spike_lens.py
"""
import hashlib, io, json, pathlib, sys
import requests
from _spike_common import (OUT, PUBLIC_TEST_IMAGE, UA, check_interpreter,
                           load_env, require)

ROOT = pathlib.Path(__file__).resolve().parent.parent
SEARCH = "https://serpapi.com/search"
UPLOAD = "https://serpapi.com/image"
SRC = ROOT / "data" / "fixtures" / "B2_nadella_smiling.jpg"


def make_derived(src: pathlib.Path) -> tuple[bytes, str, str]:
    """Crop + resize + re-encode so the bytes exist nowhere on the web.

    This is the whole point of the project: the query artifact must not be
    findable by exact-file matching.
    """
    from PIL import Image
    im = Image.open(src).convert("RGB")
    w, h = im.size
    # centre-ish crop biased upward toward the face, then downscale + re-encode
    im = im.crop((int(w * 0.18), int(h * 0.04), int(w * 0.82), int(h * 0.86)))
    im = im.resize((im.width // 2, im.height // 2), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=82, optimize=True)
    b = buf.getvalue()
    return b, hashlib.sha256(b).hexdigest(), f"{im.width}x{im.height}"


def main() -> None:
    check_interpreter()
    env = load_env()
    (key,) = require(env, "SERPAPI_KEY")

    if not SRC.exists():
        print(f"\n  Missing fixture {SRC}. Run spike_face.py first.\n"); sys.exit(2)

    src_sha = hashlib.sha256(SRC.read_bytes()).hexdigest()
    derived, der_sha, dims = make_derived(SRC)
    dpath = OUT / "derived_query.jpg"; dpath.write_bytes(derived)

    print(f"\n  source fixture : {SRC.name}  sha256={src_sha[:16]}...")
    print(f"  DERIVED query  : {dims}  {len(derived)/1024:.0f}KB  sha256={der_sha[:16]}...")
    print(f"  distinct bytes : {'YES' if der_sha != src_sha else 'NO'}   -> {dpath}")

    if len(derived) > 500_000:
        print("  WARNING: >500KB, SerpApi upload will reject it.")

    # --- try the upload path first: no third-party host ever sees the face ---
    image_id, mode = None, None
    print("\n  [1] uploading to https://serpapi.com/image ...")
    try:
        r = requests.post(UPLOAD, params={"api_key": key},
                          files={"image": ("query.jpg", derived, "image/jpeg")},
                          headers={"User-Agent": UA}, timeout=90)
        print(f"      HTTP {r.status_code}")
        (OUT / "serpapi_upload_raw.json").write_bytes(r.content)
        if r.status_code == 200:
            j = r.json()
            image_id = j.get("image_id") or (j.get("image") or {}).get("id")
            print(f"      image_id: {image_id}")
            if image_id:
                mode = "upload (image_id)"
        else:
            print(f"      body: {r.text[:300]}")
    except Exception as e:
        print(f"      upload failed: {e}")

    params = {"engine": "google_lens", "api_key": key}
    if image_id:
        params["image_id"] = image_id
    else:
        mode = "public url (FALLBACK -- upload path unavailable)"
        params["url"] = PUBLIC_TEST_IMAGE
        print("\n  [1b] upload path unusable; falling back to a public URL.")
        print("       NOTE: this weakens the spike -- that URL IS on the web,")
        print("       so exact-match results are expected and uninformative.")

    print(f"\n  [2] google_lens search  (mode: {mode}) ...")
    try:
        r = requests.get(SEARCH, params=params, headers={"User-Agent": UA}, timeout=120)
    except requests.RequestException as e:
        print(f"      NETWORK FAILURE: {e}\n"); sys.exit(1)

    print(f"      HTTP {r.status_code} ({len(r.content)} bytes)")
    raw = OUT / "serpapi_lens_raw.json"; raw.write_bytes(r.content)
    print(f"      raw saved -> {raw}")

    try:
        data = r.json()
    except ValueError:
        print(f"      non-JSON body: {r.text[:300]}\n"); sys.exit(1)

    if "error" in data:
        print(f"\n  SERPAPI ERROR: {data['error']}")
        if "run out" in str(data["error"]).lower() or "quota" in str(data["error"]).lower():
            print("  -> monthly quota exhausted.")
        print(); sys.exit(1)

    info = data.get("search_metadata", {})
    print(f"      status={info.get('status')}  id={info.get('id')}")

    vm = data.get("visual_matches") or []
    print(f"\n  top-level keys: {list(data)}")
    print(f"  visual_matches: {len(vm)}")

    # The default type=all response contains NO 'exact_matches' key at all.
    # Reading data['exact_matches'] and finding nothing would prove nothing --
    # it would only mean we never asked. To support the distinct-artifact claim
    # honestly we must issue a SECOND, explicit exact_matches query.
    print("\n  [3] explicit type=exact_matches query (the real duplicate test) ...")
    ep = dict(params)
    ep["type"] = "exact_matches"
    em, em_known = [], False
    try:
        r2 = requests.get(SEARCH, params=ep, headers={"User-Agent": UA}, timeout=120)
        (OUT / "serpapi_exact_raw.json").write_bytes(r2.content)
        d2 = r2.json()
        if "error" in d2:
            print(f"      SerpApi error: {d2['error']}")
        else:
            em = d2.get("exact_matches") or []
            em_known = "exact_matches" in d2
            print(f"      HTTP {r2.status_code}  keys: {list(d2)}")
            print(f"      'exact_matches' key present: {em_known}  count: {len(em)}")
            for m in em[:5]:
                print(f"        - {str(m.get('title', ''))[:54]}  {str(m.get('link', ''))[:58]}")
    except Exception as e:
        print(f"      exact_matches query failed: {e}")

    if em_known and not em:
        print("\n  DISTINCT-ARTIFACT EVIDENCE: Google reports ZERO exact matches")
        print("  for these bytes, yet returned same-subject visual matches.")
        print("  Exact-file / reverse-hash matching is empirically ruled out.")
    elif em:
        print(f"\n  NOTE: {len(em)} exact matches exist -- the derived crop is still")
        print("  findable byte-wise. Strengthen the derivation before the demo.")
    else:
        print("\n  exact_matches INCONCLUSIVE (query failed). Do not claim zero.")

    if not vm:
        print("\n  ZERO VISUAL MATCHES. This is the architecture-changing outcome.")
        print("  Report to lead immediately.\n"); sys.exit(0)

    print("\n  --- first 10 visual matches ---")
    for m in vm[:10]:
        print(f"   - {str(m.get('title',''))[:60]}")
        print(f"       source: {str(m.get('source',''))[:40]}  link: {str(m.get('link',''))[:70]}")
        print(f"       image : {str(m.get('image') or m.get('thumbnail') or '')[:78]}")

    print(f"\n  RESULT: {len(vm)} candidates from a derived artifact.")
    print("  Next: Phase 3 downloads these and scores faces against the input.")
    print("  Gate 0 (lens leg): PASS\n")


if __name__ == "__main__":
    main()
