r"""Phase 0 / Gate 0 spike: is the Bright Data SERP API wired up, and does its
Google Lens path return usable visual matches?

Costs 1 credit per run (free tier = 5,000/month).

    .venv\Scripts\python.exe spikes/spike_brightdata.py
"""
import json, sys, urllib.parse
import requests
from _spike_common import (OUT, PUBLIC_TEST_IMAGE, UA, check_interpreter,
                           load_env, require)

ENDPOINT = "https://api.brightdata.com/request"


def main() -> None:
    check_interpreter()
    env = load_env()
    token, zone = require(env, "BRIGHTDATA_API_TOKEN", "BRIGHTDATA_SERP_ZONE")

    # Bright Data reaches Lens via uploadbyurl, which needs a PUBLIC image URL.
    # brd_json=1 asks Bright Data to return parsed JSON instead of raw HTML.
    lens_url = (
        "https://lens.google.com/uploadbyurl?url="
        + urllib.parse.quote(PUBLIC_TEST_IMAGE, safe="")
        + "&brd_json=1"
    )
    payload = {"zone": zone, "url": lens_url, "format": "raw"}

    print(f"\n  zone     : {zone}")
    print(f"  token    : {token[:6]}...{token[-4:]} (len {len(token)})")
    print(f"  image    : {PUBLIC_TEST_IMAGE.rsplit('/', 1)[-1]}")
    print("  calling  : https://api.brightdata.com/request ...")

    try:
        r = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json",
                     "User-Agent": UA},
            json=payload,
            timeout=120,
        )
    except requests.RequestException as e:
        print(f"\n  NETWORK FAILURE: {e}\n")
        sys.exit(1)

    print(f"  HTTP {r.status_code} ({len(r.content)} bytes)")

    raw = OUT / "brightdata_lens_raw.txt"
    raw.write_bytes(r.content)
    print(f"  raw response saved -> {raw}")

    if r.status_code == 401:
        print("\n  AUTH FAILED. Token rejected. Re-copy it from the zone's")
        print("  Overview tab in the Bright Data control panel.\n"); sys.exit(1)
    if r.status_code == 400 and "zone" in r.text.lower():
        print(f"\n  ZONE PROBLEM. Bright Data says: {r.text[:300]}")
        print("  Check BRIGHTDATA_SERP_ZONE matches the zone name exactly.\n"); sys.exit(1)
    if r.status_code != 200:
        print(f"\n  UNEXPECTED STATUS. Body: {r.text[:500]}\n"); sys.exit(1)

    try:
        data = r.json()
    except ValueError:
        print("\n  Response was not JSON (brd_json may not have applied).")
        print(f"  First 400 chars:\n{r.text[:400]}\n")
        print("  Plumbing works (HTTP 200) but parsing needs the HTML path.")
        sys.exit(0)

    keys = list(data) if isinstance(data, dict) else []
    print(f"  parsed JSON keys: {keys[:12]}")

    matches = []
    for k in ("visual_matches", "similar", "images", "organic"):
        v = data.get(k) if isinstance(data, dict) else None
        if isinstance(v, list) and v:
            matches = v
            print(f"\n  '{k}' -> {len(v)} entries")
            break

    if not matches:
        print("\n  NO VISUAL MATCHES PARSED. Inspect the saved raw response.")
        print("  Gate 0 note: plumbing OK, extraction needs work.\n"); sys.exit(0)

    print("\n  --- first 8 matches ---")
    for m in matches[:8]:
        if not isinstance(m, dict):
            continue
        title = str(m.get("title") or m.get("name") or "")[:58]
        link = str(m.get("link") or m.get("url") or m.get("source") or "")[:78]
        img = m.get("image") or m.get("thumbnail") or m.get("image_url") or ""
        print(f"   - {title}")
        print(f"       page : {link}")
        print(f"       image: {str(img)[:78]}")

    print(f"\n  RESULT: Bright Data Lens returned {len(matches)} candidates.")
    print("  Gate 0 (bright data leg): PASS\n")


if __name__ == "__main__":
    main()
