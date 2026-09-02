"""Bright Data SERP API provider (fallback).

MEASURED RELIABILITY WARNING -- read before trusting this path.

Bright Data's SERP zone serves plain Google reliably (821KB of HTML, or 125KB of
parsed JSON with brd_json=1, on every attempt). Its REVERSE-IMAGE paths do not:

    lens.google.com/uploadbyurl + brd_lens + brd_json   1 of 6 attempts non-empty
    www.google.com/searchbyimage                        244KB twice, then 0 bytes
    lens.google.com/uploadbyurl + brd_json (no brd_lens) 0 bytes every time

and the single non-empty Lens response (1.3KB) contained only tab metadata, no
visual matches at all -- against SerpApi's 86KB and 59 matches for the same
image. Measured 2026-09-01; see README "Provider reliability".

This class is therefore a genuine but unreliable fallback. It makes real network
calls, retries with backoff, and raises ProviderUnavailable when it cannot
deliver. It never substitutes cached or fabricated results -- a fallback that
quietly invents candidates would be worse than having no fallback at all.
"""
from __future__ import annotations

import hashlib
import pathlib
import time
import urllib.parse

import requests

from ..base import (Candidate, ProviderAuthError, ProviderError,
                    ProviderRateLimited, ProviderUnavailable, SearchProvider,
                    SearchResult)

ENDPOINT = "https://api.brightdata.com/request"


class BrightDataLens(SearchProvider):
    name = "brightdata_google_lens"

    def __init__(self, api_token: str, zone: str, *, timeout: float = 120.0,
                 attempts: int = 4, logger=None):
        self._token = api_token
        self._zone = zone
        self._timeout = timeout
        self._attempts = attempts
        self._log = logger

    def is_configured(self) -> bool:
        return bool(self._token and self._zone)

    def _say(self, msg, *a):
        if self._log:
            self._log(msg, *a)

    def _post(self, target_url: str) -> requests.Response:
        return requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {self._token}",
                     "Content-Type": "application/json"},
            json={"zone": self._zone, "url": target_url, "format": "raw"},
            timeout=self._timeout,
        )

    def search(self, image_bytes: bytes, *, raw_dir: pathlib.Path,
               max_candidates: int, public_image_url: str | None = None) -> SearchResult:
        if not self.is_configured():
            raise ProviderAuthError(
                "BRIGHTDATA_API_TOKEN / BRIGHTDATA_SERP_ZONE are not set in .env"
            )
        if not public_image_url:
            raise ProviderError(
                "Bright Data reaches Lens via lens.google.com/uploadbyurl, which "
                "needs a publicly reachable image URL.\n"
                "  Set IMAGE_HOST_BACKEND=catbox in .env to enable this provider."
            )

        sha = hashlib.sha256(image_bytes).hexdigest()
        enc = urllib.parse.quote(public_image_url, safe="")
        target = (f"https://lens.google.com/uploadbyurl?url={enc}"
                  f"&brd_lens=visual_matches&brd_json=1")

        last = ""
        for attempt in range(1, self._attempts + 1):
            try:
                r = self._post(target)
            except requests.RequestException as e:
                last = f"network error: {e}"
                self._say("attempt %d/%d failed: %s", attempt, self._attempts, last)
                time.sleep(2 ** attempt)
                continue

            self._write_raw(raw_dir, f"brightdata_lens_attempt{attempt}.json", r.content)

            if r.status_code in (401, 403):
                raise ProviderAuthError(
                    "Bright Data rejected the token.\n"
                    "  Re-copy it from the zone's Overview tab in the control panel."
                )
            if r.status_code == 429:
                raise ProviderRateLimited("Bright Data rate limited (HTTP 429).")
            if r.status_code != 200:
                last = f"HTTP {r.status_code}: {r.text[:160]}"
                time.sleep(2 ** attempt)
                continue
            if not r.content:
                # The documented failure mode: HTTP 200 with an empty body.
                last = "HTTP 200 with empty body (Lens tab unsupported by this zone)"
                self._say("attempt %d/%d: empty body, retrying", attempt, self._attempts)
                time.sleep(2 ** attempt)
                continue

            try:
                j = r.json()
            except ValueError:
                last = "non-JSON body"
                time.sleep(2 ** attempt)
                continue

            cands = self._extract(j)
            if cands:
                return SearchResult(
                    provider=self.name,
                    candidates=self._dedupe(cands, max_candidates),
                    raw_path=raw_dir / f"brightdata_lens_attempt{attempt}.json",
                    query_image_sha256=sha,
                    exact_match_count=None,
                    notes=f"succeeded on attempt {attempt}",
                )
            last = f"parsed JSON but no visual matches (keys: {list(j)[:6]})"
            self._say("attempt %d/%d: %s", attempt, self._attempts, last)
            time.sleep(2 ** attempt)

        raise ProviderUnavailable(
            f"Bright Data Lens returned no usable candidates after "
            f"{self._attempts} attempts. Last: {last}\n"
            "  This path is known-unreliable (measured 1/6 non-empty); see the\n"
            "  README 'Provider reliability' section. Raw responses are in the\n"
            "  run directory for inspection."
        )

    @staticmethod
    def _extract(j: dict) -> list[Candidate]:
        for key in ("visual_matches", "similar", "images", "organic"):
            arr = j.get(key)
            if isinstance(arr, list) and arr:
                out = []
                for i, m in enumerate(arr):
                    if not isinstance(m, dict):
                        continue
                    img = m.get("image") or m.get("thumbnail") or m.get("image_url") or ""
                    out.append(Candidate(
                        position=int(m.get("position", i + 1)),
                        title=str(m.get("title") or m.get("name") or ""),
                        page_url=str(m.get("link") or m.get("url") or ""),
                        image_url=str(img),
                        thumbnail_url=str(m.get("thumbnail", "")),
                        source=str(m.get("source") or m.get("domain") or ""),
                        provider=BrightDataLens.name,
                    ))
                return out
        return []
