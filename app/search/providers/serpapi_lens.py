"""SerpApi google_lens provider.

Uploads the query image to SerpApi's Image API and searches by image_id, so the
face image is never placed on a public throwaway host. Verified in Phase 0:
POST /image -> image_id -> engine=google_lens returned 59 visual matches for a
derived crop whose SHA-256 exists nowhere on the web.
"""
from __future__ import annotations

import hashlib
import pathlib

import requests

from ..base import (Candidate, ProviderAuthError, ProviderError,
                    ProviderRateLimited, ProviderUnavailable, SearchProvider,
                    SearchResult)

SEARCH_URL = "https://serpapi.com/search"
UPLOAD_URL = "https://serpapi.com/image"
MAX_UPLOAD_BYTES = 500 * 1024


class SerpApiLens(SearchProvider):
    name = "serpapi_google_lens"

    def __init__(self, api_key: str, *, timeout: float = 120.0, logger=None):
        self._key = api_key
        self._timeout = timeout
        self._log = logger

    def is_configured(self) -> bool:
        return bool(self._key)

    def _say(self, msg, *a):
        if self._log:
            self._log(msg, *a)

    def _upload(self, image_bytes: bytes, raw_dir: pathlib.Path) -> str:
        if len(image_bytes) > MAX_UPLOAD_BYTES:
            raise ProviderError(
                f"Query image is {len(image_bytes) / 1024:.0f}KB; SerpApi's upload "
                f"limit is {MAX_UPLOAD_BYTES / 1024:.0f}KB.\n"
                "  Re-encode it smaller: python scripts/make_derived_input.py"
            )
        try:
            r = requests.post(
                UPLOAD_URL,
                params={"api_key": self._key},
                files={"image": ("query.jpg", image_bytes, "image/jpeg")},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise ProviderUnavailable(f"SerpApi upload unreachable: {e}") from None

        self._write_raw(raw_dir, "serpapi_upload.json", r.content)
        if r.status_code in (401, 403):
            raise ProviderAuthError(
                "SerpApi rejected the API key on upload.\n"
                "  Check SERPAPI_KEY in .env against https://serpapi.com/manage-api-key"
            )
        if r.status_code == 429:
            raise ProviderRateLimited("SerpApi rate limited during upload (HTTP 429).")
        if r.status_code != 200:
            raise ProviderUnavailable(
                f"SerpApi upload HTTP {r.status_code}: {r.text[:200]}"
            )

        try:
            j = r.json()
        except ValueError:
            raise ProviderUnavailable("SerpApi upload returned non-JSON.") from None

        image_id = j.get("image_id") or (j.get("image") or {}).get("id")
        if not image_id:
            raise ProviderError(f"SerpApi upload returned no image_id: {str(j)[:200]}")
        return image_id

    def _get(self, params: dict, raw_dir: pathlib.Path, fname: str) -> dict:
        try:
            r = requests.get(SEARCH_URL, params=params, timeout=self._timeout)
        except requests.RequestException as e:
            raise ProviderUnavailable(f"SerpApi unreachable: {e}") from None

        self._write_raw(raw_dir, fname, r.content)
        if r.status_code in (401, 403):
            raise ProviderAuthError("SerpApi rejected the API key.")
        if r.status_code == 429:
            raise ProviderRateLimited("SerpApi rate limited (HTTP 429).")

        try:
            j = r.json()
        except ValueError:
            raise ProviderUnavailable(
                f"SerpApi returned non-JSON (HTTP {r.status_code})."
            ) from None

        if "error" in j:
            err = str(j["error"])
            low = err.lower()
            if "run out" in low or "quota" in low or "plan" in low:
                raise ProviderRateLimited(
                    f"SerpApi quota exhausted: {err}\n"
                    "  Free tier is 250 searches/month; see https://serpapi.com/dashboard"
                )
            if "api key" in low or "invalid" in low:
                raise ProviderAuthError(f"SerpApi: {err}")
            raise ProviderError(f"SerpApi: {err}")
        return j

    def search(self, image_bytes: bytes, *, raw_dir: pathlib.Path,
               max_candidates: int) -> SearchResult:
        if not self.is_configured():
            raise ProviderAuthError("SERPAPI_KEY is not set in .env")

        sha = hashlib.sha256(image_bytes).hexdigest()
        image_id = self._upload(image_bytes, raw_dir)
        self._say("uploaded query image (%d bytes), image_id acquired", len(image_bytes))

        base = {"engine": "google_lens", "api_key": self._key, "image_id": image_id}
        data = self._get(base, raw_dir, "serpapi_lens.json")

        vm = data.get("visual_matches") or []
        cands = [
            Candidate(
                position=int(m.get("position", i + 1)),
                title=str(m.get("title", "")),
                page_url=str(m.get("link", "")),
                image_url=str(m.get("image") or m.get("thumbnail") or ""),
                thumbnail_url=str(m.get("thumbnail", "")),
                source=str(m.get("source", "")),
                provider=self.name,
            )
            for i, m in enumerate(vm)
        ]

        # A separate exact_matches query. The default type=all response contains
        # no 'exact_matches' key at all, so reading it there and finding nothing
        # would prove nothing -- it would only mean we never asked.
        exact_count = None
        try:
            ed = self._get({**base, "type": "exact_matches"}, raw_dir, "serpapi_exact.json")
            if "exact_matches" in ed:
                exact_count = len(ed.get("exact_matches") or [])
        except ProviderError as e:
            self._say("exact_matches probe failed (non-fatal): %s", type(e).__name__)

        return SearchResult(
            provider=self.name,
            candidates=self._dedupe(cands, max_candidates),
            raw_path=raw_dir / "serpapi_lens.json",
            query_image_sha256=sha,
            exact_match_count=exact_count,
            notes=f"{len(vm)} visual matches returned",
        )
