"""Google Cloud Vision WEB_DETECTION provider.

The second genuinely web-scale reverse-image source available on a free tier
(1,000 units/month, four times SerpApi's 250). Unlike Bright Data it needs no
public image host: the image is POSTed base64-encoded in the request body, so
the face image goes only to Google, exactly as on the SerpApi path.

WHAT IT RETURNS, AND HOW IT DIFFERS FROM LENS

  pagesWithMatchingImages   pages hosting a full or partial match -- gives BOTH
                            a page URL and an image URL, which is what the
                            evidence object needs
  visuallySimilarImages     image URLs only, no page. The image URL stands in
                            as the source, because that is honestly where the
                            image lives.

The two buckets are NOT interchangeable, and _extract() interleaves them for a
measured reason: pages are near-duplicates (they are where this image already
appears), while visually-similar images are different photographs. Emitting
pages first starved the second bucket entirely at a cap of 12.

Web Detection is still tuned for "where does this image appear", so it remains
weaker than Lens for distinct-photograph evidence. It is a redundancy path, not
a replacement.
"""
from __future__ import annotations

import base64
import dataclasses
import hashlib
import itertools
import pathlib

import requests

from ..base import (Candidate, ProviderAuthError, ProviderError,
                    ProviderRateLimited, ProviderUnavailable, SearchProvider,
                    SearchResult)

ENDPOINT = "https://vision.googleapis.com/v1/images:annotate"

# Vision rejects very large payloads; base64 inflates by ~4/3. The pipeline's
# derived inputs are ~20KB, so this is generous headroom rather than a squeeze.
MAX_IMAGE_BYTES = 4 * 1024 * 1024


class GoogleVisionWebDetection(SearchProvider):
    name = "google_vision_web_detection"

    def __init__(self, api_key: str, *, timeout: float = 90.0, logger=None):
        self._key = api_key
        self._timeout = timeout
        self._log = logger

    def is_configured(self) -> bool:
        return bool(self._key)

    def _say(self, msg, *a):
        if self._log:
            self._log(msg, *a)

    def search(self, image_bytes: bytes, *, raw_dir: pathlib.Path,
               max_candidates: int) -> SearchResult:
        if not self.is_configured():
            raise ProviderAuthError("GOOGLE_VISION_API_KEY is not set in .env")
        if len(image_bytes) > MAX_IMAGE_BYTES:
            raise ProviderError(
                f"Image is {len(image_bytes)/1e6:.1f}MB; Cloud Vision's practical "
                f"limit here is {MAX_IMAGE_BYTES/1e6:.0f}MB.\n"
                "  Re-encode it smaller: python scripts/make_derived_input.py"
            )

        sha = hashlib.sha256(image_bytes).hexdigest()
        body = {
            "requests": [{
                "image": {"content": base64.b64encode(image_bytes).decode("ascii")},
                "features": [{"type": "WEB_DETECTION",
                              "maxResults": max(max_candidates * 4, 50)}],
            }]
        }

        try:
            r = requests.post(ENDPOINT, params={"key": self._key}, json=body,
                              timeout=self._timeout)
        except requests.RequestException as e:
            raise ProviderUnavailable(f"Cloud Vision unreachable: {e}") from None

        self._write_raw(raw_dir, "google_vision_web.json", r.content)

        if r.status_code in (401, 403):
            raise ProviderAuthError(
                "Cloud Vision rejected the key.\n"
                "  Check GOOGLE_VISION_API_KEY, that the Vision API is ENABLED for\n"
                "  the project, and that billing is attached (the 1,000/month free\n"
                "  tier still requires a billing account).\n"
                f"  Response: {r.text[:200]}"
            )
        if r.status_code == 429:
            raise ProviderRateLimited(
                "Cloud Vision rate limited / quota exhausted (HTTP 429).\n"
                "  Free tier is 1,000 units per month."
            )
        if r.status_code != 200:
            raise ProviderUnavailable(
                f"Cloud Vision HTTP {r.status_code}: {r.text[:200]}")

        try:
            data = r.json()
        except ValueError:
            raise ProviderUnavailable("Cloud Vision returned non-JSON.") from None

        responses = data.get("responses") or [{}]
        first = responses[0] if responses else {}
        if "error" in first:
            msg = str(first["error"].get("message", first["error"]))
            low = msg.lower()
            if "quota" in low or "exceeded" in low:
                raise ProviderRateLimited(f"Cloud Vision: {msg}")
            if "api key" in low or "permission" in low or "disabled" in low:
                raise ProviderAuthError(f"Cloud Vision: {msg}")
            raise ProviderError(f"Cloud Vision: {msg}")

        web = first.get("webDetection") or {}
        cands = self._extract(web)

        return SearchResult(
            provider=self.name,
            candidates=self._dedupe(cands, max_candidates),
            raw_path=raw_dir / "google_vision_web.json",
            query_image_sha256=sha,
            exact_match_count=len(web.get("fullMatchingImages") or []) or None,
            notes=(f"{len(web.get('pagesWithMatchingImages') or [])} pages, "
                   f"{len(web.get('visuallySimilarImages') or [])} visually similar"),
        )

    @staticmethod
    def _extract(web: dict) -> list[Candidate]:
        """Interleave the two buckets, because they answer different questions.

        Measured on a real response, they are not interchangeable:

            pagesWithMatchingImages   cosine med 0.978, face-pHash 12-14
                                      -> 0 of 7 were a distinct photograph
            visuallySimilarImages     cosine med 0.414, face-pHash 30-32
                                      -> 4 of 4 were a distinct photograph

        Pages are where this image ALREADY APPEARS, so they are republications
        almost by definition. Visually-similar images are different photographs,
        though many are different PEOPLE and will fall below the threshold.

        Emitting all pages first -- as this did -- meant that with 62 page URLs
        against 20 visually-similar ones and a cap of 12, the visually-similar
        bucket was truncated away entirely. The Vision path could therefore
        never surface a distinct photograph at all, only republications, which
        is exactly the weak evidence this project exists to distinguish.

        Interleaving guarantees both kinds are seen within the cap and lets the
        face matching decide. A visually-similar candidate that turns out to be
        a different person simply falls below the threshold and costs one
        download.
        """
        pages: list[Candidate] = []
        for page in (web.get("pagesWithMatchingImages") or []):
            page_url = str(page.get("url") or "")
            title = str(page.get("pageTitle") or "")
            imgs = (page.get("fullMatchingImages") or []) + \
                   (page.get("partialMatchingImages") or [])
            for im in imgs:
                u = str(im.get("url") or "")
                if not u:
                    continue
                pages.append(Candidate(
                    position=0, title=title, page_url=page_url, image_url=u,
                    source=_host(page_url) or _host(u),
                    provider=GoogleVisionWebDetection.name))

        similar: list[Candidate] = []
        for im in (web.get("visuallySimilarImages") or []):
            u = str(im.get("url") or "")
            if not u:
                continue
            # No hosting page is known for these, so the image URL stands in as
            # the source -- honestly where the image lives.
            similar.append(Candidate(
                position=0, title="", page_url=u, image_url=u,
                source=_host(u), provider=GoogleVisionWebDetection.name))

        out: list[Candidate] = []
        for a, b in itertools.zip_longest(pages, similar):
            for cand in (a, b):
                if cand is not None:
                    out.append(dataclasses.replace(cand, position=len(out) + 1))
        return out


def _host(url: str) -> str:
    import urllib.parse
    try:
        h = urllib.parse.urlsplit(url).hostname or ""
        return h[4:] if h.startswith("www.") else h
    except ValueError:
        return ""
