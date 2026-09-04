"""Google Cloud Vision WEB_DETECTION provider.

The second genuinely web-scale reverse-image source available on a free tier
(1,000 units/month, four times SerpApi's 250). Unlike Bright Data it needs no
public image host: the image is POSTed base64-encoded in the request body, so
the face image goes only to Google, exactly as on the SerpApi path.

WHAT IT RETURNS, AND HOW IT DIFFERS FROM LENS

  pagesWithMatchingImages   pages hosting a full or partial match -- gives BOTH
                            a page URL and an image URL, which is what the
                            evidence object needs
  visuallySimilarImages     image URLs only, no page. Used as a supplement, with
                            the image URL standing in as the source URL, because
                            that is honestly where the image lives.

Web Detection is tuned for "where does this image appear", so it leans toward
near-duplicates more than Lens does. That is fine here and arguably useful: this
pipeline explicitly separates republications from distinct photographs, and a
provider biased toward republications simply fills the SAME_PHOTO tier. It is a
redundancy path, not a replacement for Lens.
"""
from __future__ import annotations

import base64
import hashlib
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
        """Pages first -- they carry both a page URL and an image URL."""
        out: list[Candidate] = []
        pos = 0

        for page in (web.get("pagesWithMatchingImages") or []):
            page_url = str(page.get("url") or "")
            title = str(page.get("pageTitle") or "")
            imgs = (page.get("fullMatchingImages") or []) + \
                   (page.get("partialMatchingImages") or [])
            for im in imgs:
                u = str(im.get("url") or "")
                if not u:
                    continue
                pos += 1
                out.append(Candidate(
                    position=pos, title=title, page_url=page_url, image_url=u,
                    source=_host(page_url) or _host(u),
                    provider=GoogleVisionWebDetection.name))

        # Supplementary: image URLs with no known hosting page. The image URL
        # stands in as the source, because that is where the image actually is.
        for im in (web.get("visuallySimilarImages") or []):
            u = str(im.get("url") or "")
            if not u:
                continue
            pos += 1
            out.append(Candidate(
                position=pos, title="", page_url=u, image_url=u,
                source=_host(u), provider=GoogleVisionWebDetection.name))
        return out


def _host(url: str) -> str:
    import urllib.parse
    try:
        h = urllib.parse.urlsplit(url).hostname or ""
        return h[4:] if h.startswith("www.") else h
    except ValueError:
        return ""
