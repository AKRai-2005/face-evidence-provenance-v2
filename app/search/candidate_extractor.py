"""Download candidate images, under hard bounds.

Every fetch here is to an arbitrary third-party URL that a search engine handed
us. Treat all of it as hostile input: cap the size, cap the time, check the
content type, and never let one bad candidate abort the run.
"""
from __future__ import annotations

import dataclasses
import hashlib
import pathlib

import requests

UA = "HHGoa2026-Task3/0.1 (evidence-provenance research)"

ALLOWED_TYPES = ("image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp")


@dataclasses.dataclass(frozen=True)
class FetchedCandidate:
    """A candidate image that was successfully downloaded, or a reason it wasn't."""

    position: int
    page_url: str
    image_url: str
    source: str
    ok: bool
    reason: str = ""
    content: bytes | None = None
    sha256: str = ""
    content_type: str = ""
    path: pathlib.Path | None = None

    @property
    def size(self) -> int:
        return len(self.content) if self.content else 0


class CandidateFetcher:
    def __init__(self, *, timeout: float = 8.0, max_bytes: int = 10 * 1024 * 1024,
                 logger=None):
        self._timeout = timeout
        self._max = max_bytes
        self._log = logger
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": UA})

    def _say(self, msg, *a):
        if self._log:
            self._log(msg, *a)

    def fetch_one(self, cand, out_dir: pathlib.Path) -> FetchedCandidate:
        base = dict(position=cand.position, page_url=cand.page_url,
                    image_url=cand.image_url, source=cand.source)

        if not cand.image_url.lower().startswith(("http://", "https://")):
            return FetchedCandidate(**base, ok=False, reason="non-http URL")

        try:
            r = self._session.get(cand.image_url, timeout=self._timeout, stream=True)
        except requests.Timeout:
            return FetchedCandidate(**base, ok=False, reason=f"timeout >{self._timeout:g}s")
        except requests.RequestException as e:
            return FetchedCandidate(**base, ok=False, reason=type(e).__name__)

        if r.status_code != 200:
            # 403 is common: publishers block hotlinking. Not an error in our run.
            return FetchedCandidate(**base, ok=False, reason=f"HTTP {r.status_code}")

        ctype = r.headers.get("content-type", "").split(";")[0].strip().lower()
        if ctype not in ALLOWED_TYPES:
            return FetchedCandidate(**base, ok=False,
                                    reason=f"content-type {ctype or 'unknown'}")

        # Declared length is a hint, not a guarantee -- enforce the cap on the
        # actual stream so a lying Content-Length cannot exhaust memory.
        declared = r.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > self._max:
            return FetchedCandidate(**base, ok=False,
                                    reason=f"oversized ({int(declared)/1e6:.1f}MB declared)")

        chunks, total = [], 0
        try:
            for chunk in r.iter_content(64 * 1024):
                total += len(chunk)
                if total > self._max:
                    return FetchedCandidate(**base, ok=False,
                                            reason=f"oversized (>{self._max/1e6:.0f}MB)")
                chunks.append(chunk)
        except requests.RequestException as e:
            return FetchedCandidate(**base, ok=False, reason=f"stream failed: {type(e).__name__}")
        finally:
            r.close()

        content = b"".join(chunks)
        if not content:
            return FetchedCandidate(**base, ok=False, reason="empty body")

        sha = hashlib.sha256(content).hexdigest()
        out_dir.mkdir(parents=True, exist_ok=True)
        ext = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp",
               "image/gif": ".gif", "image/bmp": ".bmp"}.get(ctype, ".bin")
        path = out_dir / f"cand{cand.position:02d}_{sha[:12]}{ext}"
        path.write_bytes(content)

        return FetchedCandidate(**base, ok=True, content=content, sha256=sha,
                                content_type=ctype, path=path)

    def fetch_all(self, candidates, out_dir: pathlib.Path) -> list[FetchedCandidate]:
        out = []
        for c in candidates:
            f = self.fetch_one(c, out_dir)
            out.append(f)
            if f.ok:
                self._say("candidate %d OK %s %.0fKB sha=%s",
                          f.position, f.content_type, f.size / 1024, f.sha256[:12])
            else:
                self._say("candidate %d SKIPPED (%s) %s",
                          f.position, f.reason, f.source or f.image_url[:40])
        ok = sum(1 for f in out if f.ok)
        self._say("downloaded %d/%d candidates", ok, len(out))
        return out
