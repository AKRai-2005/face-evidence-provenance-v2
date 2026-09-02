"""Provider-agnostic search interface.

A provider returns CANDIDATES ONLY -- page URL plus image URL. It never decides
whether something matches; that is the verification stage's job, using face
embeddings. Keeping providers dumb is what stops the pipeline from degenerating
into "the search engine said so".
"""
from __future__ import annotations

import abc
import dataclasses
import pathlib
from typing import Iterable


class ProviderError(RuntimeError):
    """Base for provider failures. Message must be actionable."""


class ProviderAuthError(ProviderError):
    pass


class ProviderRateLimited(ProviderError):
    def __init__(self, msg: str, *, remaining: int | None = None):
        self.remaining = remaining
        super().__init__(msg)


class ProviderUnavailable(ProviderError):
    """Network / 5xx / timeout -- worth falling back to another provider."""


@dataclasses.dataclass(frozen=True)
class Candidate:
    """One search hit, before any face verification."""

    position: int
    title: str
    page_url: str
    image_url: str
    source: str
    provider: str
    thumbnail_url: str = ""

    def as_record(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class SearchResult:
    provider: str
    candidates: list[Candidate]
    raw_path: pathlib.Path | None
    query_image_sha256: str
    exact_match_count: int | None = None   # None = not queried; 0 = queried, none found
    notes: str = ""


class SearchProvider(abc.ABC):
    """Implementations must be honest: never fabricate or cache-substitute."""

    name: str = "abstract"

    @abc.abstractmethod
    def is_configured(self) -> bool:
        """True when credentials for this provider are present."""

    @abc.abstractmethod
    def search(self, image_bytes: bytes, *, raw_dir: pathlib.Path,
               max_candidates: int) -> SearchResult:
        """Run a real reverse-image search. Raises ProviderError on failure."""

    # -- shared helpers --------------------------------------------------
    @staticmethod
    def _write_raw(raw_dir: pathlib.Path, name: str, content: bytes) -> pathlib.Path:
        """Persist the untouched provider response.

        This is the audit trail that lets a third party confirm the candidates
        were discovered rather than invented.
        """
        raw_dir.mkdir(parents=True, exist_ok=True)
        p = raw_dir / name
        p.write_bytes(content)
        return p

    @staticmethod
    def _dedupe(cands: Iterable[Candidate], limit: int) -> list[Candidate]:
        seen: set[str] = set()
        out: list[Candidate] = []
        for c in cands:
            if not c.image_url or c.image_url in seen:
                continue
            seen.add(c.image_url)
            out.append(c)
            if len(out) >= limit:
                break
        return out
