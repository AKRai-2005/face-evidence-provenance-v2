"""Provider and fetcher failure handling (SS9).

Every one of these paths must produce a typed error with an actionable message,
never a traceback and never a silent substitution.
"""
import json
import pathlib

import pytest
import requests

from app.search.base import (Candidate, ProviderAuthError, ProviderError,
                             ProviderRateLimited, ProviderUnavailable,
                             SearchProvider)
from app.search.candidate_extractor import CandidateFetcher
from app.search.image_host import CatboxHost, ImageHostError, NullHost, get_host
from app.search.providers.serpapi_lens import SerpApiLens


class _Resp:
    def __init__(self, status=200, body=b"", ctype="application/json", headers=None):
        self.status_code = status
        self.content = body
        self.headers = {"content-type": ctype, **(headers or {})}
        self.reason = "x"

    @property
    def text(self):
        return self.content.decode("utf-8", "replace")

    def json(self):
        return json.loads(self.content.decode("utf-8"))

    def iter_content(self, n):
        for i in range(0, len(self.content), n):
            yield self.content[i:i + n]

    def close(self):
        pass


# ---------------------------------------------------------------- providers
def test_unconfigured_provider_raises_auth_error(tmp_path):
    with pytest.raises(ProviderAuthError):
        SerpApiLens("").search(b"x", raw_dir=tmp_path, max_candidates=5)


def test_oversized_upload_is_refused_before_the_network(tmp_path, monkeypatch):
    """SerpApi caps uploads at 500KB; catch it locally with a fix, not a 4xx."""
    called = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: called.append(1))
    with pytest.raises(ProviderError) as ei:
        SerpApiLens("k").search(b"x" * (600 * 1024), raw_dir=tmp_path, max_candidates=5)
    assert "500KB" in str(ei.value) or "500" in str(ei.value)
    assert not called, "must not hit the network for an image we know is too big"


@pytest.mark.parametrize("status,exc", [
    (401, ProviderAuthError), (403, ProviderAuthError),
    (429, ProviderRateLimited), (500, ProviderUnavailable),
])
def test_upload_http_errors_map_to_typed_exceptions(tmp_path, monkeypatch, status, exc):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(status, b"{}"))
    with pytest.raises(exc):
        SerpApiLens("k").search(b"x", raw_dir=tmp_path, max_candidates=5)


def test_network_failure_on_upload_is_provider_unavailable(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("no route")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(ProviderUnavailable):
        SerpApiLens("k").search(b"x", raw_dir=tmp_path, max_candidates=5)


def test_quota_exhaustion_is_reported_as_rate_limited(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, b'{"image_id":"i"}'))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        200, b'{"error":"You have run out of searches this month"}'))
    with pytest.raises(ProviderRateLimited) as ei:
        SerpApiLens("k").search(b"x", raw_dir=tmp_path, max_candidates=5)
    assert "250" in str(ei.value)      # tells the operator the free-tier limit


def test_raw_responses_are_written_for_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "post", lambda *a, **k: _Resp(200, b'{"image_id":"i"}'))
    monkeypatch.setattr(requests, "get", lambda *a, **k: _Resp(
        200, json.dumps({"visual_matches": [
            {"position": 1, "title": "t", "link": "https://e.example/p",
             "image": "https://e.example/i.jpg", "source": "E"}]}).encode()))
    res = SerpApiLens("k").search(b"x", raw_dir=tmp_path, max_candidates=5)
    assert (tmp_path / "serpapi_upload.json").exists()
    assert (tmp_path / "serpapi_lens.json").exists()
    assert len(res.candidates) == 1
    assert res.candidates[0].page_url == "https://e.example/p"


def test_candidates_are_deduped_by_image_url():
    dupes = [Candidate(i, "t", f"https://e.example/{i}", "https://e.example/same.jpg",
                       "E", "p") for i in range(5)]
    assert len(SearchProvider._dedupe(dupes, 10)) == 1


def test_dedupe_respects_the_limit():
    cands = [Candidate(i, "t", "p", f"https://e.example/{i}.jpg", "E", "p")
             for i in range(20)]
    assert len(SearchProvider._dedupe(cands, 12)) == 12


# ------------------------------------------------------------------ fetcher
def _cand(url="https://e.example/i.jpg"):
    return Candidate(1, "t", "https://e.example/p", url, "E", "p")


def test_non_http_url_is_skipped(tmp_path):
    f = CandidateFetcher().fetch_one(_cand("data:image/png;base64,AAAA"), tmp_path)
    assert not f.ok and "non-http" in f.reason


def test_403_is_recorded_not_raised(tmp_path, monkeypatch):
    """Publishers block hotlinking constantly; it must not abort the run."""
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: _Resp(403, b""))
    f = CandidateFetcher().fetch_one(_cand(), tmp_path)
    assert not f.ok and "403" in f.reason


def test_non_image_content_type_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "get",
                        lambda *a, **k: _Resp(200, b"<html>", "text/html"))
    f = CandidateFetcher().fetch_one(_cand(), tmp_path)
    assert not f.ok and "text/html" in f.reason


def test_declared_oversize_is_rejected_without_downloading(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: _Resp(
        200, b"x", "image/jpeg", {"content-length": str(50 * 1024 * 1024)}))
    f = CandidateFetcher(max_bytes=1024).fetch_one(_cand(), tmp_path)
    assert not f.ok and "oversized" in f.reason


def test_stream_exceeding_cap_is_rejected_even_if_length_lies(tmp_path, monkeypatch):
    """A lying Content-Length must not be able to exhaust memory."""
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: _Resp(
        200, b"x" * 5000, "image/jpeg", {"content-length": "10"}))
    f = CandidateFetcher(max_bytes=1024).fetch_one(_cand(), tmp_path)
    assert not f.ok and "oversized" in f.reason


def test_timeout_is_recorded(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.Timeout()
    monkeypatch.setattr(requests.Session, "get", boom)
    f = CandidateFetcher(timeout=8).fetch_one(_cand(), tmp_path)
    assert not f.ok and "timeout" in f.reason


def test_empty_body_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(requests.Session, "get",
                        lambda *a, **k: _Resp(200, b"", "image/jpeg"))
    f = CandidateFetcher().fetch_one(_cand(), tmp_path)
    assert not f.ok and "empty" in f.reason


def test_successful_fetch_hashes_and_writes(tmp_path, monkeypatch):
    body = b"\xff\xd8\xff" + b"j" * 400
    monkeypatch.setattr(requests.Session, "get",
                        lambda *a, **k: _Resp(200, body, "image/jpeg"))
    f = CandidateFetcher().fetch_one(_cand(), tmp_path)
    assert f.ok and f.size == len(body)
    assert f.path.exists() and f.path.suffix == ".jpg"
    import hashlib
    assert f.sha256 == hashlib.sha256(body).hexdigest()


def test_one_bad_candidate_does_not_abort_the_batch(tmp_path, monkeypatch):
    seq = [_Resp(403, b""), _Resp(200, b"\xff\xd8\xffok", "image/jpeg")]
    monkeypatch.setattr(requests.Session, "get", lambda *a, **k: seq.pop(0))
    out = CandidateFetcher().fetch_all(
        [_cand("https://e.example/a.jpg"), _cand("https://e.example/b.jpg")], tmp_path)
    assert [f.ok for f in out] == [False, True]


# --------------------------------------------------------------- image host
def test_null_host_explains_how_to_enable_hosting():
    with pytest.raises(ImageHostError) as ei:
        NullHost().upload(b"x")
    assert "IMAGE_HOST_BACKEND=catbox" in str(ei.value)


def test_get_host_rejects_an_unknown_backend():
    with pytest.raises(ImageHostError):
        get_host("dropbox")


def test_catbox_delete_reports_failure_honestly():
    """litterbox exposes no delete API. We must not claim a deletion we do not
    perform -- the 1h expiry is the actual control (ETHICS.md SS4)."""
    from app.search.image_host import HostedImage
    assert CatboxHost().delete(HostedImage("https://x", "catbox")) is False
