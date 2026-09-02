"""Temporary public hosting for the query image.

Only some providers need this. SerpApi accepts a direct upload and returns an
image_id, so on that path the face image never leaves SerpApi. Bright Data
reaches Lens via lens.google.com/uploadbyurl, which requires a literal public
URL -- so hosting is provider-scoped, not a global pipeline stage.

Whatever is uploaded is deleted afterwards where the host supports it, and the
deletion (or the failure to delete) is logged. That is a privacy control, and
it is stated as one in ETHICS.md.
"""
from __future__ import annotations

import abc
import dataclasses

import requests

UA = "HHGoa2026-Task3/0.1 (evidence-provenance research)"


class ImageHostError(RuntimeError):
    pass


@dataclasses.dataclass
class HostedImage:
    url: str
    backend: str
    delete_token: str | None = None
    deleted: bool = False


class ImageHost(abc.ABC):
    name = "abstract"

    @abc.abstractmethod
    def upload(self, image_bytes: bytes) -> HostedImage: ...

    @abc.abstractmethod
    def delete(self, hosted: HostedImage) -> bool:
        """Return True only if the remote copy is confirmed gone."""


class CatboxHost(ImageHost):
    """litterbox.catbox.moe -- throwaway host with a 1-hour expiry.

    An expiring host is preferred over a permanent one: even if an explicit
    delete fails, the copy is not left indefinitely.
    """

    name = "catbox"
    ENDPOINT = "https://litterbox.catbox.moe/resources/internals/api.php"

    def __init__(self, *, timeout: float = 60.0, expiry: str = "1h"):
        self._timeout = timeout
        self._expiry = expiry

    def upload(self, image_bytes: bytes) -> HostedImage:
        try:
            r = requests.post(
                self.ENDPOINT,
                data={"reqtype": "fileupload", "time": self._expiry},
                files={"fileToUpload": ("query.jpg", image_bytes, "image/jpeg")},
                headers={"User-Agent": UA},
                timeout=self._timeout,
            )
        except requests.RequestException as e:
            raise ImageHostError(f"litterbox upload failed: {e}") from None

        if r.status_code != 200 or not r.text.startswith("http"):
            raise ImageHostError(
                f"litterbox upload failed (HTTP {r.status_code}): {r.text[:160]}"
            )
        return HostedImage(url=r.text.strip(), backend=self.name)

    def delete(self, hosted: HostedImage) -> bool:
        # litterbox exposes no delete API; the upload expires on its own.
        # Reported honestly rather than claimed as a deletion.
        return False


class NullHost(ImageHost):
    """Used when the provider accepts a direct upload, so nothing is hosted."""

    name = "none"

    def upload(self, image_bytes: bytes) -> HostedImage:
        raise ImageHostError(
            "This provider requires a public image URL, but IMAGE_HOST_BACKEND "
            "is 'serpapi_upload' (no external host).\n"
            "  Set IMAGE_HOST_BACKEND=catbox in .env to enable the Bright Data path."
        )

    def delete(self, hosted: HostedImage) -> bool:
        return True


def get_host(backend: str) -> ImageHost:
    if backend == "catbox":
        return CatboxHost()
    if backend in ("serpapi_upload", "none"):
        return NullHost()
    raise ImageHostError(f"unknown IMAGE_HOST_BACKEND: {backend!r}")
