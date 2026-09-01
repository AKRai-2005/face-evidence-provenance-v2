"""Typed configuration. Fails loudly and specifically on missing keys.

Nothing here ever logs a secret value; see logging_setup.RedactSecrets.
"""
from __future__ import annotations

import pathlib
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = pathlib.Path(__file__).resolve().parent.parent


class ConfigError(RuntimeError):
    """Raised with an actionable message when configuration is unusable."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- search providers ---
    serpapi_key: str = Field(default="", alias="SERPAPI_KEY")
    brightdata_api_token: str = Field(default="", alias="BRIGHTDATA_API_TOKEN")
    brightdata_serp_zone: str = Field(default="", alias="BRIGHTDATA_SERP_ZONE")
    image_host_backend: Literal["serpapi_upload", "catbox", "tmpfiles"] = Field(
        default="serpapi_upload", alias="IMAGE_HOST_BACKEND"
    )

    # --- blockchain ---
    base_sepolia_rpc: str = Field(
        default="https://sepolia.base.org", alias="BASE_SEPOLIA_RPC"
    )
    deployer_private_key: str = Field(default="", alias="DEPLOYER_PRIVATE_KEY")
    evidence_registry_address: str = Field(
        default="", alias="EVIDENCE_REGISTRY_ADDRESS"
    )

    # --- privacy ---
    subject_commitment_salt: str = Field(default="", alias="SUBJECT_COMMITMENT_SALT")

    # --- tunables (not secrets) ---
    max_candidates: int = 12
    fetch_timeout_s: float = 8.0
    max_download_bytes: int = 10 * 1024 * 1024
    min_face_px: int = 40           # see face/detector.py: below this, embeddings are noise
    min_det_score: float = 0.55

    @field_validator("deployer_private_key")
    @classmethod
    def _normalise_key(cls, v: str) -> str:
        v = v.strip()
        if v and not v.startswith("0x"):
            v = "0x" + v
        return v

    # --- explicit, actionable requirement checks ---
    def require_search(self) -> None:
        if not self.serpapi_key and not (
            self.brightdata_api_token and self.brightdata_serp_zone
        ):
            raise ConfigError(
                "No search provider configured.\n"
                "  Set SERPAPI_KEY in .env (free 250/month: https://serpapi.com/manage-api-key)\n"
                "  or BRIGHTDATA_API_TOKEN + BRIGHTDATA_SERP_ZONE."
            )

    def require_chain(self, *, need_contract: bool = True) -> None:
        if not self.deployer_private_key:
            raise ConfigError(
                "DEPLOYER_PRIVATE_KEY is not set in .env.\n"
                "  Generate a throwaway testnet wallet: python scripts/new_wallet.py"
            )
        if len(self.deployer_private_key) != 66:
            raise ConfigError(
                f"DEPLOYER_PRIVATE_KEY looks malformed (length {len(self.deployer_private_key)}, "
                "expected 66 incl. '0x')."
            )
        if need_contract and not self.evidence_registry_address:
            raise ConfigError(
                "EVIDENCE_REGISTRY_ADDRESS is not set in .env.\n"
                "  Deploy first: python scripts/deploy.py"
            )

    def require_salt(self) -> None:
        if len(self.subject_commitment_salt) < 32:
            raise ConfigError(
                "SUBJECT_COMMITMENT_SALT missing or too short (need >=32 hex chars).\n"
                '  Generate: python -c "import secrets;print(secrets.token_hex(32))"'
            )

    def secret_values(self) -> list[str]:
        """Everything the log filter must redact."""
        return [
            v for v in (
                self.serpapi_key,
                self.brightdata_api_token,
                self.deployer_private_key,
                self.subject_commitment_salt,
            ) if v and len(v) >= 8
        ]


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
