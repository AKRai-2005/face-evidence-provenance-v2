r"""Gate 5 guard: refuse to ship a repo that leaks a secret.

    .venv\Scripts\python.exe scripts/check_secrets.py

Checks, in order of how badly each would end the project:
  1. .env is not tracked by git, and never has been in the history.
  2. No tracked file contains a value that is currently in .env.
  3. No tracked file matches a high-signal secret pattern.
  4. No tracked file contains an absolute local path (SS13).

Exits non-zero on any finding, so it can gate a commit or CI.
"""
from __future__ import annotations

import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

# A bare 64-hex string cannot be classified by shape: a SHA-256 digest and a
# secp256k1 private key are both exactly that. This project is full of legitimate
# 64-hex values -- evidence hashes, subject commitments, image digests, tx hashes
# -- so a shape-only rule flagged every evidence artifact in sample_run/ and the
# proof block in the README. Require key-ish context on the same line instead.
#
# The authoritative checks are the two that do not depend on shape at all:
# .env must be untracked and absent from history, and no live value from .env
# may appear in any tracked file. Those catch a real leak whatever it looks like.
KEYISH = r"(?:priv(?:ate)?[_-]?key|secret|passwd|password|mnemonic|seed[_-]?phrase)"

PATTERNS = [
    ("private key (hex, in key-ish context)",
     re.compile(KEYISH + r"""["'\s:=]{0,12}(?:0x)?[0-9a-fA-F]{64}\b""", re.I)),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("PEM block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z-]{10,}")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[0-9A-Za-z]{36,}\b")),
]

# Absolute local paths must not appear in tracked files.
ABS_PATH = re.compile(r"(?:[A-Za-z]:\\Users\\|/Users/|/home/)[^\s\"'<>|]{2,}")

SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".onnx",
                 ".zip", ".pdf", ".ico"}
# Files whose job is to describe secrets, or which legitimately carry hashes.
ALLOW = {
    ".env.example",
    "scripts/check_secrets.py",
    "build/EvidenceRegistry.json",   # bytecode and hashes, not secrets
}


def tracked_files() -> list[str]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        print("  not a git repository; nothing to check")
        return []
    return [f for f in out.stdout.splitlines() if f.strip()]


# Only values under a secret-shaped KEY NAME are secrets. .env also holds
# ordinary config -- the RPC URL, the image-host backend, the deployed contract
# address -- which legitimately appear in source as defaults. Matching on every
# value flagged 7 files for containing "https://sepolia.base.org".
SECRET_KEY_HINTS = ("KEY", "TOKEN", "SECRET", "SALT", "PASSWORD", "PRIVATE")
NOT_SECRET_KEYS = {"EVIDENCE_REGISTRY_ADDRESS", "BASE_SEPOLIA_RPC",
                   "IMAGE_HOST_BACKEND", "BRIGHTDATA_SERP_ZONE"}


def env_values() -> list[str]:
    p = ROOT / ".env"
    if not p.exists():
        return []
    vals = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k in NOT_SECRET_KEYS:
            continue
        if not any(h in k.upper() for h in SECRET_KEY_HINTS):
            continue
        if len(v) >= 12:            # short values produce false positives
            vals.append(v)
    return vals


def main() -> int:
    findings: list[str] = []
    files = tracked_files()

    # 1. .env must not be tracked, now or ever.
    if ".env" in files:
        findings.append("CRITICAL: .env is tracked by git")
    hist = subprocess.run(["git", "log", "--all", "--pretty=format:%H", "--", ".env"],
                          cwd=ROOT, capture_output=True, text=True)
    if hist.returncode == 0 and hist.stdout.strip():
        n = len(hist.stdout.strip().splitlines())
        findings.append(f"CRITICAL: .env appears in {n} commit(s) in history")

    secrets = env_values()
    print(f"  {len(files)} tracked files, {len(secrets)} live secret values to look for")

    for rel in files:
        if rel in ALLOW or pathlib.Path(rel).suffix.lower() in SKIP_SUFFIXES:
            continue
        p = ROOT / rel
        if not p.exists():
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue

        # 2. live secret values
        for v in secrets:
            if v in text:
                findings.append(f"CRITICAL: {rel} contains a value from .env")
                break

        # 3. secret-shaped strings
        for label, pat in PATTERNS:
            m = pat.search(text)
            if m:
                findings.append(f"{rel}: possible {label} -- {m.group(0)[:14]}...")

        # 4. absolute local paths
        m = ABS_PATH.search(text)
        if m:
            findings.append(f"{rel}: absolute local path -- {m.group(0)[:60]}")

    print()
    if findings:
        print("  SECRET / PORTABILITY CHECK FAILED")
        for f in findings:
            print(f"    - {f}")
        print()
        return 1

    print("  SECRET CHECK CLEAN")
    print("    .env untracked and absent from history")
    print("    no live secret values in tracked files")
    print("    no secret-shaped strings, no absolute local paths")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
