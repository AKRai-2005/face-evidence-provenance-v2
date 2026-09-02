r"""Demonstrate first-seen semantics: re-record an evidence hash and be refused.

    .venv\Scripts\python.exe scripts/demo_duplicate.py --bundle runs/<id>/bundle.json

Reads the evidence hash out of a bundle that has already been recorded, attempts
to record it again, and shows the registry refusing it while citing the ORIGINAL
timestamp. Exists so the on-camera demo is one clean command rather than an
inline snippet typed under pressure.

Costs no gas: the duplicate is detected by an eth_call before any transaction is
signed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.blockchain.client import EXPLORER, ChainClient, ChainError  # noqa: E402
from app.blockchain.registry import AlreadyRecorded, Registry  # noqa: E402
from app.config import ConfigError, get_settings  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    args = ap.parse_args()

    if not args.bundle.exists():
        print(f"\n  BUNDLE NOT FOUND: {args.bundle}\n")
        return 2

    b = json.loads(args.bundle.read_text(encoding="utf-8"))
    ev_hash = b["evidence_sha256"]
    chain = b.get("chain") or {}
    commitment = chain.get("subject_commitment") or ("00" * 32)
    domain = b["evidence"]["source_domain"]

    cfg = get_settings()
    try:
        cfg.require_chain()
    except ConfigError as e:
        print(f"\nCONFIG ERROR\n{e}\n")
        return 2

    try:
        client = ChainClient(cfg.base_sepolia_rpc, cfg.deployer_private_key)
        reg = Registry(client, cfg.evidence_registry_address)
    except ChainError as e:
        print(f"\nCHAIN ERROR\n{e}\n")
        return 3

    print()
    print("  FIRST-SEEN SEMANTICS")
    print("  " + "=" * 66)
    print(f"  contract     : {reg.address}")
    print(f"  evidenceHash : {ev_hash}")
    print()

    existing = reg.get_evidence(ev_hash)
    if not existing.exists:
        print("  This hash is NOT yet on chain, so there is nothing to duplicate.")
        print("  Record it first with:  python -m app.main --image data/input.jpg")
        print()
        return 1

    when = dt.datetime.fromtimestamp(existing.timestamp, dt.timezone.utc)
    print(f"  already recorded at : {when.strftime('%Y-%m-%dT%H:%M:%SZ')}")
    print(f"  recorder            : {existing.recorder}")
    print(f"  sourceDomain        : {existing.source_domain}")
    print()
    print("  attempting to record the SAME evidence hash again ...")
    print()

    try:
        reg.record_evidence(ev_hash, commitment, domain)
    except AlreadyRecorded as e:
        for line in str(e).splitlines():
            print(f"    {line}")
        print()
        print("  The registry timestamps the FIRST sighting of a piece of evidence.")
        print("  A later party cannot overwrite it, cannot re-date it, and cannot")
        print("  claim to have found it earlier than it was recorded.")
        print(f"\n  {EXPLORER}/address/{reg.address}\n")
        return 0
    except ChainError as e:
        print(f"  UNEXPECTED CHAIN ERROR\n{e}\n")
        return 3

    print("  !!! THE DUPLICATE WAS ACCEPTED -- first-seen semantics are broken.\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
