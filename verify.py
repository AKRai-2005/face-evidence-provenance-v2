#!/usr/bin/env python3
r"""Independent verifier for a task3 evidence bundle.

Dependencies: web3 + the Python standard library. Nothing else. No InsightFace,
no torch, no SerpApi, none of the pipeline code. Run it in a bare venv:

    pip install web3
    python verify.py --bundle runs/<run_id>/bundle.json

What it does:
  1. Re-canonicalises the evidence object from the bundle, using its OWN
     implementation of the SS6 rules (deliberately re-implemented below rather
     than imported -- a verifier that calls the code it is checking would share
     any bug in it, and prove correspondingly less).
  2. SHA-256s those bytes to get the evidence hash.
  3. Reads the on-chain record for that hash from EvidenceRegistry.
  4. Reports one of:
        VERIFIED               hash present on chain, bundle intact
        TAMPER DETECTED        bundle was modified after recording
        NOT REGISTERED         hash absent from the registry

The chain proves WHEN a claim was made, and that it has not changed since.
It does not, and cannot, prove the claim was true.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import unicodedata

DEFAULT_RPC = "https://sepolia.base.org"
DEFAULT_CONTRACT = "0x38157D4652304BA251edCcffa435A0bC3F55305a"
EXPLORER = "https://sepolia.basescan.org"

GET_EVIDENCE_ABI = [{
    "inputs": [{"internalType": "bytes32", "name": "evidenceHash", "type": "bytes32"}],
    "name": "getEvidence",
    "outputs": [
        {"internalType": "bool", "name": "exists", "type": "bool"},
        {"internalType": "uint64", "name": "timestamp", "type": "uint64"},
        {"internalType": "address", "name": "recorder", "type": "address"},
        {"internalType": "bytes32", "name": "subjectCommitment", "type": "bytes32"},
        {"internalType": "string", "name": "sourceDomain", "type": "string"},
    ],
    "stateMutability": "view",
    "type": "function",
}]

EXIT_OK, EXIT_TAMPER, EXIT_UNREGISTERED, EXIT_USAGE, EXIT_RPC = 0, 1, 2, 3, 4


# --------------------------------------------------------------------------
# Canonicalization -- an independent re-implementation of the SS6 rules.
# Must agree byte-for-byte with app/evidence/canonicalizer.py; a test asserts it.
# --------------------------------------------------------------------------
def _clean(value, path="$"):
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        raise ValueError(f"float at {path}: canonical evidence must not contain floats")
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, (list, tuple)):
        return [_clean(v, f"{path}[{i}]") for i, v in enumerate(value)]
    if isinstance(value, dict):
        out = {}
        for k in sorted(value):
            if not isinstance(k, str):
                raise ValueError(f"non-string key at {path}: {k!r}")
            out[unicodedata.normalize("NFC", k)] = _clean(value[k], f"{path}.{k}")
        return out
    raise ValueError(f"unsupported type at {path}: {type(value).__name__}")


def canonicalize(obj: dict) -> bytes:
    return json.dumps(
        _clean(obj), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def compute_hash(evidence: dict) -> str:
    return hashlib.sha256(canonicalize(evidence)).hexdigest()


# --------------------------------------------------------------------------
def load_bundle(path: pathlib.Path) -> dict:
    if not path.exists():
        print(f"\n  BUNDLE NOT FOUND: {path}\n")
        sys.exit(EXIT_USAGE)
    try:
        # utf-8-sig, not utf-8: it strips a leading BOM if present and behaves
        # identically when there is none. This matters in practice -- Notepad,
        # and PowerShell 5.1's Set-Content -Encoding utf8, both write UTF-8 WITH
        # a BOM. A reviewer hand-editing the bundle to try the tamper demo would
        # otherwise be told "not valid JSON" and reasonably conclude the tool was
        # broken, instead of seeing the TAMPER DETECTED they were looking for.
        b = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as e:
        print(f"\n  BUNDLE IS NOT VALID JSON: {e}\n")
        sys.exit(EXIT_USAGE)

    missing = [k for k in ("evidence", "evidence_sha256") if k not in b]
    if missing:
        print(f"\n  BUNDLE SCHEMA MISMATCH: missing {', '.join(missing)}")
        print("  Expected a task3 bundle.json produced by app.main.\n")
        sys.exit(EXIT_USAGE)
    return b


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Independently verify a task3 evidence bundle against Base Sepolia."
    )
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    ap.add_argument("--rpc", default=DEFAULT_RPC)
    ap.add_argument("--contract", default=None,
                    help=f"registry address (default: bundle's, else {DEFAULT_CONTRACT})")
    args = ap.parse_args()

    bundle = load_bundle(args.bundle)
    evidence = bundle["evidence"]
    claimed = str(bundle["evidence_sha256"]).lower().removeprefix("0x")
    chain_meta = bundle.get("chain") or {}
    contract = (args.contract or chain_meta.get("contract") or DEFAULT_CONTRACT)

    print()
    print("  EVIDENCE BUNDLE VERIFICATION")
    print("  " + "=" * 62)
    print(f"  bundle    : {args.bundle}")
    print(f"  run_id    : {bundle.get('run_id', '<none>')}")

    # -- 1. recompute -----------------------------------------------------
    try:
        recomputed = compute_hash(evidence)
    except ValueError as e:
        print(f"\n  BUNDLE CANNOT BE CANONICALISED: {e}\n")
        return EXIT_USAGE

    print(f"\n  claimed   : {claimed}")
    print(f"  recomputed: {recomputed}")

    if recomputed != claimed:
        print("\n  TAMPER DETECTED -- evidence modified")
        print("  The bundle's evidence object does not hash to the value it claims.")
        print("  Someone edited the evidence after it was recorded.\n")
        return EXIT_TAMPER
    print("  -> bundle is internally consistent")

    # -- 2. read the chain ------------------------------------------------
    try:
        from web3 import Web3
    except ImportError:
        print("\n  web3 is not installed.  pip install web3\n")
        return EXIT_USAGE

    print(f"\n  rpc       : {args.rpc}")
    print(f"  contract  : {contract}")
    w3 = Web3(Web3.HTTPProvider(args.rpc, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print(f"\n  RPC UNREACHABLE: {args.rpc}")
        print("  Try --rpc https://sepolia.base.org\n")
        return EXIT_RPC

    try:
        c = w3.eth.contract(address=w3.to_checksum_address(contract), abi=GET_EVIDENCE_ABI)
        exists, ts, recorder, commitment, domain = c.functions.getEvidence(
            bytes.fromhex(recomputed)
        ).call()
    except Exception as e:
        print(f"\n  CHAIN READ FAILED: {type(e).__name__}: {e}\n")
        return EXIT_RPC

    if not exists:
        print("\n  NOT REGISTERED")
        print("  The bundle is internally consistent, but this evidence hash has")
        print("  never been recorded in the registry at that address.\n")
        return EXIT_UNREGISTERED

    import datetime
    when = datetime.datetime.fromtimestamp(ts, datetime.timezone.utc)
    ch = commitment.hex() if isinstance(commitment, (bytes, bytearray)) else str(commitment)

    print("\n  on-chain record")
    print(f"    first seen        : {when.strftime('%Y-%m-%dT%H:%M:%SZ')}  (unix {ts})")
    print(f"    recorder          : {recorder}")
    print(f"    subjectCommitment : {ch.removeprefix('0x')}")
    print(f"    sourceDomain      : {domain}")

    ev_domain = evidence.get("source_domain", "")
    if domain and ev_domain and domain != ev_domain:
        print(f"\n  TAMPER DETECTED -- domain mismatch")
        print(f"  chain says {domain!r}, bundle says {ev_domain!r}.\n")
        return EXIT_TAMPER

    print("\n  VERIFIED")
    print(f"  This evidence was recorded on Base Sepolia at "
          f"{when.strftime('%Y-%m-%dT%H:%M:%SZ')} and has not changed since.")
    print(f"  {EXPLORER}/address/{contract}")
    if chain_meta.get("tx_hash"):
        print(f"  {EXPLORER}/tx/{chain_meta['tx_hash']}")
    print("\n  Note: the registry proves WHEN this claim was made and that it has")
    print("  not been altered. It does not attest that the match is correct.\n")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
