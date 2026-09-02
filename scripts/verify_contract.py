r"""Publish verified source for the deployed EvidenceRegistry on BaseScan.

Uses Etherscan V2 (one key, all chains) with solidity-standard-json-input, so
the explorer recompiles the exact input scripts/compile.py used -- no flattening
and no bytecode-mismatch guesswork.

    .venv\Scripts\python.exe scripts/verify_contract.py
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

import requests

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_settings  # noqa: E402

API = "https://api.etherscan.io/v2/api"
CHAIN_ID = 84532
EXPLORER = "https://sepolia.basescan.org"


def main() -> int:
    cfg = get_settings()
    if not cfg.basescan_api_key:
        print("\n  BASESCAN_API_KEY is not set in .env.\n"
              "  Get a free key at https://etherscan.io/myapikey (works for all\n"
              "  chains including Base Sepolia), then add to .env:\n"
              "      BASESCAN_API_KEY=your_key_here\n")
        return 2
    if not cfg.evidence_registry_address:
        print("\n  EVIDENCE_REGISTRY_ADDRESS is not set. Deploy first:\n"
              "      python scripts/deploy.py\n")
        return 2

    art = json.loads((ROOT / "build" / "EvidenceRegistry.json").read_text(encoding="utf-8"))
    addr = cfg.evidence_registry_address

    print(f"\n  contract : {addr}")
    print(f"  compiler : {art['solcLongVersion']}")
    print(f"  format   : solidity-standard-json-input (optimizer {art['optimizerRuns']} runs)")

    data = {
        "chainid": str(CHAIN_ID),
        "module": "contract",
        "action": "verifysourcecode",
        "apikey": cfg.basescan_api_key,
        "contractaddress": addr,
        "sourceCode": json.dumps(art["standardJsonInput"]),
        "codeformat": "solidity-standard-json-input",
        "contractname": f"{pathlib.Path(art['sourcePath']).name}:EvidenceRegistry",
        "compilerversion": art["solcLongVersion"],
        "optimizationUsed": "1",
        "runs": str(art["optimizerRuns"]),
        "constructorArguements": "",       # explorer's historical spelling
        "licenseType": "3",                # MIT
    }

    print("  submitting ...")
    r = requests.post(API, params={"chainid": str(CHAIN_ID)}, data=data, timeout=60)
    try:
        j = r.json()
    except ValueError:
        print(f"  non-JSON response (HTTP {r.status_code}): {r.text[:300]}\n")
        return 1

    if str(j.get("status")) != "1":
        msg = str(j.get("result") or j.get("message"))
        print(f"  SUBMIT FAILED: {msg}")
        if "already verified" in msg.lower():
            print(f"  Already verified: {EXPLORER}/address/{addr}#code\n")
            return 0
        print()
        return 1

    guid = j["result"]
    print(f"  guid     : {guid}")
    print("  polling for result ...")

    for attempt in range(20):
        time.sleep(5)
        s = requests.get(API, params={
            "chainid": str(CHAIN_ID), "module": "contract",
            "action": "checkverifystatus", "guid": guid,
            "apikey": cfg.basescan_api_key,
        }, timeout=30)
        res = str(s.json().get("result", ""))
        if "Pending" in res:
            print(f"    [{attempt + 1}] {res}")
            continue
        print(f"    -> {res}")
        ok = "Pass" in res or "already verified" in res.lower()
        print(f"\n  {'VERIFIED' if ok else 'NOT VERIFIED'}: {EXPLORER}/address/{addr}#code\n")
        return 0 if ok else 1

    print("\n  Timed out waiting for verification status.")
    print(f"  Check manually: {EXPLORER}/address/{addr}#code\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
