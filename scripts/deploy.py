r"""Deploy EvidenceRegistry to Base Sepolia and pin the address into .env.

    .venv\Scripts\python.exe scripts/deploy.py

Deploys once. If EVIDENCE_REGISTRY_ADDRESS is already set and live, this
refuses rather than silently spending gas on a second copy -- pass --force to
override.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.blockchain.client import EXPLORER, ChainClient, ChainError  # noqa: E402
from app.blockchain.registry import Registry  # noqa: E402
from app.config import ConfigError, get_settings  # noqa: E402


def pin_address(address: str) -> bool:
    env = ROOT / ".env"
    if not env.exists():
        return False
    txt = env.read_text(encoding="utf-8")
    if re.search(r"^EVIDENCE_REGISTRY_ADDRESS=.*$", txt, re.M):
        txt = re.sub(r"^EVIDENCE_REGISTRY_ADDRESS=.*$",
                     f"EVIDENCE_REGISTRY_ADDRESS={address}", txt, count=1, flags=re.M)
    else:
        txt += f"\nEVIDENCE_REGISTRY_ADDRESS={address}\n"
    env.write_text(txt, encoding="utf-8")
    return True


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="deploy even if one is pinned")
    args = ap.parse_args()

    cfg = get_settings()
    try:
        cfg.require_chain(need_contract=False)
    except ConfigError as e:
        print(f"\nCONFIG ERROR\n{e}\n"); return 2

    if cfg.evidence_registry_address and not args.force:
        print(f"\n  Already pinned: {cfg.evidence_registry_address}")
        print(f"  {EXPLORER}/address/{cfg.evidence_registry_address}")
        print("  Use --force to deploy another copy.\n")
        return 0

    from scripts.compile import compile_registry
    art = compile_registry()
    print(f"\n  compiled  : solc {art['solcVersion']}, optimizer {art['optimizerRuns']} runs, "
          f"{len(art['bytecode']) // 2 - 1} bytes")

    try:
        client = ChainClient(cfg.base_sepolia_rpc, cfg.deployer_private_key)
        bal = client.balance_wei()
        print(f"  deployer  : {client.address}")
        print(f"  chainId   : {client.chain_id}")
        print(f"  balance   : {client.w3.from_wei(bal, 'ether')} ETH")
        print("  deploying ...")
        reg, res = Registry.deploy(client)
    except ChainError as e:
        print(f"\nCHAIN ERROR\n{e}\n"); return 3

    cost = client.w3.from_wei(res.cost_wei(), "ether")
    print(f"\n  address   : {reg.address}")
    print(f"  tx        : {res.tx_hash}")
    print(f"  block     : {res.block_number}  gasUsed={res.gas_used}  cost={cost} ETH")
    print(f"  remaining : {client.w3.from_wei(client.balance_wei(), 'ether')} ETH")
    print(f"\n  BaseScan  : {EXPLORER}/address/{reg.address}")
    print(f"  tx        : {res.explorer_url()}")
    print(f"\n  pinned to .env: {pin_address(reg.address)}")
    print("  Also record this address in README.md and .env.example.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
