r"""Compile contracts/EvidenceRegistry.sol -> build/EvidenceRegistry.json.

Pinned compiler version and settings, because the deployed bytecode must be
reproducible for BaseScan source verification.
"""
from __future__ import annotations

import json
import pathlib

import solcx

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "contracts" / "EvidenceRegistry.sol"
OUT = ROOT / "build" / "EvidenceRegistry.json"

SOLC_VERSION = "0.8.24"
OPTIMIZER_RUNS = 200


def compile_registry() -> dict:
    if SOLC_VERSION not in [str(v) for v in solcx.get_installed_solc_versions()]:
        solcx.install_solc(SOLC_VERSION)

    std = {
        "language": "Solidity",
        "sources": {SRC.name: {"content": SRC.read_text(encoding="utf-8")}},
        "settings": {
            "optimizer": {"enabled": True, "runs": OPTIMIZER_RUNS},
            "evmVersion": "cancun",
            "outputSelection": {"*": {"*": ["abi", "evm.bytecode.object", "metadata"]}},
        },
    }
    res = solcx.compile_standard(std, solc_version=SOLC_VERSION)
    c = res["contracts"][SRC.name]["EvidenceRegistry"]
    art = {
        "contractName": "EvidenceRegistry",
        "solcVersion": SOLC_VERSION,
        "optimizerRuns": OPTIMIZER_RUNS,
        "evmVersion": "cancun",
        "abi": c["abi"],
        "bytecode": "0x" + c["evm"]["bytecode"]["object"],
        "sourcePath": str(SRC.relative_to(ROOT)).replace("\\", "/"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(art, indent=2), encoding="utf-8")
    return art


if __name__ == "__main__":
    a = compile_registry()
    print(f"  solc        : {a['solcVersion']} (optimizer {a['optimizerRuns']} runs, {a['evmVersion']})")
    print(f"  bytecode    : {len(a['bytecode']) // 2 - 1} bytes")
    print(f"  abi entries : {len(a['abi'])}")
    print(f"  artifact    : {OUT.relative_to(ROOT)}")
