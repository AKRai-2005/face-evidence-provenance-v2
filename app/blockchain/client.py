"""Base Sepolia connection, fee strategy, and transaction submission.

Written against web3.py v8 (raw_transaction, not the v6 rawTransaction spelling
that most tutorials still show).
"""
from __future__ import annotations

import dataclasses
import json
import pathlib

from eth_account import Account
from web3 import Web3
from web3.exceptions import Web3Exception

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
ARTIFACT = ROOT / "build" / "EvidenceRegistry.json"

BASE_SEPOLIA_CHAIN_ID = 84532
EXPLORER = "https://sepolia.basescan.org"
FAUCET = "https://portal.cdp.coinbase.com/products/faucet"

# Base is an OP-stack L2: base fee sits around 0.005 gwei, ~200x below the
# 1 gwei tip that is idiomatic on Ethereum mainnet. Using a mainnet-sized tip
# here overpays by orders of magnitude -- measured: a 21k-gas empty self-transfer
# cost 0.0000211 ETH at a 1 gwei tip, which would make a deployment unaffordable
# on a small faucet grant. 0.01 gwei is still ~2x the base fee and lands fine.
DEFAULT_PRIORITY_FEE_GWEI = 0.01


class ChainError(RuntimeError):
    """Raised with an actionable message; never leaks a traceback to the user."""


class InsufficientFunds(ChainError):
    pass


@dataclasses.dataclass(frozen=True)
class TxResult:
    tx_hash: str
    block_number: int
    gas_used: int
    effective_gas_price: int
    status: int

    @property
    def ok(self) -> bool:
        return self.status == 1

    def explorer_url(self, base: str = EXPLORER) -> str:
        return f"{base}/tx/{self.tx_hash}"

    def cost_wei(self) -> int:
        return self.gas_used * self.effective_gas_price


def load_artifact() -> dict:
    if not ARTIFACT.exists():
        raise ChainError(
            f"Contract artifact missing at {ARTIFACT}.\n"
            "  Compile it first: python scripts/compile.py"
        )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


class ChainClient:
    def __init__(self, rpc: str, private_key: str, *, chain_id: int = BASE_SEPOLIA_CHAIN_ID,
                 priority_fee_gwei: float = DEFAULT_PRIORITY_FEE_GWEI, provider=None):
        self.w3 = Web3(provider) if provider is not None else Web3(
            Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30})
        )
        if not self.w3.is_connected():
            raise ChainError(
                f"RPC unreachable: {rpc}\n"
                "  Check BASE_SEPOLIA_RPC in .env, or try https://sepolia.base.org"
            )
        self.account = Account.from_key(private_key)
        self._key = private_key
        self.expected_chain_id = chain_id
        self.priority_fee = self.w3.to_wei(priority_fee_gwei, "gwei")

        actual = self.w3.eth.chain_id
        if chain_id is not None and actual != chain_id and provider is None:
            raise ChainError(
                f"Connected to chainId {actual}, expected {chain_id} (Base Sepolia).\n"
                "  Check BASE_SEPOLIA_RPC in .env."
            )
        self.chain_id = actual

    @property
    def address(self) -> str:
        return self.account.address

    def balance_wei(self) -> int:
        return self.w3.eth.get_balance(self.account.address)

    def fee_params(self) -> dict:
        base = self.w3.eth.get_block("latest").get("baseFeePerGas", 0)
        return {
            "maxPriorityFeePerGas": self.priority_fee,
            # 2x base covers a few blocks of base-fee growth; unused portion is
            # refunded under EIP-1559, so a generous ceiling costs nothing.
            "maxFeePerGas": base * 2 + self.priority_fee,
        }

    def _require_funds(self, gas: int, max_fee: int) -> None:
        need, have = gas * max_fee, self.balance_wei()
        if have < need:
            raise InsufficientFunds(
                f"Insufficient funds for gas.\n"
                f"  address : {self.account.address}\n"
                f"  balance : {self.w3.from_wei(have, 'ether')} ETH\n"
                f"  need    : up to {self.w3.from_wei(need, 'ether')} ETH "
                f"({gas} gas @ {self.w3.from_wei(max_fee, 'gwei')} gwei)\n"
                f"  Fund it : {FAUCET}  (Base Sepolia)"
            )

    def send(self, tx: dict, *, gas_buffer: float = 1.25) -> TxResult:
        """Fill fees/nonce/gas, sign, broadcast, wait. Raises ChainError on failure."""
        tx = dict(tx)
        tx.setdefault("from", self.account.address)
        tx.setdefault("chainId", self.chain_id)
        tx.setdefault("nonce", self.w3.eth.get_transaction_count(self.account.address))
        tx.setdefault("type", 2)
        tx.update({k: v for k, v in self.fee_params().items() if k not in tx})

        if "gas" not in tx:
            try:
                tx["gas"] = int(self.w3.eth.estimate_gas(tx) * gas_buffer)
            except Web3Exception as e:
                raise ChainError(f"Gas estimation failed: {e}") from None

        self._require_funds(tx["gas"], tx["maxFeePerGas"])

        try:
            signed = self.w3.eth.account.sign_transaction(tx, self._key)
            raw = getattr(signed, "raw_transaction", None)
            if raw is None:                      # web3 <7 spelling
                raw = signed.rawTransaction
            h = self.w3.eth.send_raw_transaction(raw)
        except Exception as e:
            raise ChainError(f"Transaction broadcast failed: {type(e).__name__}: {e}") from None

        try:
            r = self.w3.eth.wait_for_transaction_receipt(h, timeout=180)
        except Exception:
            raise ChainError(
                f"No receipt within 180s for {h.hex()}.\n  Check {EXPLORER}/tx/0x{h.hex().lstrip('0x')}"
            ) from None

        hx = h.hex()
        if not hx.startswith("0x"):
            hx = "0x" + hx
        return TxResult(
            tx_hash=hx,
            block_number=r["blockNumber"],
            gas_used=r["gasUsed"],
            effective_gas_price=r.get("effectiveGasPrice", tx["maxFeePerGas"]),
            status=r["status"],
        )
