r"""Phase 0 / Gate 0 spike: can we sign and broadcast a real Base Sepolia
transaction from the funded wallet, and see it on BaseScan?

Sends a 0-value self-transaction. Costs only gas (a few gwei of test ETH).
The private key is read from .env and NEVER printed.

    .venv\Scripts\python.exe spikes/spike_chain.py
"""
import sys
from web3 import Web3
from eth_account import Account
from _spike_common import check_interpreter, load_env, require

CHAIN_ID = 84532
EXPLORER = "https://sepolia.basescan.org"


def main() -> None:
    check_interpreter()
    env = load_env()
    (pk,) = require(env, "DEPLOYER_PRIVATE_KEY")
    rpc = env.get("BASE_SEPOLIA_RPC") or "https://sepolia.base.org"

    if not pk.startswith("0x"):
        pk = "0x" + pk
    acct = Account.from_key(pk)              # key stays in memory only
    print(f"\n  address : {acct.address}")
    print(f"  rpc     : {rpc}")

    w3 = Web3(Web3.HTTPProvider(rpc, request_kwargs={"timeout": 30}))
    if not w3.is_connected():
        print(f"\n  RPC UNREACHABLE: {rpc}\n"); sys.exit(1)

    cid = w3.eth.chain_id
    print(f"  chainId : {cid} ({'Base Sepolia OK' if cid == CHAIN_ID else 'UNEXPECTED'})")
    if cid != CHAIN_ID:
        print(f"\n  Expected {CHAIN_ID}. Check BASE_SEPOLIA_RPC.\n"); sys.exit(1)

    bal = w3.eth.get_balance(acct.address)
    print(f"  balance : {w3.from_wei(bal, 'ether')} ETH  block={w3.eth.block_number}")
    if bal == 0:
        print("\n  WALLET UNFUNDED. Fund it at:")
        print("    https://portal.cdp.coinbase.com/products/faucet")
        print(f"    address: {acct.address}\n")
        sys.exit(1)

    # EIP-1559 type-2 transaction. Base is an OP-stack chain; base fee is tiny.
    latest = w3.eth.get_block("latest")
    base_fee = latest.get("baseFeePerGas", 0)
    tip = w3.to_wei(1, "gwei")
    tx = {
        "from": acct.address,
        "to": acct.address,          # self-send: proves signing + broadcast
        "value": 0,
        "nonce": w3.eth.get_transaction_count(acct.address),
        "chainId": CHAIN_ID,
        "type": 2,
        "maxPriorityFeePerGas": tip,
        "maxFeePerGas": base_fee * 2 + tip,
        "gas": 21_000,
    }
    print(f"\n  sending 0-value self-tx  nonce={tx['nonce']} "
          f"maxFee={w3.from_wei(tx['maxFeePerGas'],'gwei')} gwei ...")

    try:
        signed = w3.eth.account.sign_transaction(tx, pk)
        # web3 v7+ renamed rawTransaction -> raw_transaction
        raw = getattr(signed, "raw_transaction", None) or signed.rawTransaction
        txh = w3.eth.send_raw_transaction(raw)
    except Exception as e:
        print(f"\n  BROADCAST FAILED: {type(e).__name__}: {e}\n"); sys.exit(1)

    print(f"  tx hash : {txh.hex()}")
    print(f"  waiting for receipt ...")
    try:
        rcpt = w3.eth.wait_for_transaction_receipt(txh, timeout=180)
    except Exception as e:
        print(f"\n  NO RECEIPT within timeout: {e}\n"); sys.exit(1)

    ok = rcpt["status"] == 1
    print(f"  block   : {rcpt['blockNumber']}  gasUsed={rcpt['gasUsed']}  "
          f"status={'SUCCESS' if ok else 'REVERTED'}")
    h = txh.hex()
    if not h.startswith("0x"):
        h = "0x" + h
    print(f"\n  BaseScan: {EXPLORER}/tx/{h}")
    print(f"\n  Gate 0 (chain leg): {'PASS' if ok else 'FAIL'}\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
