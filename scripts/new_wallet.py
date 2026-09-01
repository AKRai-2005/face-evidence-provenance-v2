"""Generate a fresh THROWAWAY Base Sepolia wallet and print what to paste into .env.

Run this yourself; the private key is printed on your machine only and is never
sent anywhere. Use it only for testnet. Never fund it with real ETH.

    python scripts/new_wallet.py
"""
from eth_account import Account


def main() -> None:
    Account.enable_unaudited_hdwallet_features()
    acct = Account.create()
    print()
    print("  NEW THROWAWAY TESTNET WALLET")
    print("  " + "-" * 66)
    print(f"  address     : {acct.address}")
    print(f"  private key : {acct.key.hex()}")
    print("  " + "-" * 66)
    print("  1. Paste the private key into .env as DEPLOYER_PRIVATE_KEY=")
    print("  2. Fund the ADDRESS at https://portal.cdp.coinbase.com/products/faucet")
    print("     (Base Sepolia, 0.1 test ETH / 24h, no mainnet balance needed)")
    print("  3. Confirm arrival: https://sepolia.basescan.org/address/" + acct.address)
    print()
    print("  TESTNET ONLY. This key is unencrypted. Do not reuse it anywhere.")
    print()


if __name__ == "__main__":
    main()
