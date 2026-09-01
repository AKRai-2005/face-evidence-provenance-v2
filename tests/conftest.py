import pytest
from eth_account import Account
from web3 import EthereumTesterProvider, Web3

from app.blockchain.client import ChainClient
from app.blockchain.registry import Registry


@pytest.fixture(scope="session")
def artifact():
    from scripts.compile import compile_registry
    return compile_registry()


@pytest.fixture()
def chain(artifact):
    """In-process EVM. Tests must not need a network or a funded wallet."""
    provider = EthereumTesterProvider()
    w3 = Web3(provider)
    funded = w3.eth.accounts[0]
    acct = Account.create()
    w3.eth.send_transaction(
        {"from": funded, "to": acct.address, "value": w3.to_wei(10, "ether")}
    )
    return ChainClient(
        rpc="", private_key=acct.key.hex(), chain_id=None, provider=provider
    )


@pytest.fixture()
def registry(chain):
    reg, _ = Registry.deploy(chain)
    return reg
