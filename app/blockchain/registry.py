"""Typed wrapper over the EvidenceRegistry contract."""
from __future__ import annotations

import dataclasses
import time

from .client import ChainClient, ChainError, TxResult, load_artifact


class AlreadyRecorded(ChainError):
    """The registry rejected a duplicate. This is first-seen semantics working."""

    def __init__(self, evidence_hash: str, first_seen: int):
        self.evidence_hash = evidence_hash
        self.first_seen = first_seen
        super().__init__(
            f"Evidence already recorded.\n"
            f"  evidenceHash : {evidence_hash}\n"
            f"  first seen   : unix {first_seen}\n"
            "  The registry timestamps FIRST sighting, so re-recording is refused."
        )


@dataclasses.dataclass(frozen=True)
class EvidenceRecord:
    exists: bool
    timestamp: int
    recorder: str
    subject_commitment: str
    source_domain: str


def _b32(h: str) -> bytes:
    """Accept a 64-char hex digest with or without 0x."""
    h = h[2:] if h.startswith("0x") else h
    if len(h) != 64:
        raise ValueError(f"expected a 32-byte hex hash, got {len(h)} chars")
    return bytes.fromhex(h)


class Registry:
    def __init__(self, client: ChainClient, address: str):
        art = load_artifact()
        self.client = client
        self.address = client.w3.to_checksum_address(address)
        self.contract = client.w3.eth.contract(address=self.address, abi=art["abi"])

    @classmethod
    def deploy(cls, client: ChainClient) -> tuple["Registry", TxResult]:
        art = load_artifact()
        c = client.w3.eth.contract(abi=art["abi"], bytecode=art["bytecode"])
        tx = c.constructor().build_transaction(
            {"from": client.address,
             "nonce": client.w3.eth.get_transaction_count(client.address),
             "chainId": client.chain_id, "type": 2, **client.fee_params()}
        )
        tx.pop("gasPrice", None)
        res = client.send(tx)
        if not res.ok:
            raise ChainError(f"Deployment reverted. See {res.explorer_url()}")
        rcpt = client.w3.eth.get_transaction_receipt(res.tx_hash)
        return cls(client, rcpt["contractAddress"]), res

    def record_evidence(self, evidence_hash: str, subject_commitment: str,
                        source_domain: str) -> TxResult:
        # Check state before simulating. Decoding a custom-error revert is not
        # portable: a real RPC returns hex in exc.data, while eth-tester embeds
        # the raw bytes in a string repr. Reading the record is one cheap
        # eth_call, behaves identically everywhere, and yields the original
        # timestamp directly -- which is exactly what the caller needs.
        existing = self.get_evidence(evidence_hash)
        if existing.exists:
            raise AlreadyRecorded(evidence_hash, existing.timestamp)

        # Simulate so any other revert surfaces before we spend gas.
        try:
            self.contract.functions.recordEvidence(
                _b32(evidence_hash), _b32(subject_commitment), source_domain
            ).call({"from": self.client.address})
        except Exception as e:
            first = self._decode_already_recorded(e)
            if first is not None:      # lost a race between the read and here
                raise AlreadyRecorded(evidence_hash, first) from None
            raise ChainError(f"recordEvidence would revert: {e}") from None

        tx = self.contract.functions.recordEvidence(
            _b32(evidence_hash), _b32(subject_commitment), source_domain
        ).build_transaction(
            {"from": self.client.address,
             "nonce": self.client.w3.eth.get_transaction_count(self.client.address),
             "chainId": self.client.chain_id, "type": 2, **self.client.fee_params()}
        )
        tx.pop("gasPrice", None)
        return self.client.send(tx)

    def _decode_already_recorded(self, exc: Exception) -> int | None:
        """Pull firstSeen out of an AlreadyRecorded(bytes32,uint64) revert.

        Revert data reaches us in several shapes depending on the backend:
        a real RPC gives a hex string on exc.data, eth-tester gives raw bytes,
        and some wrappers nest it under exc.data['data']. Normalise to hex.
        """
        sel = self.contract.w3.keccak(text="AlreadyRecorded(bytes32,uint64)")[:4].hex()
        sel = sel.removeprefix("0x")

        candidates: list[str] = []
        data = getattr(exc, "data", None)
        if isinstance(data, dict):
            data = data.get("data")
        if isinstance(data, (bytes, bytearray)):
            candidates.append(bytes(data).hex())
        elif isinstance(data, str):
            candidates.append(data.removeprefix("0x"))
        for a in exc.args:
            if isinstance(a, (bytes, bytearray)):
                candidates.append(bytes(a).hex())
        candidates.append(str(exc))

        for blob in candidates:
            i = blob.find(sel)
            if i == -1:
                continue
            payload = blob[i + 8:]
            if len(payload) < 128:
                continue
            try:
                return int(payload[64:128], 16)
            except ValueError:
                continue
        return None

    def wait_for_record(self, evidence_hash: str, *, timeout: float = 90.0,
                        interval: float = 2.0) -> EvidenceRecord:
        """Read a just-written record, tolerating RPC read-after-write lag.

        Public RPC endpoints are load-balanced. A getEvidence() issued straight
        after wait_for_transaction_receipt() can land on a node that has not yet
        applied that block and will answer exists=False for a record that
        demonstrably exists -- observed live on sepolia.base.org. Reporting
        "NOT REGISTERED" for evidence we just wrote would be a false negative in
        the most visible part of the pipeline, so we poll until it materialises.
        """
        deadline = time.monotonic() + timeout
        last = None
        while True:
            last = self.get_evidence(evidence_hash)
            if last.exists or time.monotonic() >= deadline:
                return last
            time.sleep(interval)

    def get_evidence(self, evidence_hash: str, *, block_identifier=None) -> EvidenceRecord:
        call = self.contract.functions.getEvidence(_b32(evidence_hash))
        exists, ts, rec, sc, dom = (
            call.call(block_identifier=block_identifier)
            if block_identifier is not None else call.call()
        )
        sc_hex = sc.hex() if isinstance(sc, (bytes, bytearray)) else str(sc)
        return EvidenceRecord(
            exists=bool(exists), timestamp=int(ts), recorder=str(rec),
            subject_commitment=sc_hex[2:] if sc_hex.startswith("0x") else sc_hex,
            source_domain=str(dom),
        )
