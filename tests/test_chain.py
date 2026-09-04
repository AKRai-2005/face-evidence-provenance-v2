"""EvidenceRegistry behaviour, against an in-process EVM."""
import pytest

from app.blockchain.registry import AlreadyRecorded

H1 = "11" * 32
H2 = "22" * 32
SC = "ab" * 32
DOMAIN = "example.com"


def test_deploy_returns_address(registry):
    assert registry.address.startswith("0x") and len(registry.address) == 42


def test_unregistered_hash_does_not_exist(registry):
    rec = registry.get_evidence(H2)
    assert rec.exists is False
    assert rec.timestamp == 0


def test_record_then_read_back(registry):
    res = registry.record_evidence(H1, SC, DOMAIN)
    assert res.ok

    rec = registry.get_evidence(H1)
    assert rec.exists is True
    assert rec.timestamp > 0
    assert rec.subject_commitment == SC
    assert rec.source_domain == DOMAIN
    assert rec.recorder.lower() == registry.client.address.lower()


def test_duplicate_is_rejected_with_original_timestamp(registry):
    """First-seen semantics: the second attempt must fail, citing the first."""
    registry.record_evidence(H1, SC, DOMAIN)
    first = registry.get_evidence(H1).timestamp

    with pytest.raises(AlreadyRecorded) as ei:
        registry.record_evidence(H1, SC, DOMAIN)

    assert ei.value.first_seen == first
    assert "first seen" in str(ei.value)


def test_duplicate_rejected_even_with_different_metadata(registry):
    """The evidenceHash alone is the key -- changing the domain must not help."""
    registry.record_evidence(H1, SC, DOMAIN)
    with pytest.raises(AlreadyRecorded):
        registry.record_evidence(H1, "cd" * 32, "other.example")


def test_distinct_hashes_coexist(registry):
    registry.record_evidence(H1, SC, "a.example")
    registry.record_evidence(H2, SC, "b.example")
    assert registry.get_evidence(H1).source_domain == "a.example"
    assert registry.get_evidence(H2).source_domain == "b.example"


def test_event_emitted(registry):
    res = registry.record_evidence(H1, SC, DOMAIN)
    rcpt = registry.client.w3.eth.get_transaction_receipt(res.tx_hash)
    logs = registry.contract.events.EvidenceRecorded().process_receipt(rcpt)
    assert len(logs) == 1
    args = logs[0]["args"]
    assert args["evidenceHash"].hex().lstrip("0x") == H1
    assert args["sourceDomain"] == DOMAIN


def test_hash_accepts_0x_prefix(registry):
    registry.record_evidence("0x" + H1, "0x" + SC, DOMAIN)
    assert registry.get_evidence(H1).exists is True


def test_malformed_hash_rejected(registry):
    with pytest.raises(ValueError):
        registry.get_evidence("deadbeef")


def test_wait_for_record_polls_through_rpc_lag(registry, monkeypatch):
    """Regression: a load-balanced RPC can answer exists=False for a record we
    just wrote. Observed live on sepolia.base.org; an in-process EVM has instant
    consistency and cannot reproduce it, so the polling is tested with a stub."""
    from app.blockchain.registry import EvidenceRecord

    real = registry.get_evidence
    calls = {"n": 0}

    def lagging(h, **kw):
        calls["n"] += 1
        if calls["n"] < 3:          # first two reads land on a stale node
            return EvidenceRecord(False, 0, "0x" + "0" * 40, "00" * 32, "")
        return real(h, **kw)

    registry.record_evidence(H1, SC, DOMAIN)
    monkeypatch.setattr(registry, "get_evidence", lagging)

    rec = registry.wait_for_record(H1, timeout=10, interval=0.01)
    assert rec.exists is True
    assert calls["n"] >= 3


def test_wait_for_record_gives_up_and_reports_absent(registry):
    """A genuinely unregistered hash must still return (not hang forever)."""
    rec = registry.wait_for_record("ee" * 32, timeout=0.05, interval=0.01)
    assert rec.exists is False
