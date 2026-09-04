"""Configuration validators.

These exist so a missing key produces an actionable message rather than a
confusing failure three stages later. That only holds if the messages actually
say what to do, so the tests assert on content, not just on the exception type.
"""
import pytest

from app.config import ConfigError, Settings


def _s(**over):
    """Every field is pinned explicitly.

    Settings reads .env, so any field left unset here would silently pull the
    developer's real credential into the test -- which is both non-hermetic and
    a way to print a live key into CI output. Found the hard way.
    """
    base = dict(SERPAPI_KEY="", GOOGLE_VISION_API_KEY="", BRIGHTDATA_API_TOKEN="",
                BRIGHTDATA_SERP_ZONE="", DEPLOYER_PRIVATE_KEY="",
                EVIDENCE_REGISTRY_ADDRESS="", SUBJECT_COMMITMENT_SALT="",
                BASESCAN_API_KEY="", BASE_SEPOLIA_RPC="https://example.invalid",
                IMAGE_HOST_BACKEND="serpapi_upload")
    base.update(over)
    return Settings(**base)


def test_no_provider_configured_names_every_option():
    with pytest.raises(ConfigError) as ei:
        _s().require_search()
    msg = str(ei.value)
    assert "SERPAPI_KEY" in msg and "GOOGLE_VISION_API_KEY" in msg
    assert "BRIGHTDATA" in msg


def test_any_single_provider_satisfies_the_search_requirement():
    _s(SERPAPI_KEY="k").require_search()
    _s(GOOGLE_VISION_API_KEY="k").require_search()
    _s(BRIGHTDATA_API_TOKEN="t", BRIGHTDATA_SERP_ZONE="z").require_search()


def test_brightdata_needs_both_token_and_zone():
    with pytest.raises(ConfigError):
        _s(BRIGHTDATA_API_TOKEN="t").require_search()
    with pytest.raises(ConfigError):
        _s(BRIGHTDATA_SERP_ZONE="z").require_search()


def test_missing_key_tells_you_how_to_make_one():
    with pytest.raises(ConfigError) as ei:
        _s().require_chain()
    assert "new_wallet.py" in str(ei.value)


def test_malformed_key_length_is_reported_with_the_actual_length():
    with pytest.raises(ConfigError) as ei:
        _s(DEPLOYER_PRIVATE_KEY="0xdeadbeef").require_chain()
    assert "malformed" in str(ei.value) and "66" in str(ei.value)


def test_private_key_gets_an_0x_prefix_if_omitted():
    assert _s(DEPLOYER_PRIVATE_KEY="ab" * 32).deployer_private_key.startswith("0x")


def test_key_already_prefixed_is_not_double_prefixed():
    k = "0x" + "ab" * 32
    assert _s(DEPLOYER_PRIVATE_KEY=k).deployer_private_key == k


def test_missing_contract_points_at_the_deploy_script():
    with pytest.raises(ConfigError) as ei:
        _s(DEPLOYER_PRIVATE_KEY="0x" + "ab" * 32).require_chain()
    assert "deploy.py" in str(ei.value)


def test_contract_not_required_when_only_deploying():
    _s(DEPLOYER_PRIVATE_KEY="0x" + "ab" * 32).require_chain(need_contract=False)


def test_short_salt_is_rejected_with_a_generator_command():
    with pytest.raises(ConfigError) as ei:
        _s(SUBJECT_COMMITMENT_SALT="abcd").require_salt()
    assert "token_hex" in str(ei.value)


def test_every_configured_secret_is_offered_for_redaction():
    """A secret the log filter never sees is a secret that can leak."""
    s = _s(SERPAPI_KEY="s" * 40, GOOGLE_VISION_API_KEY="g" * 39,
           BRIGHTDATA_API_TOKEN="b" * 36, DEPLOYER_PRIVATE_KEY="ab" * 32,
           SUBJECT_COMMITMENT_SALT="c" * 64, BASESCAN_API_KEY="d" * 34)
    vals = s.secret_values()
    assert len(vals) == 6
    assert all(len(v) >= 8 for v in vals)


def test_unset_secrets_are_not_offered_for_redaction():
    """Redacting the empty string would blank every log line."""
    assert _s(SERPAPI_KEY="k" * 40).secret_values() == ["k" * 40]
