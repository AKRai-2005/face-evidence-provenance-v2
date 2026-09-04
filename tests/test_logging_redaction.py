"""The redaction filter is a security control, so it gets real tests."""
import logging

from app.logging_setup import RedactSecrets

SECRET = "sk_live_" + "a" * 40


def _record(msg, args=()):
    return logging.LogRecord("t", logging.INFO, __file__, 1, msg, args, None)


def _rendered(msg, args=()):
    r = _record(msg, args)
    RedactSecrets([SECRET]).filter(r)
    return r.getMessage()


def test_redacts_secret_in_message():
    assert SECRET not in _rendered(f"key={SECRET}")
    assert "<redacted:48chars>" in _rendered(f"key={SECRET}")


def test_redacts_secret_passed_as_arg():
    assert SECRET not in _rendered("key=%s", (SECRET,))


def test_redacts_secret_in_dict_arg():
    # logging unwraps a mapping only when it arrives inside a tuple, which is
    # what Logger._log produces for logger.info("%(k)s", {...}).
    assert SECRET not in _rendered("key=%(k)s", ({"k": SECRET},))


def test_preserves_numeric_formatting():
    """Regression: coercing args to str broke %d/%.3f and raised inside logging."""
    assert _rendered("%d faces, det %.3f", (2, 0.9014)) == "2 faces, det 0.901"


def test_short_strings_are_not_treated_as_secrets():
    r = logging.LogRecord("t", logging.INFO, __file__, 1, "value=%s", ("abc",), None)
    RedactSecrets(["abc"]).filter(r)
    assert r.getMessage() == "value=abc"


def test_longest_secret_masked_first():
    inner, outer = "b" * 10, "b" * 10 + "TAIL12345"
    r = _record("%s", (outer,))
    RedactSecrets([inner, outer]).filter(r)
    assert "b" * 10 not in r.getMessage()
