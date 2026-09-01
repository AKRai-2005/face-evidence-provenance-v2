"""Structured [STAGE] logging with a hard secret-redaction filter.

Two things this must get right:
  1. No secret ever reaches a log line, a run.log, or the terminal. The filter
     is applied to the message AND its args, because '%s' formatting would
     otherwise smuggle a key past a naive message-only filter.
  2. Windows consoles default to cp1252 and raise UnicodeEncodeError on the
     first non-ASCII character (hit for real in Phase 0 on a Polish filename).
"""
from __future__ import annotations

import logging
import pathlib
import sys

STAGES = ("FACE", "HOST", "SEARCH", "FETCH", "VERIFY", "EVIDENCE", "CHAIN", "RUN")


class RedactSecrets(logging.Filter):
    """Replace any known secret substring with a stable placeholder."""

    def __init__(self, secrets: list[str]) -> None:
        super().__init__()
        # longest first, so a key containing another is masked whole
        self._secrets = sorted({s for s in secrets if s and len(s) >= 8}, key=len, reverse=True)

    def _scrub(self, text: str) -> str:
        for s in self._secrets:
            if s in text:
                text = text.replace(s, f"<redacted:{len(s)}chars>")
        return text

    def _scrub_arg(self, value):
        """Scrub strings only.

        Coercing every arg to str would break %d / %.3f formatting in the
        caller's message and raise inside logging itself -- which surfaces as a
        traceback, the one thing the failure-handling contract forbids. Non-str
        args cannot carry a secret substring anyway; a secret is always a str.
        """
        return self._scrub(value) if isinstance(value, str) else value

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            record.msg = self._scrub(record.msg)
        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: self._scrub_arg(v) for k, v in record.args.items()}
            else:
                record.args = tuple(self._scrub_arg(a) for a in record.args)
        return True


class StageFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        stage = getattr(record, "stage", None)
        prefix = f"[{stage}] " if stage else ""
        base = super().format(record)
        return f"{prefix}{base}"


def setup_logging(
    *, run_dir: pathlib.Path | None = None, secrets: list[str] | None = None,
    verbose: bool = False,
) -> logging.Logger:
    # Force UTF-8 on the console; cp1252 will otherwise crash on non-ASCII.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    logger = logging.getLogger("task3")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    redactor = RedactSecrets(secrets or [])

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(StageFormatter("%(message)s"))
    ch.addFilter(redactor)
    logger.addHandler(ch)

    if run_dir is not None:
        run_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(StageFormatter("%(asctime)s %(levelname)-7s %(message)s"))
        fh.addFilter(redactor)
        logger.addHandler(fh)

    return logger


def stage(logger: logging.Logger, name: str):
    """Return a bound logging callable for one pipeline stage."""
    if name not in STAGES:
        raise ValueError(f"unknown stage {name!r}; expected one of {STAGES}")

    def log(msg: str, *args, level: int = logging.INFO) -> None:
        logger.log(level, msg, *args, extra={"stage": name})

    return log
