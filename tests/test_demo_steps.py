"""DEMO_STEPS.json is followed literally, in front of judges.

A walkthrough that names a script that moved, a flag that was renamed, or a
fallback run that no longer matches what is committed fails in the worst
place to find out. It also deliberately contains no expected outputs -- it is
visible to the people being shown the demo -- and that is a property worth
keeping true as it is edited.
"""
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent
DOC_PATH = ROOT / "DEMO_STEPS.json"


@pytest.fixture(scope="module")
def doc():
    return json.loads(DOC_PATH.read_text(encoding="utf-8"))


def _commands(node):
    if isinstance(node, dict):
        if "run" in node:
            yield node["run"]
        for v in node.values():
            yield from _commands(v)
    elif isinstance(node, list):
        for v in node:
            yield from _commands(v)


def test_file_has_no_control_characters():
    raw = DOC_PATH.read_bytes()
    bad = [hex(c) for c in range(32) if c not in (9, 10, 13) and bytes([c]) in raw]
    assert not bad, f"control characters (a Windows path escaped wrongly?): {bad}"


def test_steps_are_numbered_in_order(doc):
    assert [s["step"] for s in doc["steps"]] == list(range(1, len(doc["steps"]) + 1))


def test_every_repository_file_a_command_names_exists(doc):
    """Paths under runs\\RUN_ID, tampered\\ and fresh\\ are created during the
    demo; everything else must already be in the repository."""
    missing = []
    for cmd in _commands(doc):
        for token in re.findall(r"[\w.\\/-]+\.(?:py|jpg|json)\b", cmd):
            norm = token.replace("\\", "/")
            if norm.startswith((".venv/", "fresh/", "runs/", "tampered/")):
                continue
            if not (ROOT / norm).exists():
                missing.append((token, cmd[:60]))
    assert not missing, f"commands name files that do not exist: {missing}"


def _source_for(cmd: str):
    m = re.search(r"-m app\.main\b", cmd)
    if m:
        return ROOT / "app" / "main.py"
    m = re.search(r"(scripts[\\/][\w]+\.py|\bverify\.py)", cmd)
    return ROOT / m.group(1).replace("\\", "/") if m else None


def test_every_flag_is_still_accepted_by_the_program_it_is_passed_to(doc):
    stale = []
    for cmd in _commands(doc):
        src = _source_for(cmd)
        if src is None or not src.exists():
            continue  # a missing program is reported by the file-existence test
        # only the flags after the program name belong to it
        tail = cmd.split("app.main", 1)[-1] if "app.main" in cmd else cmd.split(src.name, 1)[-1]
        text = src.read_text(encoding="utf-8")
        for flag in re.findall(r"(?<![\w-])--[a-z][a-z-]*", tail):
            if f'"{flag}"' not in text:
                stale.append((flag, src.name))
    assert not stale, f"flags no longer defined by their program: {stale}"


def test_placeholders_used_in_commands_are_defined(doc):
    defined = set(doc["conventions"]["placeholders"])
    used = {p for cmd in _commands(doc) for p in re.findall(r"\b(RUN_ID|CHAIN_TX)\b", cmd)}
    assert used <= defined, f"undefined placeholders: {used - defined}"


def test_fallback_is_the_committed_sample_run(doc):
    """The fallback must be a run whose bundle is in the repository, so what it
    points to can be checked by anyone -- not only on the demo machine."""
    committed = json.loads((ROOT / "sample_run_2" / "bundle.json").read_text(encoding="utf-8-sig"))
    fb = json.dumps(doc["fallback"])
    assert committed["run_id"] in fb, "fallback run id is not sample_run_2's run"
    assert committed["chain"]["tx_hash"] in fb, "fallback chain tx is not sample_run_2's transaction"


def test_contains_no_expected_outputs(doc):
    """Visible to judges: it says what to run and what to show, never what the
    result will be."""
    text = DOC_PATH.read_text(encoding="utf-8")
    verdicts = ["VERIFIED", "TAMPER DETECTED", "NOT REGISTERED", "PROBES LOOK INVERTED",
                "Evidence already recorded", "NO CANDIDATE ABOVE THRESHOLD", "PASS --"]
    assert not [v for v in verdicts if v in text], "a verdict string appears in the walkthrough"
    assert not re.search(r"(?<![\w.])[01]\.\d{4}\b", text), "a similarity score appears in the walkthrough"

    def keys(node):
        if isinstance(node, dict):
            for k, v in node.items():
                yield k
                yield from keys(v)
        elif isinstance(node, list):
            for v in node:
                yield from keys(v)
    assert not [k for k in keys(doc) if "expect" in k.lower() or k.lower() == "output"]
