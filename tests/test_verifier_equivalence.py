"""The verifier must stay byte-identical to the pipeline, and stay standalone.

`verify.py` deliberately re-implements canonicalization instead of importing it,
so that a bug in the pipeline's canonicalizer cannot hide inside the tool used
to check it. That independence is only worth something if two properties hold,
and neither is guaranteed by construction:

  1. the two implementations agree on *every* input, not on four hand-picked ones
  2. verify.py keeps importing nothing from this project

A judge runs verify.py against a bundle on a machine that has web3 and nothing
else. If (1) breaks, valid evidence fails to verify. If (2) breaks, the tool
stops being independent and the separation becomes theatre.
"""
import ast
import pathlib
import random
import sys

import pytest

from app.evidence.canonicalizer import CanonicalizationError, canonical_bytes
from app.evidence.hasher import evidence_hash

import verify as v

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Strings chosen to stress the rules the two implementations must share: NFC
# normalisation, no ASCII escaping, sorted keys, and UTF-8 output.
TRICKY = [
    "", "a", "Z", "0",
    "cafe\u0301",           # e + combining acute -> normalises
    "caf\u00e9",            # precomposed, the NFC form of the above
    "spotka\u0142 si\u0119",
    "\u00c5",               # angstrom-adjacent
    "A\u030a",              # A + combining ring -> same NFC as above
    "\u4e2d\u6587",
    "\U0001f600",           # astral plane
    "tab	here", 'quote"here', "back" + chr(92) + "slash",
    "  leading and trailing  ",
    "\u0130stanbul",        # dotted capital I
]


def _rand_scalar(rng):
    pick = rng.random()
    if pick < 0.35:
        return rng.choice(TRICKY)
    if pick < 0.60:
        return rng.randint(-2**40, 2**40)
    if pick < 0.75:
        return rng.choice([True, False])
    if pick < 0.85:
        return None
    return rng.choice(["schema", "source_url", "face_similarity_bp", "retrieved_at"])


def _rand_key(rng):
    # Bias toward realistic field names, but keep the normalising strings in
    # play -- they are what makes key handling non-trivial.
    if rng.random() < 0.5:
        return rng.choice(["schema", "a", "b", "z", "source_url", "n", "nested"])
    return rng.choice(TRICKY) or "empty"


def _rand_obj(rng, depth=0):
    if depth >= 3 or rng.random() < 0.3:
        return _rand_scalar(rng)
    if rng.random() < 0.35:
        return [_rand_obj(rng, depth + 1) for _ in range(rng.randint(0, 4))]
    out = {}
    for _ in range(rng.randint(0, 5)):
        out[_rand_key(rng)] = _rand_obj(rng, depth + 1)
    return out


def test_the_two_implementations_agree_on_thousands_of_generated_objects():
    """Seeded, so a failure is reproducible rather than a flake."""
    rng = random.Random(20260904)
    compared = 0
    for _ in range(3000):
        obj = _rand_obj(rng)
        if not isinstance(obj, dict):
            continue
        try:
            expected = canonical_bytes(obj)
        except CanonicalizationError:
            # The pipeline refuses it; the verifier must refuse it too, rather
            # than accepting something the pipeline would never have produced.
            with pytest.raises(ValueError):
                v.canonicalize(obj)
            compared += 1
            continue
        assert v.canonicalize(obj) == expected, f"divergence on {obj!r}"
        assert v.compute_hash(obj) == evidence_hash(obj)
        compared += 1
    assert compared > 500, f"generator produced too few dict cases ({compared})"


# --- the NFC key-collision defect this file was written to pin down ---------
@pytest.mark.parametrize("impl", [canonical_bytes, v.canonicalize])
def test_keys_that_normalise_onto_each_other_are_refused_not_silently_dropped(impl):
    """Two distinct keys can share one NFC form. Assigning both into a dict
    keeps whichever came last and discards the other silently, so two objects
    with different content would notarise to the same digest. The canonicalizer
    rejects floats rather than rounding them; it must reject this for the same
    reason."""
    with pytest.raises((CanonicalizationError, ValueError)):
        impl({"cafe\u0301": 1, "caf\u00e9": 2})


def test_a_single_normalising_key_is_still_accepted():
    """Only the collision is an error. Normalisation itself must keep working,
    or every non-ASCII field name would break."""
    assert canonical_bytes({"cafe\u0301": 1}) == '{"caf\u00e9":1}'.encode("utf-8")
    assert v.canonicalize({"cafe\u0301": 1}) == canonical_bytes({"cafe\u0301": 1})


def test_collision_is_detected_at_any_depth():
    nested = {"outer": {"inner": {"cafe\u0301": 1, "caf\u00e9": 2}}}
    with pytest.raises(CanonicalizationError):
        canonical_bytes(nested)
    with pytest.raises(ValueError):
        v.canonicalize(nested)


# --- the standalone guarantee ----------------------------------------------
def _imported_roots(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                roots.add(node.module.split(".")[0])
            elif node.level:
                roots.add(".")           # any relative import at all
    return roots


def test_verifier_imports_nothing_from_this_project():
    """The README tells a judge to run verify.py with web3 and the standard
    library alone. An `from app...` import added later would still pass every
    other test in this suite while quietly breaking that instruction."""
    roots = _imported_roots(ROOT / "verify.py")
    forbidden = {r for r in roots if r in {"app", "scripts", "tests", "."}}
    assert not forbidden, f"verify.py must not import project code, found: {forbidden}"


def test_verifier_third_party_dependency_is_only_web3():
    roots = _imported_roots(ROOT / "verify.py")
    third_party = {r for r in roots if r not in sys.stdlib_module_names and r != "."}
    assert third_party == {"web3"}, (
        f"verify.py may depend on web3 only; found {third_party}. Anything else "
        "must also be installed by whoever verifies a bundle."
    )
