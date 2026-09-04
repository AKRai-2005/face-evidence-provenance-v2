r"""Open ONE record's subject commitment, without weakening any other record.

    .venv\Scripts\python.exe scripts/prove_subject.py --bundle runs/<id>/bundle.json --image <photo>

The registry stores subjectCommitment = SHA-256(record_salt || quantised
embedding). Nothing biometric is on chain, so on its own the commitment says
nothing about who a record concerns. This is how that link is substantiated
after the fact:

    record_salt = HMAC-SHA256(master_salt, run_id)
    recompute   = SHA-256(record_salt || quantise(embedding_of(image)))
    compare     = recompute vs the commitment already on chain

If they match, the holder of the master salt has demonstrated that this record
concerns the person in that photograph -- and has revealed only this record's
salt, not the master.

WHY THAT MATTERS. Under the earlier single-salt design, substantiating one
record meant publishing the one salt that protects every record, so the act of
proving a claim compromised all the others. Per-record derivation makes opening
a record a local event.

The master salt is read from .env and is never printed.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    import cv2

    from app.config import ConfigError, get_settings
    from app.face.detector import FaceEngine
    from app.face.encoder import derive_record_salt, subject_commitment

    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True, type=pathlib.Path)
    ap.add_argument("--image", required=True, type=pathlib.Path,
                    help="a photograph of the subject you claim the record concerns")
    ap.add_argument("--reveal-record-salt", action="store_true",
                    help="print the derived per-record salt, for handing to a third "
                         "party so they can check this without the master")
    args = ap.parse_args()

    for p in (args.bundle, args.image):
        if not p.exists():
            print(f"\n  NOT FOUND: {p}\n")
            return 2

    cfg = get_settings()
    try:
        cfg.require_salt()
    except ConfigError as e:
        print(f"\nCONFIG ERROR\n{e}\n")
        return 2

    bundle = json.loads(args.bundle.read_text(encoding="utf-8-sig"))
    run_id = bundle.get("run_id", "")
    on_chain = (bundle.get("chain") or {}).get("subject_commitment", "")
    if not run_id or not on_chain:
        print("\n  BUNDLE HAS NO run_id / subject_commitment.")
        print("  Only bundles from a chain-recording run can be opened.\n")
        return 2

    img = cv2.imread(str(args.image))
    if img is None:
        print(f"\n  NOT A DECODABLE IMAGE: {args.image}\n")
        return 2

    engine = FaceEngine.shared()
    faces = engine.detect(img)
    if not faces:
        print(f"\n  NO FACE DETECTED IN {args.image.name}\n")
        return 2
    face = max(faces, key=lambda f: f.det_score)

    record_salt = derive_record_salt(cfg.subject_commitment_salt, run_id)
    recomputed = subject_commitment(face.embedding, record_salt)

    print()
    print("  SUBJECT COMMITMENT -- OPENING ONE RECORD")
    print("  " + "=" * 66)
    print(f"  run_id           : {run_id}")
    print(f"  image            : {args.image.name}  (det {face.det_score:.3f})")
    print("  record salt      : derived as HMAC-SHA256(master, run_id)")
    if args.reveal_record_salt:
        print(f"                     {record_salt}")
    print()
    print(f"  on chain         : {on_chain}")
    print(f"  recomputed       : {recomputed}")
    print()

    if recomputed == on_chain:
        print("  MATCH -- this record demonstrably concerns the person in that image.")
        print()
        print("  Only THIS record's salt was needed. The master salt was not")
        print("  revealed, and every other record remains exactly as protected as")
        print("  it was before.")
        print()
        return 0

    print("  NO MATCH -- this record does not concern the person in that image,")
    print("  or the image is too different from the one originally embedded.")
    print()
    print("  Note the asymmetry: a match is strong evidence, but a non-match is")
    print("  weak. The commitment is over a quantised embedding, so a photograph")
    print("  far from the original in pose or lighting can fail to reproduce it")
    print("  even for the right person.")
    print()
    return 1


if __name__ == "__main__":
    sys.exit(main())
