r"""Gate 1 entry point: prove the face stage works standalone.

    .venv\Scripts\python.exe -m app.face --image data/fixtures/B2_nadella_smiling.jpg
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import sys

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from ..config import get_settings
from ..logging_setup import setup_logging
from .detector import FaceEngine, FaceTooBlurry, FaceTooSmall, NoFaceDetected
from .encoder import subject_commitment, to_basis_points

EXIT_NO_FACE, EXIT_QUALITY, EXIT_BAD_INPUT = 10, 11, 12


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.face", description="Face detection + ArcFace embedding")
    ap.add_argument("--image", required=True, type=pathlib.Path)
    ap.add_argument("--all-faces", action="store_true", help="show every detected face")
    args = ap.parse_args(argv)

    cfg = get_settings()
    log = setup_logging(secrets=cfg.secret_values())
    con = Console()

    if not args.image.exists():
        con.print(f"[bold red]INPUT NOT FOUND[/] {args.image}")
        return EXIT_BAD_INPUT

    raw = args.image.read_bytes()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        con.print(f"[bold red]NOT A DECODABLE IMAGE[/] {args.image}")
        return EXIT_BAD_INPUT

    engine = FaceEngine.shared(logger=log)
    try:
        primary, faces = engine.detect_primary(
            img, min_face_px=cfg.min_face_px, min_det_score=cfg.min_det_score
        )
    except NoFaceDetected as e:
        con.print(f"[bold red]NO FACE DETECTED[/]\n{e}")
        return EXIT_NO_FACE
    except (FaceTooSmall, FaceTooBlurry) as e:
        con.print(f"[bold red]QUALITY GATE REJECTED INPUT[/]\n{e}")
        return EXIT_QUALITY

    h, w = img.shape[:2]
    con.print()
    con.print(f"  file    : {args.image}  ({w}x{h}, {len(raw)/1024:.0f} KB)")
    con.print(f"  sha256  : {hashlib.sha256(raw).hexdigest()}")
    con.print(f"  model   : {engine.model_id()}")
    con.print("  metric  : cosine_l2normed")

    t = Table(show_header=True, header_style="bold")
    for c in ("idx", "bbox", "det", "px", "sharpness", "face pHash", ""):
        t.add_column(c)
    for f in (faces if args.all_faces else [primary]):
        t.add_row(
            str(f.index), str(list(f.bbox)), f"{f.det_score:.4f}", str(f.face_px),
            f"{f.sharpness:.1f}", f.face_phash,
            "[green]<- selected[/]" if f.index == primary.index else "",
        )
    con.print(t)

    e = primary.embedding
    con.print(f"  embedding: dim={e.shape[0]} dtype={e.dtype} L2norm={np.linalg.norm(e):.6f}")
    con.print(f"  first 8  : [{', '.join(f'{v:+.4f}' for v in e[:8])}, ...]")
    con.print(f"  det conf : [bold]{primary.det_score:.4f}[/]  ({to_basis_points(primary.det_score)} bp)")

    if cfg.subject_commitment_salt:
        con.print(f"  subject commitment: {subject_commitment(e, cfg.subject_commitment_salt)}")
        con.print("    (salted SHA-256; no biometric data is published)")
    else:
        con.print("  [yellow]subject commitment: skipped, SUBJECT_COMMITMENT_SALT unset[/]")

    if len(faces) > 1 and not args.all_faces:
        con.print(f"  [yellow]note: {len(faces)} faces detected; --all-faces to see them[/]")
    con.print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
