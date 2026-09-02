r"""Build a DERIVED input artifact whose bytes exist nowhere on the web.

    .venv\Scripts\python.exe scripts/make_derived_input.py --source <img> --out data/input.jpg

This is the premise of the whole project. If the input file were byte-identical
to something already published, a match could be explained by exact-file lookup
and the face recognition would be decorative. Cropping, rescaling and
re-encoding guarantees a different SHA-256 while preserving the face.

It prints the proof so the claim can be checked rather than taken on trust.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, type=pathlib.Path)
    ap.add_argument("--out", default=ROOT / "data" / "input.jpg", type=pathlib.Path)
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--quality", type=int, default=82)
    ap.add_argument("--crop", type=float, default=0.08,
                    help="fraction trimmed from each edge")
    ap.add_argument("--mirror", action="store_true",
                    help="also mirror horizontally (defeats more matchers, but "
                         "note ArcFace is largely mirror-invariant)")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"\n  SOURCE NOT FOUND: {args.source}\n")
        return 2

    src_bytes = args.source.read_bytes()
    src_sha = hashlib.sha256(src_bytes).hexdigest()

    im = Image.open(io.BytesIO(src_bytes)).convert("RGB")
    w0, h0 = im.size

    c = max(0.0, min(0.45, args.crop))
    im = im.crop((int(w0 * c), int(h0 * c), int(w0 * (1 - c)), int(h0 * (1 - c))))
    if args.scale != 1.0:
        im = im.resize((max(1, int(im.width * args.scale)),
                        max(1, int(im.height * args.scale))), Image.LANCZOS)
    if args.mirror:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)

    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=args.quality, optimize=True)
    out_bytes = buf.getvalue()
    out_sha = hashlib.sha256(out_bytes).hexdigest()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(out_bytes)

    print()
    print("  DERIVED INPUT ARTIFACT")
    print("  " + "-" * 66)
    print(f"  source : {args.source}")
    print(f"           {w0}x{h0}  {len(src_bytes)/1024:.0f}KB  sha256 {src_sha}")
    print(f"  output : {args.out}")
    print(f"           {im.width}x{im.height}  {len(out_bytes)/1024:.0f}KB  sha256 {out_sha}")
    print("  " + "-" * 66)
    print(f"  transforms : crop {c:.0%}/edge, scale {args.scale}, "
          f"JPEG q{args.quality}{', mirrored' if args.mirror else ''}")
    print(f"  identical  : {'YES -- NOT DERIVED' if out_sha == src_sha else 'NO'}")
    print()
    if len(out_bytes) > 500 * 1024:
        print(f"  WARNING: {len(out_bytes)/1024:.0f}KB exceeds SerpApi's 500KB upload")
        print("  limit. Lower --scale or --quality.\n")
    print("  These bytes have never been published, so an exact-file or")
    print("  reverse-hash lookup cannot find them. Any match must come from")
    print("  the face itself.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
