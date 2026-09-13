r"""Open the visual result for a run: input face, retrieved image, source links.

    .venv\Scripts\python.exe scripts\show_result.py --run RUN_ID

Every new run writes runs\RUN_ID\result.html automatically. This (re)builds it
from the run's recorded artifacts -- so it also works for runs made before the
page existed -- and opens it in the default browser.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import webbrowser

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    from app.report import write_report

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", required=True, help="run id, or a path to a run directory")
    ap.add_argument("--no-open", action="store_true", help="write the page without opening a browser")
    args = ap.parse_args(argv)

    run_dir = pathlib.Path(args.run)
    if not run_dir.is_dir():
        run_dir = ROOT / "runs" / args.run
    if not (run_dir / "bundle.json").exists():
        print(f"\n  NO BUNDLE IN {run_dir}")
        print("  Use the run id from the summary panel, without angle brackets.\n")
        return 2

    out = write_report(run_dir)
    print(f"\n  visual result: {out}\n")
    if not args.no_open:
        webbrowser.open(out.resolve().as_uri())
    return 0


if __name__ == "__main__":
    sys.exit(main())
