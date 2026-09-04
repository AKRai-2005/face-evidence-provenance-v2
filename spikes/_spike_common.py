"""Shared helpers for Phase 0 spikes. Deliberately dependency-light."""
import os, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "spikes" / "out"
OUT.mkdir(parents=True, exist_ok=True)

UA = "HHGoa2026-Task3-Research/0.1 (ashutoshkumarrai19o7@gmail.com) python-requests"

# A stable, publicly reachable image of a public figure. Used to validate
# provider plumbing without needing the image-hosting step to work first.
PUBLIC_TEST_IMAGE = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/1/19/"
    "Satya_smiling-print.jpg/1280px-Satya_smiling-print.jpg"
)


def load_env() -> dict:
    """Minimal .env reader -- no dependency on pydantic for a throwaway spike."""
    env = {}
    p = ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    for k, v in os.environ.items():
        if k in ("SERPAPI_KEY", "BRIGHTDATA_API_TOKEN", "BRIGHTDATA_SERP_ZONE") and v:
            env.setdefault(k, v)
    return env


def require(env: dict, *keys: str) -> list:
    missing = [k for k in keys if not env.get(k)]
    if missing:
        print(f"\n  MISSING CONFIG: {', '.join(missing)}")
        print(f"  Add them to {ROOT / '.env'} (copy .env.example if you have not).")
        print("  This spike makes no network call without them.\n")
        sys.exit(2)
    return [env[k] for k in keys]


def check_interpreter() -> None:
    """Windows Store Python aliases shadow an activated venv; catch it early."""
    if "WindowsApps" in sys.executable:
        print("\n  WRONG INTERPRETER")
        print(f"  Running: {sys.executable}")
        print("  This is the Windows Store alias, not the venv. Project deps are")
        print("  NOT installed there. Use the venv interpreter explicitly:")
        print(r"      .venv\Scripts\python.exe spikes/<script>.py")
        print()
        sys.exit(3)
