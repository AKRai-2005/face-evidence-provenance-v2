"""A visual result page for one run: the input face, the image that was
retrieved with the matched face boxed, and exactly where it came from.

The terminal output records all of this, but only as text. Someone watching a
demonstration wants to SEE the retrieved photograph and follow its link, not
read a SHA-256. This renders what the run already recorded into a single
self-contained HTML file -- images embedded, no network access -- and changes
nothing the evidence hash covers.

It does not name anyone. The page title shown is the source page's own text.
"""
from __future__ import annotations

import base64
import html
import io
import json
import pathlib
import re

MAX_SIDE = 900          # embedded images are downscaled to keep the file small


def _read(p: pathlib.Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def _capture_dir(run_dir: pathlib.Path, bundle: dict) -> pathlib.Path:
    """A replay writes to its own directory; its images live in the capture."""
    m = re.search(r"replay of (\S+?),", bundle.get("notes") or "")
    if m and not (run_dir / "candidates").exists():
        src = run_dir.parent / m.group(1)
        if src.exists():
            return src
    return run_dir


def _find_candidate(capture: pathlib.Path, sha256: str | None) -> pathlib.Path | None:
    if not sha256:
        return None
    hits = sorted((capture / "candidates").glob(f"cand*_{sha256[:12]}*"))
    return hits[0] if hits else None


def _data_uri(path: pathlib.Path, box=None) -> str | None:
    try:
        from PIL import Image, ImageDraw
        im = Image.open(path).convert("RGB")
    except Exception:
        return None
    if box:
        d = ImageDraw.Draw(im)
        w = max(3, round(max(im.size) / 220))
        x1, y1, x2, y2 = [int(v) for v in box]
        d.rectangle((x1, y1, x2, y2), outline=(245, 158, 11), width=w)
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _link(url: str | None) -> str:
    if not url:
        return '<span class="muted">not recorded</span>'
    u = html.escape(url, quote=True)
    return f'<a href="{u}" target="_blank" rel="noopener noreferrer">{u}</a>'


def write_report(run_dir: pathlib.Path) -> pathlib.Path:
    """Write run_dir/result.html from the run's recorded artifacts; return its path."""
    run_dir = pathlib.Path(run_dir)
    bundle = _read(run_dir / "bundle.json")
    ev = bundle["evidence"]
    capture = _capture_dir(run_dir, bundle)
    e = html.escape

    # input image
    input_uri = None
    if (capture / "run.json").exists():
        meta = _read(capture / "run.json")
        p = capture / "input" / meta.get("input_name", "")
        if p.exists():
            input_uri = _data_uri(p)

    # matched image, with the matched face boxed, and its direct link
    cand_path = _find_candidate(capture, ev.get("matched_image_sha256"))
    matched_uri = _data_uri(cand_path, ev.get("matched_face_bbox")) if cand_path else None
    image_url = None
    if cand_path and (capture / "search.json").exists():
        pos = int(cand_path.name[4:6])
        for c in _read(capture / "search.json").get("candidates", []):
            if c.get("position") == pos:
                image_url = c.get("image_url")
                break

    cutoff = None
    calib = pathlib.Path(__file__).resolve().parent.parent / "calibration" / "results.json"
    if calib.exists():
        cutoff = _read(calib).get("same_photo_phash_max")

    sim = ev.get("face_similarity_bp", 0) / 1e4
    lo = ev.get("face_similarity_lo_bp", 0) / 1e4
    hi = ev.get("face_similarity_hi_bp", 0) / 1e4
    chain = bundle.get("chain") or {}
    tx = chain.get("tx_hash")
    replay_note = "replay of" in (bundle.get("notes") or "")

    def img_block(uri, caption):
        body = f'<img src="{uri}" alt="{e(caption)}">' if uri else '<div class="noimg">image not available in this run directory</div>'
        return f'<figure>{body}<figcaption>{e(caption)}</figcaption></figure>'

    others = []
    for r in bundle.get("runner_up_scores") or []:
        p = _find_candidate(capture, r.get("sha256"))
        uri = _data_uri(p, r.get("matched_face_bbox")) if p else None
        others.append(
            '<div class="other">'
            + (f'<img src="{uri}" alt="candidate {r.get("position")}">' if uri else '<div class="noimg small">no image</div>')
            + f'<div class="meta"><b>{e(str(r.get("source") or ""))}</b>'
            f'<span>cosine {float(r.get("similarity", 0)):.4f} · {e(str(r.get("verdict", "")))}</span>'
            f'<span class="links">{_link(r.get("page_url"))}</span></div></div>')

    rows = lambda pairs: "".join(f"<tr><th>{e(k)}</th><td>{v}</td></tr>" for k, v in pairs)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Result — {e(bundle.get("run_id", ""))}</title>
<style>
:root {{ --ink:#17161a; --muted:#6a6670; --rule:#e3e0e6; --ground:#f7f6f8; --paper:#fff; --accent:#b45309; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--ground); color:var(--ink); font:15px/1.5 "Segoe UI", system-ui, -apple-system, Arial, sans-serif; }}
main {{ max-width:1180px; margin:0 auto; padding:28px 22px 48px; }}
h1 {{ font-size:26px; margin:0 0 2px; letter-spacing:-.01em; }}
.sub {{ color:var(--muted); margin:0 0 22px; font-size:14px; }}
.banner {{ background:#fff7e6; border:1px solid #f3d19e; padding:8px 12px; border-radius:6px; margin:0 0 18px; font-size:14px; }}
.pair {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
figure {{ margin:0; background:var(--paper); border:1px solid var(--rule); border-radius:8px; padding:12px; }}
figure img {{ width:100%; height:auto; display:block; border-radius:4px; }}
figcaption {{ margin-top:8px; font-weight:600; font-size:14px; }}
.noimg {{ height:240px; display:grid; place-items:center; color:var(--muted); background:var(--ground); border-radius:4px; }}
.noimg.small {{ height:120px; font-size:13px; }}
section {{ background:var(--paper); border:1px solid var(--rule); border-radius:8px; padding:14px 16px; margin-top:18px; }}
h2 {{ font-size:15px; text-transform:uppercase; letter-spacing:.08em; color:var(--accent); margin:0 0 10px; }}
table {{ border-collapse:collapse; width:100%; }}
th {{ text-align:left; color:var(--muted); font-weight:500; padding:6px 14px 6px 0; width:190px; vertical-align:top; white-space:nowrap; }}
td {{ padding:6px 0; overflow-wrap:anywhere; font-variant-numeric:tabular-nums; }}
a {{ color:#1d4ed8; }}
code {{ font-family:Consolas, "Cascadia Mono", monospace; font-size:13px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fill, minmax(210px, 1fr)); gap:14px; }}
.other {{ border:1px solid var(--rule); border-radius:6px; padding:8px; }}
.other img {{ width:100%; height:150px; object-fit:contain; background:var(--ground); border-radius:4px; display:block; }}
.meta {{ display:flex; flex-direction:column; gap:2px; margin-top:6px; font-size:13px; }}
.meta .links {{ overflow-wrap:anywhere; }}
.muted {{ color:var(--muted); }}
.foot {{ color:var(--muted); font-size:13px; margin-top:18px; }}
@media (max-width:760px) {{ .pair {{ grid-template-columns:1fr; }} th {{ width:auto; }} }}
</style></head><body><main>
<h1>Evidence result</h1>
<p class="sub">Run <code>{e(bundle.get("run_id", ""))}</code> · retrieved {e(str(ev.get("retrieved_at", "")))} via {e(str(ev.get("search_provider", "")))}</p>
{'<p class="banner">This is a replay of a captured run. It was not written to the chain; verify the original run for the on-chain record.</p>' if replay_note else ''}
<div class="pair">
  {img_block(input_uri, "Input face (the derived image that was searched)")}
  {img_block(matched_uri, "Retrieved image — matched face boxed")}
</div>

<section><h2>Where it was retrieved from</h2><table>
{rows([
    ("Source image", _link(image_url)),
    ("Source page", _link(ev.get("source_url"))),
    ("Page title", e(str(ev.get("page_title") or "—"))),
    ("Domain", e(str(ev.get("source_domain") or "—"))),
    ("Downloaded copy", f"<code>{e(str(cand_path.relative_to(capture.parent.parent) if cand_path else '—'))}</code>"),
])}
</table></section>

<section><h2>The match</h2><table>
{rows([
    ("Face cosine similarity", f"{sim:.4f} &nbsp;<span class='muted'>range {lo:.4f} – {hi:.4f} over {e(str(ev.get('tta_views', '')))} augmented views</span>"),
    ("Calibrated threshold", f"{ev.get('threshold_bp', 0) / 1e4:.4f}"),
    ("Face-region pHash distance", f"{e(str(ev.get('phash_hamming_distance')))}" + (f" <span class='muted'>(same-photo cutoff {cutoff})</span>" if cutoff is not None else "")),
    ("Verdict", e(str(ev.get("verdict", "")))),
    ("Evidence strength", e(str(ev.get("evidence_strength", "")))),
    ("Faces in retrieved image", e(str(ev.get("faces_in_candidate", "")))),
    ("Input SHA-256", f"<code>{e(str(ev.get('input_image_sha256', '')))}</code>"),
    ("Retrieved image SHA-256", f"<code>{e(str(ev.get('matched_image_sha256', '')))}</code>"),
])}
</table></section>

<section><h2>On chain</h2><table>
{rows([
    ("Evidence hash", f"<code>{e(str(bundle.get('evidence_sha256', '')))}</code>"),
    ("Transaction", _link(chain.get("explorer")) if tx else '<span class="muted">not recorded in this run</span>'),
    ("Contract", f"<code>{e(str(chain.get('contract', '—')))}</code>"),
])}
</table></section>

{'<section><h2>Other candidates scored</h2><div class="grid">' + "".join(others) + '</div></section>' if others else ''}

<p class="foot">Generated from the run's recorded artifacts. This page does not identify anyone: the page title is the source page's own text, and the claim is that an image at this URL contains a face within a calibrated distance of the input's.</p>
</main></body></html>
"""
    out = run_dir / "result.html"
    out.write_text(doc, encoding="utf-8")
    return out
