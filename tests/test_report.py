"""The visual result page: input face, retrieved image, and where it came from.

Built from committed fixtures in a temporary run directory, because runs/ is
gitignored. The page is shown to the people a demonstration is for, so it must
embed its images (no network), survive any page title a search engine returns,
and never let that title inject markup.
"""
import hashlib
import json
import pathlib
import shutil

from app.report import write_report

ROOT = pathlib.Path(__file__).resolve().parent.parent
FIX = ROOT / "data" / "fixtures"
TITLE = 'Мошенник <script>alert(1)</script> & "quotes"'


def _run(runs: pathlib.Path, run_id="20260101T000000Z_rep001") -> pathlib.Path:
    d = runs / run_id
    for sub in ("input", "candidates", "raw"):
        (d / sub).mkdir(parents=True)
    shutil.copyfile(ROOT / "data" / "input.jpg", d / "input" / "input.jpg")
    matched = FIX / "B1_nadella_2017.jpg"
    other = FIX / "A2_pichai_warsaw2022.jpg"
    msha = hashlib.sha256(matched.read_bytes()).hexdigest()
    osha = hashlib.sha256(other.read_bytes()).hexdigest()
    shutil.copyfile(matched, d / "candidates" / f"cand03_{msha[:12]}.jpg")
    shutil.copyfile(other, d / "candidates" / f"cand07_{osha[:12]}.jpg")
    (d / "run.json").write_text(json.dumps({"run_id": run_id, "input_name": "input.jpg"}), encoding="utf-8")
    (d / "search.json").write_text(json.dumps({"candidates": [
        {"position": 3, "source": "Example", "page_url": "https://example.org/page?a=1&b=2",
         "image_url": "https://img.example.org/photo.jpg"},
        {"position": 7, "source": "Other", "page_url": "https://other.example.org/",
         "image_url": "https://other.example.org/x.jpg"}]}), encoding="utf-8")
    evidence = {
        "input_image_sha256": "aa" * 32, "matched_image_sha256": msha,
        "matched_face_bbox": [40, 40, 200, 220], "faces_in_candidate": 1,
        "face_similarity_bp": 7208, "face_similarity_lo_bp": 7100, "face_similarity_hi_bp": 7300,
        "tta_views": 6, "threshold_bp": 2149, "phash_hamming_distance": 26,
        "verdict": "DISTINCT_PHOTO", "evidence_strength": "strong",
        "source_url": "https://example.org/page?a=1&b=2", "source_domain": "example.org",
        "page_title": TITLE, "retrieved_at": "2026-01-01T00:00:00Z",
        "search_provider": "serpapi_google_lens"}
    bundle = {"run_id": run_id, "evidence": evidence, "evidence_sha256": "bb" * 32,
              "chain": {"tx_hash": "0x" + "cc" * 32, "explorer": "https://sepolia.basescan.org/tx/0x" + "cc" * 32,
                        "contract": "0x38157D4652304BA251edCcffa435A0bC3F55305a"},
              "runner_up_scores": [{"position": 7, "source": "Other", "page_url": "https://other.example.org/",
                                    "sha256": osha, "similarity": 0.1, "verdict": "NO_MATCH",
                                    "matched_face_bbox": [10, 10, 60, 60]}],
              "notes": "score spread: 0.1"}
    (d / "bundle.json").write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return d


def test_page_shows_input_retrieved_image_and_source_links(tmp_path):
    out = write_report(_run(tmp_path / "runs"))
    page = out.read_text(encoding="utf-8")
    assert out.name == "result.html"
    assert page.count('src="data:image/jpeg;base64,') >= 3, "input, retrieved and candidate images must be embedded"
    assert 'href="https://img.example.org/photo.jpg"' in page, "the retrieved image's own link"
    assert 'href="https://example.org/page?a=1&amp;b=2"' in page, "the source page link, correctly escaped"
    assert "https://sepolia.basescan.org/tx/0x" in page


def test_page_loads_nothing_from_the_network(tmp_path):
    page = write_report(_run(tmp_path / "runs")).read_text(encoding="utf-8")
    assert 'src="http' not in page and "<link" not in page and "@import" not in page


def test_a_hostile_page_title_is_shown_as_text_not_markup(tmp_path):
    page = write_report(_run(tmp_path / "runs")).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in page
    assert "Мошенник &lt;script&gt;alert(1)&lt;/script&gt; &amp; &quot;quotes&quot;" in page


def test_a_replay_output_draws_its_images_from_the_capture(tmp_path):
    runs = tmp_path / "runs"
    src = _run(runs)
    replay = runs / "20260101T000100Z_rep002"
    replay.mkdir()
    b = json.loads((src / "bundle.json").read_text(encoding="utf-8"))
    b["run_id"] = replay.name
    b["notes"] = f"score spread: 0.1; replay of {src.name}, not notarised"
    b.pop("chain")
    (replay / "bundle.json").write_text(json.dumps(b, ensure_ascii=False), encoding="utf-8")
    page = write_report(replay).read_text(encoding="utf-8")
    assert page.count('src="data:image/jpeg;base64,') >= 2
    assert "replay of a captured run" in page
    assert "not recorded in this run" in page


def test_show_result_script_writes_without_opening_a_browser(tmp_path):
    from scripts.show_result import main
    d = _run(tmp_path / "runs")
    assert main(["--run", str(d), "--no-open"]) == 0
    assert (d / "result.html").exists()
    assert main(["--run", str(tmp_path / "missing"), "--no-open"]) == 2
