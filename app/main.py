r"""Orchestrator + CLI.

    .venv\Scripts\python.exe -m app.main --image data/input.jpg

Stages: FACE -> HOST -> SEARCH -> FETCH -> VERIFY -> EVIDENCE -> CHAIN.
Every run writes runs/<utc>_<short_id>/ with the input, raw provider responses,
downloaded candidates, evidence.json, canonical.json, bundle.json and run.log.
That directory is the audit trail, and it is what makes --replay legitimate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pathlib
import secrets
import sys
import time

import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from .config import ConfigError, get_settings
from .evidence.bundle import EvidenceInputs, build_bundle, build_evidence, write_artifacts
from .evidence.canonicalizer import domain_of, normalise_url
from .evidence.hasher import evidence_hash
from .face.detector import FaceEngine, FaceTooBlurry, FaceTooSmall, NoFaceDetected
from .face.encoder import subject_commitment
from .face.matcher import CalibrationMissing, Thresholds, Verdict
from .face.ranking import rank, score_candidate, score_spread
from .logging_setup import setup_logging, stage
from .search.base import (ProviderAuthError, ProviderError, ProviderRateLimited,
                          ProviderUnavailable)
from .search.candidate_extractor import CandidateFetcher, load_cached
from .search.image_host import ImageHostError, get_host
from .search.providers.brightdata_lens import BrightDataLens
from .search.providers.serpapi_lens import SerpApiLens

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CONSENT = ROOT / ".consent"

(EXIT_OK, EXIT_NO_FACE, EXIT_QUALITY, EXIT_BAD_INPUT, EXIT_CONFIG, EXIT_SEARCH,
 EXIT_NO_CANDIDATES, EXIT_NO_MATCH, EXIT_CHAIN, EXIT_CALIBRATION) = (
    0, 10, 11, 12, 13, 20, 21, 22, 30, 40)

CONSENT_TEXT = """\
This tool searches the public web for images containing a given face, and
notarises the result on a public blockchain.

  * Use it only on your own face, or on a public figure whose images are
    already broadly indexed. Never on a private individual.
  * A similarity score above the threshold is EVIDENCE, not proof of identity.
  * The blockchain records WHEN a claim was made. It does not validate it.

See ETHICS.md. Type 'yes' to acknowledge (recorded once in .consent): """


def consent_gate(con: Console, *, assume: bool = False) -> bool:
    if CONSENT.exists():
        return True
    if assume:
        CONSENT.write_text(
            f"acknowledged (--yes) {dt.datetime.now(dt.timezone.utc).isoformat()}\n",
            encoding="utf-8")
        return True
    if not sys.stdin.isatty():
        con.print("[red]No .consent file and stdin is not a terminal.[/]")
        con.print("Run once interactively, or pass --yes.")
        return False
    try:
        answer = input(CONSENT_TEXT).strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if answer != "yes":
        con.print("[yellow]Not acknowledged; exiting.[/]")
        return False
    CONSENT.write_text(
        f"acknowledged {dt.datetime.now(dt.timezone.utc).isoformat()}\n", encoding="utf-8")
    return True


def new_run_dir(run_id: str | None = None) -> tuple[pathlib.Path, str]:
    if run_id is None:
        run_id = (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                  + "_" + secrets.token_hex(3))
    d = RUNS / run_id
    (d / "raw").mkdir(parents=True, exist_ok=True)
    (d / "candidates").mkdir(parents=True, exist_ok=True)
    return d, run_id


def build_providers(cfg, log):
    """Primary first, then fallbacks. Only configured providers are included."""
    out = []
    if cfg.serpapi_key:
        out.append(SerpApiLens(cfg.serpapi_key, logger=stage(log, "SEARCH")))
    if cfg.brightdata_api_token and cfg.brightdata_serp_zone:
        out.append(BrightDataLens(cfg.brightdata_api_token, cfg.brightdata_serp_zone,
                                  logger=stage(log, "SEARCH")))
    return out


def run_search(providers, image_bytes, raw_dir, cfg, log, hosted_url=None):
    """Try each provider in order. Never silently substitute -- log every fallback."""
    say = stage(log, "SEARCH")
    last = None
    for i, p in enumerate(providers):
        try:
            say("querying %s ...", p.name)
            kw = {}
            if isinstance(p, BrightDataLens):
                kw["public_image_url"] = hosted_url
            res = p.search(image_bytes, raw_dir=raw_dir,
                           max_candidates=cfg.max_candidates, **kw)
            say("%s returned %d candidates (%s)", p.name, len(res.candidates), res.notes)
            if res.candidates:
                return res
            say("%s FAILED (zero candidates)", p.name)
            last = ProviderUnavailable(f"{p.name} returned zero candidates")
        except ProviderRateLimited as e:
            say("%s FAILED (rate limit) %s", p.name, str(e).splitlines()[0])
            last = e
        except ProviderAuthError as e:
            say("%s FAILED (auth) %s", p.name, str(e).splitlines()[0])
            last = e
        except ProviderError as e:
            say("%s FAILED (%s) %s", p.name, type(e).__name__, str(e).splitlines()[0])
            last = e
        if i + 1 < len(providers):
            say("falling back to %s ...", providers[i + 1].name)
    raise last or ProviderUnavailable("no search provider is configured")


def distinct_artifact_panel(con, *, input_sha, input_dims, best, thresholds, verdict):
    identical = (best.sha256 == input_sha)
    lines = [
        f"  input image   sha256: {input_sha}",
        f"                        {input_dims[0]}x{input_dims[1]}, local",
        f"  matched image sha256: {best.sha256}",
        f"                        {best.image_size[0]}x{best.image_size[1]}, "
        f"from {domain_of(best.page_url) or best.source}",
        "",
        f"  identical files: {'YES' if identical else 'NO'}"
        f"{'' if identical else '          <- reverse-image-hash matching is ruled out'}",
        f"  face cosine similarity: {best.similarity:.4f}",
        f"  calibrated threshold:   {thresholds.similarity:.4f}  "
        f"(TAR {thresholds.tar:.3f} @ FAR {thresholds.far:g}, see calibration/)",
        f"  face-region pHash distance: {best.phash_distance}"
        f"   (same-photo cutoff {thresholds.same_photo_phash})",
        "",
        f"  verdict: {verdict.value}",
    ]
    colour = {"DISTINCT_PHOTO": "green", "SAME_PHOTO": "yellow",
              "EXACT_DUPLICATE": "red", "NO_MATCH": "red"}[verdict.name]
    con.print(Panel("\n".join(lines), title="[bold]DISTINCT-ARTIFACT PROOF[/]",
                    border_style=colour, expand=False))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="app.main",
                                 description="Face-derived evidence provenance pipeline")
    ap.add_argument("--image", type=pathlib.Path, help="input face image")
    ap.add_argument("--replay", metavar="RUN_ID", help="re-run against a captured run")
    ap.add_argument("--no-chain", action="store_true", help="skip the blockchain write")
    ap.add_argument("--yes", action="store_true", help="accept the consent notice")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args(argv)

    con = Console()
    if not args.image and not args.replay:
        con.print("[red]Provide --image PATH (or --replay RUN_ID).[/]")
        return EXIT_BAD_INPUT

    cfg = get_settings()
    if not consent_gate(con, assume=args.yes):
        return EXIT_CONFIG

    # --- threshold must come from measured data ---------------------------
    try:
        thresholds = Thresholds.load()
    except CalibrationMissing as e:
        con.print(f"[red]CALIBRATION MISSING[/]\n{e}")
        return EXIT_CALIBRATION

    replaying = bool(args.replay)
    if replaying:
        run_dir = RUNS / args.replay
        if not run_dir.exists():
            con.print(f"[red]No such run: {run_dir}[/]")
            return EXIT_BAD_INPUT
        run_id = args.replay
        if not (run_dir / "run.json").exists():
            con.print(f"[red]RUN IS NOT REPLAYABLE[/] {run_dir}")
            con.print("  No run.json manifest. Only runs captured by a live")
            con.print("  execution of this pipeline can be replayed.")
            return EXIT_BAD_INPUT
    else:
        run_dir, run_id = new_run_dir()

    log = setup_logging(run_dir=run_dir, secrets=cfg.secret_values(), verbose=args.verbose)
    say = stage(log, "RUN")

    if replaying:
        _m = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        con.print(Panel(
            f"[bold yellow]REPLAY MODE[/] -- using captured responses from run {run_id}\n"
            f"No live network calls. Original capture: {_m.get('started_at')}",
            border_style="yellow", expand=False))

    t_start = time.time()
    say("run %s", run_id)

    # ---------------- FACE ------------------------------------------------
    import cv2
    if replaying:
        meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        src = run_dir / "input" / meta["input_name"]
    else:
        src = args.image
    if not src.exists():
        con.print(f"[red]INPUT NOT FOUND[/] {src}")
        return EXIT_BAD_INPUT

    raw = src.read_bytes()
    input_sha = hashlib.sha256(raw).hexdigest()
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        con.print(f"[red]NOT A DECODABLE IMAGE[/] {src}")
        return EXIT_BAD_INPUT

    if not replaying:
        (run_dir / "input").mkdir(exist_ok=True)
        (run_dir / "input" / src.name).write_bytes(raw)
        # run.json is the manifest --replay reads. Written here, before any
        # network call, so a run that fails midway is still replayable.
        (run_dir / "run.json").write_text(json.dumps({
            "run_id": run_id,
            "started_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "input_name": src.name,
            "input_sha256": input_sha,
            "input_width": img.shape[1],
            "input_height": img.shape[0],
        }, indent=2), encoding="utf-8")

    fsay = stage(log, "FACE")
    engine = FaceEngine.shared(logger=log)
    try:
        face, all_faces = engine.detect_primary(
            img, min_face_px=cfg.min_face_px, min_det_score=cfg.min_det_score)
    except NoFaceDetected as e:
        con.print(f"[red]NO FACE DETECTED[/]\n{e}")
        return EXIT_NO_FACE
    except (FaceTooSmall, FaceTooBlurry) as e:
        con.print(f"[red]QUALITY GATE REJECTED INPUT[/]\n{e}")
        return EXIT_QUALITY

    h, w = img.shape[:2]
    fsay("input %dx%d sha256=%s", w, h, input_sha[:16])
    fsay("detected %d face(s); using idx %d det=%.4f %dpx sharpness=%.0f",
         len(all_faces), face.index, face.det_score, face.face_px, face.sharpness)
    fsay("model %s, 512-d embedding, metric cosine_l2normed", engine.model_id())

    # ---------------- HOST + SEARCH --------------------------------------
    raw_dir = run_dir / "raw"
    if replaying:
        from .search.base import Candidate, SearchResult
        cached = json.loads((run_dir / "search.json").read_text(encoding="utf-8"))
        search_res = SearchResult(
            provider=cached["provider"],
            candidates=[Candidate(**c) for c in cached["candidates"]],
            raw_path=raw_dir / "serpapi_lens.json",
            query_image_sha256=cached["query_image_sha256"],
            exact_match_count=cached.get("exact_match_count"),
            notes="replayed from capture")
        stage(log, "SEARCH")("replayed %d candidates from capture",
                             len(search_res.candidates))
    else:
        try:
            cfg.require_search()
        except ConfigError as e:
            con.print(f"[red]CONFIG ERROR[/]\n{e}")
            return EXIT_CONFIG

        hosted = None
        host = get_host(cfg.image_host_backend)
        if host.name != "none":
            try:
                hosted = host.upload(raw)
                stage(log, "HOST")("uploaded to %s -> %s", host.name, hosted.url)
            except ImageHostError as e:
                stage(log, "HOST")("upload failed: %s", e)

        try:
            search_res = run_search(build_providers(cfg, log), raw, raw_dir, cfg, log,
                                    hosted_url=hosted.url if hosted else None)
        except ProviderError as e:
            con.print(f"[red]SEARCH FAILED[/]\n{e}")
            return EXIT_SEARCH
        finally:
            if hosted:
                gone = host.delete(hosted)
                stage(log, "HOST")(
                    "hosted copy %s",
                    "deleted" if gone else "left to expire (host has no delete API)")

        (run_dir / "search.json").write_text(json.dumps({
            "provider": search_res.provider,
            "query_image_sha256": search_res.query_image_sha256,
            "exact_match_count": search_res.exact_match_count,
            "candidates": [c.as_record() for c in search_res.candidates],
        }, indent=2), encoding="utf-8")

    if not search_res.candidates:
        con.print("[red]ZERO CANDIDATES[/] The search returned nothing to verify.")
        return EXIT_NO_CANDIDATES

    # ---------------- FETCH ----------------------------------------------
    if replaying:
        # Replay must be fully offline, or the banner is a lie and a network
        # failure would break a replay exactly as it breaks a live run.
        fetched = load_cached(search_res.candidates, run_dir / "candidates",
                              logger=stage(log, "FETCH"))
    else:
        fetcher = CandidateFetcher(timeout=cfg.fetch_timeout_s,
                                   max_bytes=cfg.max_download_bytes,
                                   logger=stage(log, "FETCH"))
        fetched = fetcher.fetch_all(search_res.candidates, run_dir / "candidates")
    usable = [f for f in fetched if f.ok]
    if not usable:
        con.print("[red]NO CANDIDATE IMAGES COULD BE DOWNLOADED[/]")
        return EXIT_NO_CANDIDATES

    # ---------------- VERIFY ---------------------------------------------
    vsay = stage(log, "VERIFY")
    scored = []
    for f in usable:
        s = score_candidate(engine=engine, input_face=face, input_sha256=input_sha,
                            fetched=f, thresholds=thresholds)
        if s is None:
            vsay("candidate %d: no face detected", f.position)
            continue
        scored.append(s)
        vsay("candidate %d %-22s cos=%+.4f pHashD=%s %s",
             s.position, s.source[:22], s.similarity, s.phash_distance, s.verdict.name)

    if not scored:
        con.print("[red]NO FACES FOUND IN ANY CANDIDATE[/]")
        return EXIT_NO_CANDIDATES

    ranked = rank(scored)
    spread = score_spread(scored)

    t = Table(title="Ranked candidates (strongest evidence first)", show_lines=False)
    for c in ("#", "source", "cosine", "pHashD", "faces", "verdict"):
        t.add_column(c)
    for s in ranked[:12]:
        style = {"DISTINCT_PHOTO": "green", "SAME_PHOTO": "yellow",
                 "EXACT_DUPLICATE": "red", "NO_MATCH": "dim"}[s.verdict.name]
        t.add_row(str(s.position), s.source[:26], f"{s.similarity:+.4f}",
                  str(s.phash_distance), str(s.faces_in_candidate),
                  f"[{style}]{s.verdict.name}[/]")
    con.print(t)

    best = ranked[0] if ranked and ranked[0].verdict is not Verdict.NO_MATCH else None
    if best is None:
        con.print(Panel(
            f"[bold]NO CANDIDATE ABOVE THRESHOLD[/]\n\n"
            f"  best cosine    : {spread.get('best')}\n"
            f"  threshold      : {thresholds.similarity:.4f}\n"
            f"  candidates     : {len(scored)}\n\n"
            "This is a valid outcome and is reported honestly.\n"
            "The threshold is NOT lowered to force a match.",
            border_style="red", expand=False))
        return EXIT_NO_MATCH

    distinct_artifact_panel(con, input_sha=input_sha, input_dims=(w, h), best=best,
                            thresholds=thresholds, verdict=best.verdict)

    # ---------------- EVIDENCE -------------------------------------------
    esay = stage(log, "EVIDENCE")
    ev = build_evidence(EvidenceInputs(
        input_image_sha256=input_sha,
        input_face_phash=face.face_phash,
        matched_image_sha256=best.sha256,
        matched_image_phash=best.face_phash,
        phash_hamming_distance=best.phash_distance,
        face_similarity=best.similarity,
        threshold=thresholds.similarity,
        faces_in_candidate=best.faces_in_candidate,
        matched_face_index=best.matched_face_index,
        matched_face_bbox=best.matched_face_bbox,
        source_url=best.page_url,
        page_title=next((c.title for c in search_res.candidates
                         if c.position == best.position), ""),
        search_provider=search_res.provider,
        search_query_image_sha256=search_res.query_image_sha256,
        candidates_examined=len(fetched),
        retrieved_at=dt.datetime.now(dt.timezone.utc),
        model=engine.model_id(),
        verdict=best.verdict.name,
        exact_match_count=search_res.exact_match_count,
    ))
    ev_hash = evidence_hash(ev)
    esay("evidence sha256 = %s", ev_hash)

    # ---------------- CHAIN ----------------------------------------------
    chain_meta = {}
    if not args.no_chain and not replaying:
        csay = stage(log, "CHAIN")
        try:
            cfg.require_chain()
            cfg.require_salt()
            from .blockchain.client import ChainClient, ChainError
            from .blockchain.registry import AlreadyRecorded, Registry

            client = ChainClient(cfg.base_sepolia_rpc, cfg.deployer_private_key)
            reg = Registry(client, cfg.evidence_registry_address)
            commitment = subject_commitment(face.embedding, cfg.subject_commitment_salt)
            csay("recording on Base Sepolia (chainId %d) ...", client.chain_id)
            res = reg.record_evidence(ev_hash, commitment, ev["source_domain"])
            rec = reg.wait_for_record(ev_hash)
            csay("block %d gas %d cost %s ETH", res.block_number, res.gas_used,
                 client.w3.from_wei(res.cost_wei(), "ether"))
            chain_meta = {
                "network": "base-sepolia", "chain_id": client.chain_id,
                "contract": reg.address, "tx_hash": res.tx_hash,
                "block_number": res.block_number,
                "recorded_at_unix": rec.timestamp,
                "subject_commitment": commitment,
                "explorer": res.explorer_url(),
            }
            con.print(f"  [green]recorded on chain[/]  {res.explorer_url()}")
        except AlreadyRecorded as e:
            csay("duplicate refused: first seen unix %d", e.first_seen)
            con.print(f"[yellow]ALREADY RECORDED[/]\n{e}")
            chain_meta = {"network": "base-sepolia",
                          "contract": cfg.evidence_registry_address,
                          "already_recorded": True, "first_seen_unix": e.first_seen}
        except (ChainError, ConfigError) as e:
            csay("chain write failed: %s", str(e).splitlines()[0])
            con.print(f"[yellow]CHAIN WRITE SKIPPED[/]\n{e}")
            chain_meta = {"error": str(e).splitlines()[0]}
    elif replaying:
        stage(log, "CHAIN")("replay mode: no chain write")

    bundle = build_bundle(
        evidence=ev, run_id=run_id, chain=chain_meta,
        runner_up=[s.as_record() for s in ranked[1:6]],
        notes=f"score spread: {spread}")
    paths = write_artifacts(run_dir, ev, bundle)

    con.print()
    con.print(Panel(
        f"  run           : {run_id}\n"
        f"  input sha256  : {input_sha[:32]}...\n"
        f"  matched       : {domain_of(best.page_url)}  cosine {best.similarity:.4f}\n"
        f"  verdict       : {best.verdict.value}\n"
        f"  evidence hash : {ev_hash}\n"
        f"  chain tx      : {chain_meta.get('tx_hash', '(not recorded)')}\n"
        f"  elapsed       : {time.time()-t_start:.1f}s\n\n"
        f"  Verify independently (web3 only, none of this code):\n"
        f"    python verify.py --bundle {paths['bundle'].relative_to(ROOT)}\n\n"
        f"  Reproduce the hash with the standard library alone:\n"
        f"    python -c \"import hashlib,sys;print(hashlib.sha256("
        f"open(sys.argv[1],'rb').read()).hexdigest())\" "
        f"{paths['canonical'].relative_to(ROOT)}",
        title="[bold]JUDGE SUMMARY[/]", border_style="cyan", expand=False))
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
