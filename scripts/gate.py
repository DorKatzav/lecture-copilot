"""Milestone gate: `python scripts/gate.py --m N` prints one line per check and `GATE M<N>: PASS k/k`.

Checks read real state (binaries, the Ollama server, files, eval/stage0.json). A crashing check is a FAIL,
never a crash of the gate. The secret scan reports file names only — it never prints key material.
"""

import argparse
import asyncio
import json
import math
import os
import sqlite3
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from lecture_copilot.config import (
    BUDGET_S,
    CHUNK_BUDGET_S,
    COST_BUDGET_USD,
    COURSES_ROOT,
    DB_PATH,
    DIGEST_BUDGET_S,
    DIGEST_MODEL,
    EMBED_MODEL,
    LIVE_MODEL,
    OLLAMA_URL,
    ROOT,
    Profile,
)
from lecture_copilot.scriptcheck import terminal_text

STAGE0 = ROOT / "eval" / "stage0.json"
FIXTURE = ROOT / "eval" / "fixture_10min.m4a"
SECRET_PATTERN = r"AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_.-]{40,}|ntn_[A-Za-z0-9]{20,}|secret_[A-Za-z0-9]{20,}"


@dataclass(frozen=True)
class Result:
    name: str
    status: str  # PASS | FAIL | SKIP
    detail: str


def _ok(name: str, detail: str) -> Result:
    return Result(name, "PASS", detail)


def _fail(name: str, detail: str) -> Result:
    return Result(name, "FAIL", detail)


# ---------- checks ----------

def check_mw_version(cmd: tuple[str, ...] = ("mw", "version")) -> Result:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return _fail("mw_version", "mw not on PATH — MacWhisper → Settings → Advanced → Install CLI")
    if p.returncode != 0:
        return _fail("mw_version", f"exit {p.returncode}: {(p.stderr or p.stdout).strip()[:120]}")
    return _ok("mw_version", (p.stdout.strip().splitlines() or ["ok"])[0])


def check_ollama_models(client: httpx.Client | None = None,
                        required: tuple[str, ...] = tuple(dict.fromkeys((LIVE_MODEL, DIGEST_MODEL, EMBED_MODEL)))
                        ) -> Result:
    client = client or httpx.Client(base_url=OLLAMA_URL, timeout=5)
    try:
        names = {m["name"].removesuffix(":latest") for m in client.get("/api/tags").json()["models"]}
    except httpx.HTTPError as e:
        return _fail("ollama_models", f"ollama not reachable at {OLLAMA_URL} ({type(e).__name__})")
    missing = [m for m in required if m.removesuffix(":latest") not in names]
    if missing:
        return _fail("ollama_models", f"missing: {', '.join(missing)} — run scripts/setup_models.sh")
    return _ok("ollama_models", ", ".join(required))


def _load_vec(con: sqlite3.Connection) -> str:
    import sqlite_vec
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    return con.execute("select vec_version()").fetchone()[0]


def check_sqlite_vec(results: dict, loader: Callable[[sqlite3.Connection], str] = _load_vec) -> Result:
    try:
        return _ok("sqlite_vec", f"sqlite-vec {loader(sqlite3.connect(':memory:'))} loads")
    except Exception as e:
        if results.get("sqlite_vec", {}).get("fallback") == "numpy":
            return _ok("sqlite_vec", f"does not load ({type(e).__name__}); numpy fallback flagged")
        return _fail("sqlite_vec", f"does not load ({type(e).__name__}) and no fallback flagged — stage0.py sqlite")


def _ffprobe(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def check_fixture(path: Path = FIXTURE, duration: Callable[[Path], float] = _ffprobe) -> Result:
    if not path.exists():
        return _fail("fixture", f"{path.name} missing — scripts/make_fixture.sh <lecture>")
    secs = duration(path)
    if abs(secs - 600) > 5:
        return _fail("fixture", f"{path.name} is {secs:.0f} s, expected 600 ± 5")
    return _ok("fixture", f"{path.name} {secs:.0f} s")


def check_mw_bench(results: dict) -> Result:
    mw = results.get("mw") or {}
    runs = mw.get("runs_s") or []
    if len(runs) < 5 or mw.get("p50_s") is None:
        return _fail("mw_bench", f"{len(runs)}/5 runs logged — stage0.py mw-bench")
    if mw.get("verdict") not in ("hot", "cold"):
        return _fail("mw_bench", "no hot/cold verdict written")
    return _ok("mw_bench", f"p50 {mw['p50_s']} s over {len(runs)} runs → {mw['verdict']}, ASR = {mw.get('decision')}")


def check_extract_bench(results: dict, model: str = LIVE_MODEL, min_n: int = 10, min_share: float = 0.9) -> Result:
    ex = (results.get("extract") or {}).get(model) or {}
    n, ok = ex.get("n", 0), ex.get("valid_first", 0)
    if n < min_n:
        return _fail("extract_bench", f"{model}: {n} chunks measured, need {min_n} — stage0.py extract-bench")
    if ok < math.ceil(min_share * n):
        return _fail("extract_bench", f"{model}: {ok}/{n} valid JSON on first attempt, need ≥ {min_share:.0%}")
    return _ok("extract_bench", f"{model}: {ok}/{n} valid JSON on first attempt")


def check_mic(results: dict, root: Path = ROOT) -> Result:
    mic = results.get("mic") or {}
    if not mic.get("file") or not (root / mic["file"]).exists():
        return _fail("mic_seat", "no recording — record 1 min from the seat, then stage0.py mic <file>")
    if mic.get("readable") not in ("yes", "no"):
        return _fail("mic_seat", "transcript not judged — stage0.py mic-verdict yes|no")
    return _ok("mic_seat", f"recorded, readable: {mic['readable']}")


def _git_grep_files(root: Path, opts: tuple[str, ...] = (), revs: tuple[str, ...] = ()) -> tuple[int, list[str], str]:
    # options before the pattern, revisions after it
    p = subprocess.run(["git", "-C", str(root), "grep", "-I", "-l", "-z", *opts, "-iE", SECRET_PATTERN, *revs],
                       capture_output=True, text=True)
    return p.returncode, [f for f in p.stdout.split("\0") if f], p.stderr


def check_secret_scan(root: Path = ROOT) -> Result:
    """Working tree (tracked + untracked, .gitignore respected), the index, and every commit — what a push publishes."""
    revs = subprocess.run(["git", "-C", str(root), "rev-list", "--all"], capture_output=True, text=True).stdout.split()
    scopes = [("working tree", ("--untracked",), ()), ("index", ("--cached",), ())]
    if revs:
        scopes.append(("history", (), tuple(revs)))
    hits: list[str] = []
    for label, opts, commits in scopes:
        code, files, err = _git_grep_files(root, opts, commits)
        if code == 0:
            if label == "history":  # "<sha>:<path>" → "<sha7>:<path>"
                files = [f"{f.split(':', 1)[0][:7]}:{f.split(':', 1)[1]}" for f in files]
            hits += [f"{label}: {f}" for f in dict.fromkeys(files)]
        elif code != 1:
            return _fail("secret_scan", f"git grep ({label}) failed: {err.strip()[:120]}")
    if hits:
        return _fail("secret_scan", "key material in " + "; ".join(hits))
    return _ok("secret_scan", "no key material in the working tree, the index or history")


# ---------- M1: the pipeline on the fixture (real state in db/copilot.sqlite) ----------

REPLAY_HINT = ('python -m lecture_copilot.cli replay eval/fixture_10min.m4a --course "AI Developers — Python" '
               '--date 2026-06-19 (twice)')
PRESSURE = {1: "normal", 2: "warning", 4: "critical"}
SWAP_GROWTH_LIMIT_MB = 1024


def _fixture_runs(db: Path, fixture: Path) -> tuple[list[str], list[dict], sqlite3.Connection]:
    """Lecture ids replayed from the fixture, and their complete runs (oldest first)."""
    con = sqlite3.connect(db)
    lectures = [r[0] for r in con.execute("select id from lectures where source = 'file' and audio_path = ?",
                                          (str(fixture.resolve()),))]
    runs = []
    for lid in lectures:
        for ref, out in con.execute("select input_ref, output_json from decisions where node = 'run' "
                                    "and lecture_id = ? order by ts", (lid,)):
            runs.append({"lecture_id": lid, "run_id": ref, **json.loads(out)})
    return lectures, [r for r in runs if "source_error" not in r], con


def _table_counts(con: sqlite3.Connection, lecture_id: str) -> dict[str, int]:
    return {t: con.execute(f"select count(*) from {t} where lecture_id = ?", (lecture_id,)).fetchone()[0]
            for t in ("segments", "items", "claims")}


def check_fixture_replay(db: Path = DB_PATH, fixture: Path = FIXTURE) -> Result:
    _, runs, _ = _fixture_runs(db, fixture)
    if not runs:
        return _fail("fixture_replay", f"no complete replay of {fixture.name} — {REPLAY_HINT}")
    last = runs[-1]
    c = last["counts"]
    detail = (f"{c['segments']} segments · {c['items']} items · {c['claims']} claims "
              f"({last['chunks']} chunks: {', '.join(f'{k} {v}' for k, v in sorted(last['status'].items()))})")
    if c["segments"] < 10 or c["items"] < 5 or c["claims"] < 1:
        return _fail("fixture_replay", f"{detail}; need segments ≥ 10, items ≥ 5, claims ≥ 1")
    return _ok("fixture_replay", detail)


def check_rerun_upsert(db: Path = DB_PATH, fixture: Path = FIXTURE) -> Result:
    """Replay is an upsert: one lecture row, and the tables hold exactly what the last replay wrote — nothing
    from an older run. Counts may differ between replays: the ASR provider is not deterministic (D-M1-5);
    identical counts with deterministic providers are asserted in tests/test_pipeline.py."""
    lectures, runs, con = _fixture_runs(db, fixture)
    if len(lectures) != 1:
        return _fail("rerun_upsert", f"{len(lectures)} lecture rows for {fixture.name}, expected 1")
    if len(runs) < 2:
        return _fail("rerun_upsert", f"{len(runs)} complete replay(s) — {REPLAY_HINT}")
    a, b = runs[-2]["counts"], runs[-1]["counts"]
    table = _table_counts(con, lectures[0])
    if table != b:
        return _fail("rerun_upsert", f"tables hold {table}, the last replay wrote {b}")
    # ids are ULIDs (time-ordered): a row minted before the last run started is a leftover
    stale = {t: con.execute(f"select count(*) from {t} where lecture_id = ? and id < ?",
                            (lectures[0], runs[-1]["run_id"])).fetchone()[0] for t in ("segments", "items", "claims")}
    if any(stale.values()):
        left = ", ".join(f"{t} {n}" for t, n in stale.items() if n)
        return _fail("rerun_upsert", f"rows older than the last replay: {left}")
    diff = ", ".join(f"{k} {a[k]}→{b[k]}" for k in a if a[k] != b[k])
    detail = f"1 lecture, {len(runs)} replays, tables = last replay {b}, 0 older rows"
    return _ok("rerun_upsert", detail + (f"; vs previous replay: {diff} (ASR varies, D-M1-5)" if diff else
                                         "; identical to the previous replay"))


def check_chunk_budget(db: Path = DB_PATH, fixture: Path = FIXTURE) -> Result:
    _, runs, con = _fixture_runs(db, fixture)
    if not runs:
        return _fail("chunk_budget", f"no complete replay — {REPLAY_HINT}")
    last = runs[-1]
    rows = [json.loads(o) | {"ref": ref} for ref, o in con.execute(
        "select input_ref, output_json from decisions where node = 'chunk' and input_ref like ?",
        (last["run_id"] + "#%",))]
    over = [f"{r['ref'].split('#')[1]} {r['total_s']} s" for r in rows if r["total_s"] > CHUNK_BUDGET_S]
    t = last["timing"]
    p95 = {"asr": t["asr_s"]["p95"], "extract": t["extract_s"]["p95"]}
    detail = (f"p95 asr {p95['asr']} s (≤ {BUDGET_S['asr']}) · extract {p95['extract']} s (≤ {BUDGET_S['extract']}) "
              f"· total {t['total_s']['p95']} s, max {t['total_s']['max']} s (≤ {CHUNK_BUDGET_S}) "
              f"over {len(rows)} chunks")
    stages_over = [k for k, v in p95.items() if v is not None and v > BUDGET_S[k]]
    if over or stages_over:
        return _fail("chunk_budget", f"over budget: {', '.join(over + [f'p95 {k}' for k in stages_over])} — {detail}")
    return _ok("chunk_budget", detail)


def check_peak_memory(db: Path = DB_PATH, fixture: Path = FIXTURE) -> Result:
    _, runs, _ = _fixture_runs(db, fixture)
    m = runs[-1].get("memory") if runs else None
    if not m or not m.get("samples"):
        return _fail("peak_memory", f"no memory samples in the last replay — {REPLAY_HINT}")
    need = [x.removesuffix(":latest") for x in (LIVE_MODEL, EMBED_MODEL)]
    missing = [x for x in need if x not in m["models_at_peak_gb"]] + [
        x for x in ("MacWhisper",) if x not in m["procs_peak_mb"]]
    procs = ", ".join(f"{k} {v / 1024:.1f} GB" for k, v in sorted(m["procs_peak_mb"].items(), key=lambda kv: -kv[1])
                      if v >= 100)
    pressure = PRESSURE.get(m["max_pressure"], str(m["max_pressure"]))
    detail = (f"peak {m['peak_used_gb']} of {m['total_gb']} GB, pressure {pressure}, swap {m['swap_growth_mb']:+.0f} MB"
              f" · {procs}")
    if missing:
        return _fail("peak_memory", f"not resident at the peak: {', '.join(missing)} — {detail}")
    # D-M1-6: it fits when the OS never reaches critical pressure and swap does not grow during the run
    if m["max_pressure"] >= 4 or m["swap_growth_mb"] > SWAP_GROWTH_LIMIT_MB:
        return _fail("peak_memory", f"{detail} (limits: pressure below critical, swap growth ≤ "
                                    f"{SWAP_GROWTH_LIMIT_MB} MB)")
    return _ok("peak_memory", detail)


def _ollama_serve_env() -> str | None:
    """Command line + environment of the running `ollama serve` (macOS `ps eww` shows a process's environment)."""
    pids = subprocess.run(["pgrep", "-f", "^ollama serve"], capture_output=True, text=True).stdout.split()
    if not pids:
        return None
    return subprocess.run(["ps", "eww", "-o", "command=", "-p", pids[0]], capture_output=True, text=True).stdout


def check_prompt_cache(server_env: Callable[[], str | None] = _ollama_serve_env) -> Result:
    """D-M1-4: llama-server's host-RAM prompt cache (8 GiB by default) must be off in the running Ollama server."""
    env = server_env()
    if env is None:
        return _fail("prompt_cache", "ollama serve is not running — scripts/ollama_serve.sh")
    if "LLAMA_ARG_CACHE_RAM=0" not in env.split():
        return _fail("prompt_cache", "ollama serve runs with llama-server's 8 GiB prompt cache on — restart it with "
                                     "scripts/ollama_serve.sh (D-M1-4)")
    return _ok("prompt_cache", "ollama serve runs with LLAMA_ARG_CACHE_RAM=0 (prompt cache off, D-M1-4)")


async def _asr_error_run(work: Path, asr, client: httpx.AsyncClient, chunks: list) -> dict:
    from lecture_copilot.pipeline import Ctx, run
    from lecture_copilot.store.db import Store, new_id

    store = Store(work / "gate_m1.sqlite")
    try:
        course = store.upsert_course("gate", language="he")
        lid = store.upsert_lecture(course, audio_path="gate:asr_error", source="file", title="gate", date="-",
                                   fact_check=False)

        async def source():
            for c in chunks:
                yield c

        async with client:
            ctx = Ctx(lecture_id=lid, course_id=course, course_name="gate", lecture_title="gate",
                      profile=Profile(fact_check=False, language="he"), store=store, asr=asr, ollama=client,
                      run_id=new_id(), runs_dir=work)
            await run(source(), ctx)
        return {json.loads(o)["idx"]: json.loads(o) for (o,) in store.con.execute(
            "select output_json from decisions where node = 'chunk' and input_ref like ?", (ctx.run_id + "#%",))}
    finally:
        store.close()


def _asr_error_inputs(work: Path) -> list:
    from lecture_copilot.audio.sources import AudioChunk
    bad, good = work / "chunk_0001.wav", work / "chunk_0002.wav"
    bad.write_bytes(b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(range(256)) * 4)   # a header, then garbage
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", "120", "-t", "30", "-i", str(FIXTURE), "-ac", "1", "-ar",
                    "16000", str(good)], check=True)
    return [AudioChunk("gate", 1, bad, 0.0, 30.0), AudioChunk("gate", 2, good, 30.0, 60.0)]


def check_asr_error(work: Path | None = None, asr=None, client: httpx.AsyncClient | None = None,
                    chunks: list | None = None) -> Result:
    """Break it on purpose: a corrupt wav through the real pipeline must be marked failed, the next chunk must run."""
    from lecture_copilot.asr.macwhisper import MacWhisperASR

    with tempfile.TemporaryDirectory() as tmp:
        work = work or Path(tmp)
        chunks = chunks if chunks is not None else _asr_error_inputs(work)
        out = asyncio.run(_asr_error_run(work, asr or MacWhisperASR(), client or httpx.AsyncClient(base_url=OLLAMA_URL),
                                         chunks))
    first, second = out.get(1, {}), out.get(2, {})
    detail = (f"corrupt chunk → {first.get('status')} ({terminal_text(first.get('error', ''), 90)}); "
              f"next chunk → {second.get('status')}")
    if first.get("status") != "asr_failed" or second.get("status") != "ok":
        return _fail("asr_error", detail)
    return _ok("asr_error", detail + "; run completed")


def check_tests(cmds: tuple[tuple[str, ...], ...] = ((sys.executable, "-m", "pytest", "-q"),
                                                      (sys.executable, "-m", "ruff", "check", "."))) -> Result:
    lines = []
    for cmd in cmds:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        last = (p.stdout.strip().splitlines() or [""])[-1]
        if p.returncode != 0:
            return _fail("tests", f"{' '.join(cmd[-3:])} exit {p.returncode}: {last[:100]}")
        lines.append(last)
    return _ok("tests", " · ".join(x for x in lines if x) or "ok")


# ---------- M2: the Digest (real state in db/copilot.sqlite and COURSES_ROOT) ----------

IMG_DIR = ROOT / "docs" / "reports" / "img"
LECTURE_FILES = ("digest.md", "digest.html", "transcript.txt", "claims.json")


def _digests(db: Path, where: str = "") -> list[sqlite3.Row]:
    con = sqlite3.connect(db)
    con.row_factory = sqlite3.Row
    return con.execute("select l.id, l.source, l.title, s.digest_md from lecture_summaries s "
                       f"join lectures l on l.id = s.lecture_id {where} order by l.ended_at").fetchall()


def _last_sink_folder(db: Path) -> tuple[dict | None, Path | None]:
    con = sqlite3.connect(db)
    row = con.execute("select output_json from decisions where node = 'sink' order by ts desc limit 1").fetchone()
    out = json.loads(row[0]) if row else None
    return out, Path(out["folder"]) if out and out.get("folder") else None


def check_digest_sections(db: Path = DB_PATH) -> Result:
    from lecture_copilot.output.digest import SECTIONS, section_headings
    rows = _digests(db)
    if not rows:
        return _fail("digest_sections", "no Digest in the database — replay a lecture")
    bad = [r["id"][-6:] for r in rows if section_headings(r["digest_md"]) != SECTIONS]
    if bad:
        return _fail("digest_sections", f"{len(bad)}/{len(rows)} Digest(s) without the 9 sections in order: "
                                        f"{', '.join(bad)}")
    return _ok("digest_sections", f"{len(rows)} Digest(s), each with the 9 sections in the fixed order")


def check_digest_time(db: Path = DB_PATH, min_minutes: int = 60, max_minutes: int = 120) -> Result:
    """The spec promises the Digest within two minutes for a lecture of up to two hours. The latest Digest of every
    lecture of 60–120 minutes is judged; longer recordings (a four-hour Zoom day) are reported, not judged."""
    con = sqlite3.connect(db)
    latest: dict[str, dict] = {}
    for lid, o in con.execute("select lecture_id, output_json from decisions where node = 'digest' "
                              "and input_ref like '%#digest' order by ts"):
        latest[lid] = json.loads(o)
    judged = [r for r in latest.values() if min_minutes <= r.get("minutes", 0) <= max_minutes]
    longer = [r for r in latest.values() if r.get("minutes", 0) > max_minutes]
    if not judged:
        return _fail("digest_time",
                     f"no Digest of a lecture of {min_minutes}–{max_minutes} min — replay a full lecture")

    def line(r: dict) -> str:
        bad = f", degraded: {', '.join(r['degraded'])}" if r["degraded"] else ""
        return f"{r['minutes']} min → {r['total_s']} s ({r['blocks']} blocks{bad})"
    detail = f"{'; '.join(line(r) for r in judged)}, budget {DIGEST_BUDGET_S} s"
    if longer:
        detail += f" · longer, not judged: {'; '.join(line(r) for r in longer)}"
    if any(r["degraded"] or r["total_s"] >= DIGEST_BUDGET_S for r in judged):
        return _fail("digest_time", detail)
    return _ok("digest_time", detail)


def check_vtt_replay(db: Path = DB_PATH) -> Result:
    from lecture_copilot.output.digest import SECTIONS, section_headings
    rows = _digests(db, "where l.source = 'transcript'")
    if not rows:
        return _fail("vtt_replay", "no transcript lecture with a Digest — replay a .vtt")
    n = len(section_headings(rows[-1]["digest_md"]))
    if section_headings(rows[-1]["digest_md"]) != SECTIONS:
        return _fail("vtt_replay", f"the transcript replay has {n} sections, not the 9 in order")
    return _ok("vtt_replay", "a transcript replay ends in the same 9 sections as an audio replay")


def check_course_folder(db: Path = DB_PATH, courses_root: Path = COURSES_ROOT) -> Result:
    out, folder = _last_sink_folder(db)
    if out is None:
        return _fail("course_folder", "no sink run logged — replay a lecture")
    if out["status"] != "ok" or folder is None:
        return _fail("course_folder", f"the last sink run failed: {terminal_text(out.get('error', ''), 100)}")
    root = Path(courses_root).resolve()
    if root not in folder.resolve().parents:
        return _fail("course_folder", "the lecture folder is not under COURSES_ROOT")
    missing = [f for f in LECTURE_FILES if not (folder / f).is_file()]
    if not (folder.parent / "index.md").is_file():
        missing.append("index.md")
    if missing:
        return _fail("course_folder", f"missing in the lecture folder: {', '.join(missing)}")
    where = "inside Google Drive" if "GoogleDrive" in str(root) or "Google Drive" in str(root) else \
        "a local folder — Google Drive for desktop is not set up yet (PLAN §7)"
    return _ok("course_folder", f"{len(LECTURE_FILES)} files + index.md under COURSES_ROOT, {where}")


def check_html_rtl(db: Path = DB_PATH, img_dir: Path = IMG_DIR) -> Result:
    """Markup is checked here; rendering is checked by eye — the gate wants the two screenshots that prove it."""
    import re

    from lecture_copilot.output.digest import SECTIONS
    out, folder = _last_sink_folder(db)
    if folder is None or not (folder / "digest.html").is_file():
        return _fail("html_rtl", "no digest.html from the last sink run")
    html = (folder / "digest.html").read_text(encoding="utf-8")
    heads = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<h2[^>]*>(.*?)</h2>", html, re.DOTALL)]
    if '<html lang="he" dir="rtl">' not in html:
        return _fail("html_rtl", 'digest.html does not open with <html lang="he" dir="rtl">')
    if [next((s for s in SECTIONS if h.startswith(s)), h) for h in heads] != SECTIONS:
        return _fail("html_rtl", f"digest.html has {len(heads)} sections, not the 9 in order")
    shots = [img_dir / "m2_digest_desktop.jpg", img_dir / "m2_digest_phone.jpg"]
    missing = [p.name for p in shots if not p.is_file() or p.stat().st_size < 1000]
    if missing:
        return _fail("html_rtl", f"no browser screenshot: {', '.join(missing)} in docs/reports/img/")
    return _ok("html_rtl", 'lang="he" dir="rtl", 9 sections; browser screenshots at full and phone width')


# ---------- M3: the course memory (real state in db/copilot.sqlite; one live run) ----------

BENCHMARK = ROOT / "eval" / "benchmark.json"
MEMORY_BUDGET_S = BUDGET_S["embed"] + 1          # one batch of embeddings + one search (spec: ≤ 2 + ≤ 1)


def _transcript_lectures(con: sqlite3.Connection) -> list[sqlite3.Row]:
    con.row_factory = sqlite3.Row
    return con.execute("select * from lectures where source = 'transcript' order by date, started_at").fetchall()


def _two_lectures(db: Path) -> tuple[sqlite3.Connection, sqlite3.Row, sqlite3.Row] | None:
    con = sqlite3.connect(db)
    rows = _transcript_lectures(con)
    return (con, rows[-2], rows[-1]) if len(rows) >= 2 else None


def check_shared_concepts(db: Path = DB_PATH, benchmark_path: Path = BENCHMARK, min_share: float = 0.8) -> Result:
    """Dor's labels: eval/benchmark.json → shared_concepts[{term, first, again}] (dates). A labelled term counts
    as flagged when the later lecture has it (term or key) with first_seen pointing at the earlier lecture."""
    if not benchmark_path.is_file():
        return Result("shared_concepts", "SKIP", "needs eval/benchmark.json with shared_concepts — Dor's labelling "
                                                 "(PLAN §7); candidates: scripts/m3.py candidates")
    labels = json.loads(benchmark_path.read_text(encoding="utf-8")).get("shared_concepts", [])
    pair = _two_lectures(db)
    if not labels or pair is None:
        return _fail("shared_concepts", "no labels or fewer than two transcript lectures replayed")
    con, first, again = pair
    flagged, missed = 0, []
    for lab in labels:
        term = lab["term"].strip().lower()
        row = con.execute(
            "select first_seen_lecture_id from items where lecture_id = ? and kind = 'concept' "
            "and (lower(text) = ? or lower(canonical_key) = ?)", (again["id"], term, term)).fetchone()
        if row and row[0] == first["id"]:
            flagged += 1
        else:
            missed.append(lab["term"])
    detail = f"{flagged}/{len(labels)} labelled shared concepts flagged as already said"
    if flagged < math.ceil(min_share * len(labels)):
        return _fail("shared_concepts", f"{detail}; missed: {', '.join(missed[:8])}")
    return _ok("shared_concepts", detail)


def check_search_top1(db: Path = DB_PATH, n: int = 10, min_share: float = 0.8) -> Result:
    """Every k-th concept of the latest transcript lecture: search(term) must return that concept (or one with
    its key) first. The spec's example query, CAC, is not in this course."""
    from lecture_copilot.store.db import Store
    pair = _two_lectures(db)
    if pair is None:
        return _fail("search_top1", "fewer than two transcript lectures replayed")
    con, _, latest = pair
    con.close()
    store = Store(db)
    try:
        rows = store.items(latest["id"], kind="concept")
        if not rows:
            return _fail("search_top1", "the latest lecture has no concepts")
        step = max(1, len(rows) // n)
        sample = rows[::step][:n]
        hits, misses = 0, []
        for r in sample:
            top = store.search(r["text"], latest["course_id"], k=1)
            same_key = (top[0].canonical_key or "").lower() == (r["canonical_key"] or "").lower() if top else False
            if top and (top[0].id == r["id"] or same_key or top[0].text.lower() == r["text"].lower()):
                hits += 1
            else:
                misses.append(r["text"])
        detail = f"{hits}/{len(sample)} sampled terms return their own concept first"
        if hits < math.ceil(min_share * len(sample)):
            return _fail("search_top1", f"{detail}; missed: {', '.join(terminal_text(m, 30) for m in misses[:6])}")
        return _ok("search_top1", detail)
    finally:
        store.close()


def check_continuation(db: Path = DB_PATH) -> Result:
    pair = _two_lectures(db)
    if pair is None:
        return _fail("continuation", "fewer than two transcript lectures replayed")
    con, first, latest = pair
    row = con.execute("select digest_md from lecture_summaries where lecture_id = ?", (latest["id"],)).fetchone()
    if row is None:
        return _fail("continuation", "the latest transcript lecture has no Digest")
    md = row[0]
    if latest["continues_id"] != first["id"] or "## המשך מ-" not in md:
        return _fail("continuation", "the Digest does not continue the previous lecture")
    if "לא השווה" in md.split("## המשך מ-", 1)[1].split("\n## ")[0] or "**מה חדש:**" not in md:
        return _fail("continuation", "the continuation chapter is present but the model wrote nothing in it")
    return _ok("continuation", "the latest Digest continues the previous lecture with new / repeated / contradicts")


def check_memory_budget(db: Path = DB_PATH) -> Result:
    con = sqlite3.connect(db)
    row = con.execute("select output_json from decisions where node = 'run' order by ts desc limit 1").fetchone()
    if row is None:
        return _fail("memory_budget", "no run logged")
    p95 = (json.loads(row[0]).get("timing", {}).get("memory_s") or {}).get("p95")
    if p95 is None:
        return _fail("memory_budget", "the last run has no memory timing")
    detail = f"memory per chunk p95 {p95} s (embed ≤ {BUDGET_S['embed']} + search ≤ 1)"
    return _ok("memory_budget", detail) if p95 <= MEMORY_BUDGET_S else _fail("memory_budget", detail)


def check_numpy_fallback(db: Path = DB_PATH, tests: tuple[tuple[str, ...], ...] = (
        (sys.executable, "-m", "pytest", "-q", "tests/test_search.py", "tests/test_memory.py"),)) -> Result:
    """The same tests pass with both vector backends (they are parametrized), and on the real database the numpy
    backend returns the same first hit as sqlite-vec for the sampled terms."""
    from lecture_copilot.store.db import Store
    for cmd in tests:
        p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        if p.returncode != 0:
            return _fail("numpy_fallback", f"{' '.join(cmd[-2:])} exit {p.returncode}")
    pair = _two_lectures(db)
    if pair is None:
        return _fail("numpy_fallback", "fewer than two transcript lectures replayed")
    con, _, latest = pair
    con.close()
    agree, total = 0, 0
    a, b = Store(db, vec_backend="sqlite-vec"), Store(db, vec_backend="numpy")
    try:
        for r in a.items(latest["id"], kind="concept")[::7][:10]:
            total += 1
            ha, hb = a.search(r["text"], latest["course_id"], k=1), b.search(r["text"], latest["course_id"], k=1)
            agree += bool(ha) and bool(hb) and ha[0].id == hb[0].id
    finally:
        a.close()
        b.close()
    detail = f"tests pass with both backends; numpy agrees with sqlite-vec on {agree}/{total} first hits"
    return _ok("numpy_fallback", detail) if total and agree == total else _fail("numpy_fallback", detail)


CONTRADICTION_A = """WEBVTT

00:00:01.000 --> 00:00:40.000
נדבר על תקופת ההחזר. תקופת ההחזר המקובלת למיזם היא 18 חודשים. זה המספר שהמשקיעים מצפים לו.
"""
CONTRADICTION_B = """WEBVTT

00:00:01.000 --> 00:00:40.000
תיקון לשבוע שעבר: תקופת ההחזר המקובלת למיזם היא 12 חודשים, לא 18. המשקיעים מצפים ל-12.
"""


async def _contradiction_run(work: Path, client: httpx.AsyncClient) -> list[dict]:
    from lecture_copilot.asr.transcript import TranscriptASR
    from lecture_copilot.audio.transcript import TranscriptSource
    from lecture_copilot.pipeline import Ctx, run
    from lecture_copilot.store.db import Store, new_id

    store = Store(work / "gate_m3.sqlite")
    try:
        course = store.upsert_course("gate", language="he")
        claims = []
        async with client:
            lectures = (("a.vtt", CONTRADICTION_A, "2026-10-01"), ("b.vtt", CONTRADICTION_B, "2026-10-08"))
            for name, text, date in lectures:
                (work / name).write_text(text, encoding="utf-8")
                lid = store.upsert_lecture(course, audio_path=str(work / name), source="transcript", title=name,
                                           date=date, fact_check=False)
                ctx = Ctx(lecture_id=lid, course_id=course, course_name="gate", lecture_title=name,
                          profile=Profile(fact_check=False, language="he"), store=store, asr=TranscriptASR(),
                          ollama=client, run_id=new_id(), runs_dir=work)
                await run(TranscriptSource(lid, work / name, runs_dir=work), ctx)
                store.end_lecture(lid)
                claims.append(store.claims(lid))
        return claims
    finally:
        store.close()


def check_contradiction(work: Path | None = None, client: httpx.AsyncClient | None = None) -> Result:
    """Break it on purpose: lecture B says the opposite of lecture A; B's claim must gain +20 and link to A's."""
    with tempfile.TemporaryDirectory() as tmp:
        work = work or Path(tmp)
        a, b = asyncio.run(_contradiction_run(work, client or httpx.AsyncClient(base_url=OLLAMA_URL)))
    if not a:
        return _fail("contradiction", "lecture A produced no claim to contradict")
    linked = [c for c in b if c["contradicts_id"]]
    if not linked:
        return _fail("contradiction", f"lecture B's {len(b)} claim(s) were not marked as contradicting A "
                                      f"(A had {len(a)})")
    c = linked[0]
    before = c["importance"] - 20
    return _ok("contradiction", f"B's claim contradicts A's: importance {before} → {c['importance']}, linked")


# ---------- M4: verifier, ranker, eval ----------

RESULTS = ROOT / "eval" / "results.json"


def _results(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def check_verdict_accuracy(path: Path = RESULTS, min_accuracy: float = 0.8) -> Result:
    r = _results(path)
    if r is None:
        return _fail("verdict_accuracy", "no eval/results.json — python -m lecture_copilot.cli eval")
    v = r["verdicts"]
    detail = (f"{v['accuracy']:.0%} on {v['n']} claims, injected errors caught {v['injected_caught']}/{v['injected']}, "
              f"unchecked {r.get('unchecked', 0)} · labels: {r.get('labels_by', '?')}")
    return _ok("verdict_accuracy", detail) if v["accuracy"] >= min_accuracy else _fail("verdict_accuracy", detail)


def check_precision(path: Path = RESULTS, min_precision: float = 0.8) -> Result:
    """Precision@5 of 'material' on the ranked claims; a lecture with fewer than five labelled material claims
    is reported at its own k and not judged (D-M4-4)."""
    r = _results(path)
    if r is None:
        return _fail("precision_at_5", "no eval/results.json — python -m lecture_copilot.cli eval")
    judged = {d: p for d, p in r["precision_at_5"].items() if p["k"] >= 5}
    parts = [f"{d} {p['precision']:.0%} (k={p['k']})" for d, p in r["precision_at_5"].items()]
    detail = " · ".join(parts)
    detail += f" · judged: {', '.join(judged)}" if judged else " · no lecture with 5 material claims"
    if not judged:
        return _fail("precision_at_5", detail)
    avg = sum(p["precision"] for p in judged.values()) / len(judged)
    return _ok("precision_at_5", detail) if avg >= min_precision else _fail("precision_at_5", detail)


def check_cache_hit(path: Path = RESULTS) -> Result:
    r = _results(path)
    if r is None:
        return _fail("cache_hit", "no eval/results.json")
    return (_ok("cache_hit", "a repeated claim was answered from fact_cache without a call") if r.get("cache_hit")
            else _fail("cache_hit", "the repeated claim reached the network again"))


def check_cost(db: Path = DB_PATH) -> Result:
    con = sqlite3.connect(db)
    rows = con.execute("select lecture_id, sum(cost_usd), count(*) from decisions where node = 'net' "
                       "and lecture_id in (select id from lectures) group by lecture_id").fetchall()
    if not rows:
        return _fail("cost_per_lecture", "no outbound call logged for any lecture — replay or verify one")
    worst = max(rows, key=lambda r: r[1] or 0)
    detail = (f"{len(rows)} lecture(s) with calls; most expensive ${worst[1]:.4f} over {worst[2]} calls "
              f"(budget ${COST_BUDGET_USD})")
    return _ok("cost_per_lecture", detail) if worst[1] < COST_BUDGET_USD else _fail("cost_per_lecture", detail)


def check_ci_offline() -> Result:
    """CI never calls Gemini: tests do not import the SDK, the workflow has no key, .env is not tracked."""
    hits = subprocess.run(["git", "-C", str(ROOT), "grep", "-lE", r"^(from|import) google", "--", "tests"],
                          capture_output=True, text=True).stdout.split()
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    tracked = subprocess.run(["git", "-C", str(ROOT), "ls-files", ".env"], capture_output=True, text=True).stdout
    problems = []
    if hits:
        problems.append(f"tests import the SDK: {', '.join(hits)}")
    if "GEMINI" in ci:
        problems.append("ci.yml mentions GEMINI")
    if tracked.strip():
        problems.append(".env is tracked")
    if problems:
        return _fail("ci_offline", "; ".join(problems))
    return _ok("ci_offline", "tests use stubs only, ci.yml has no key, .env is not tracked")


OUTAGE_VTT = """WEBVTT

00:00:01.000 --> 00:00:40.000
נדבר על תקופת ההחזר. תקופת ההחזר המקובלת למיזם היא 18 חודשים. זה המספר שהמשקיעים מצפים לו.
"""


class _Flaky:
    """A verifier backend whose first call fails like a lost connection."""

    def __init__(self, backend):
        self.backend, self.calls = backend, 0

    async def generate(self, system, user):
        self.calls += 1
        if self.calls == 1:
            raise httpx.ConnectError("wifi off")
        return await self.backend.generate(system, user)


async def _outage_run(work: Path, client: httpx.AsyncClient, gemini) -> tuple[dict, list[dict]]:
    from lecture_copilot.agents.verifier import Verifier, VerifierWorker
    from lecture_copilot.asr.transcript import TranscriptASR
    from lecture_copilot.audio.transcript import TranscriptSource
    from lecture_copilot.pipeline import Ctx, run
    from lecture_copilot.store.db import Store, new_id
    from lecture_copilot.store.net import Net

    store = Store(work / "gate_m4.sqlite")
    try:
        course = store.upsert_course("gate", language="he")
        (work / "a.vtt").write_text(OUTAGE_VTT, encoding="utf-8")
        lid = store.upsert_lecture(course, audio_path=str(work / "a.vtt"), source="transcript", title="a",
                                   date="2026-10-01", fact_check=True)
        async with client:
            worker = VerifierWorker(Verifier(store, Net(store, backoff_s=0), _Flaky(gemini)), lid)
            ctx = Ctx(lecture_id=lid, course_id=course, course_name="gate", lecture_title="a",
                      profile=Profile(fact_check=True, language="he"), store=store, asr=TranscriptASR(),
                      ollama=client, run_id=new_id(), runs_dir=work, verifier=worker)
            summary = await run(TranscriptSource(lid, work / "a.vtt", runs_dir=work), ctx)
        log = [json.loads(o) for (o,) in store.con.execute(
            "select output_json from decisions where node = 'verifier' and lecture_id = ? order by ts", (lid,))]
        return summary, log
    finally:
        store.close()


def check_outage(work: Path | None = None, client: httpx.AsyncClient | None = None, gemini=None) -> Result:
    """Break it on purpose: the network drops on the first verification; the claim must go unchecked and be
    verified in the batch at stop."""
    with tempfile.TemporaryDirectory() as tmp:
        work = work or Path(tmp)
        if gemini is None:
            from lecture_copilot.agents.verifier import GeminiAPI
            from lecture_copilot.config import load_env
            load_env()
            gemini = GeminiAPI(os.environ["GEMINI_API_KEY"])
        summary, log = asyncio.run(_outage_run(work, client or httpx.AsyncClient(base_url=OLLAMA_URL), gemini))
    v = summary.get("verifier") or {}
    statuses = [row["status"] for row in log]
    detail = f"verifier log: {' → '.join(statuses) or 'empty'}; at stop: {v}"
    if "unchecked" not in statuses or not v.get("retried") or v.get("verified", 0) < 1 or v.get("unchecked"):
        return _fail("outage", detail)
    return _ok("outage", detail)


# ---------- M5: the live product ----------

MIC_CHECK = ROOT / "runs" / "m5" / "mic_check.json"
LIVE_MIN_S = 20 * 60


def _latest_web_run(db: Path) -> tuple[dict, dict | None] | None:
    con = sqlite3.connect(db)
    for lid, o in con.execute("select lecture_id, output_json from decisions where node = 'run' "
                              "and input_ref != 'resume' order by ts desc"):
        r = json.loads(o)
        if r.get("via") == "web" and (r.get("source_kind") == "mic" or r.get("pace") == "realtime"):
            d = con.execute("select output_json from decisions where node = 'digest' and lecture_id = ? "
                            "and input_ref like '%#digest' order by ts desc limit 1", (lid,)).fetchone()
            return r, json.loads(d[0]) if d else None
    return None


def check_live_simulation(db: Path = DB_PATH) -> Result:
    """A lecture of ≥ 20 minutes that came in live through the page (the mic, or a real-time file replay as the
    stand-in while the mic waits for Dor's permission): rows arrived, nothing failed, Digest within budget."""
    found = _latest_web_run(db)
    if found is None:
        return _fail("live_simulation", "no live lecture through the page (mic or real-time replay) — run copilot")
    r, d = found
    minutes = r["audio_s"] / 60
    failed = sum(v for k, v in r["status"].items() if k not in ("ok", "empty"))
    detail = (f"{minutes:.0f} min via the page ({r.get('source_kind')}, pace {r.get('pace')}), chunks {r['status']}, "
              f"Digest {d['total_s'] if d else '—'} s")
    if r["audio_s"] < LIVE_MIN_S or failed or d is None or d["total_s"] >= DIGEST_BUDGET_S or d["degraded"]:
        return _fail("live_simulation", detail)
    return _ok("live_simulation", detail)


def check_crash_resume(db: Path = DB_PATH) -> Result:
    con = sqlite3.connect(db)
    row = con.execute(
        "select r.lecture_id, r.output_json from decisions r join lecture_summaries s on s.lecture_id = r.lecture_id "
        "join lectures l on l.id = r.lecture_id where r.node = 'run' and r.input_ref = 'resume' "
        "and l.status = 'digested' order by r.ts desc limit 1").fetchone()
    if row is None:
        return _fail("crash_resume", "no lecture was resumed after an interruption and digested")
    o = json.loads(row[1])
    return _ok("crash_resume", f"an interrupted lecture was resumed from {o.get('chunks')} saved chunks and digested")


def check_mic_level(path: Path = MIC_CHECK) -> Result:
    """Dor runs `python -m lecture_copilot.cli miccheck` from his Terminal (that is where macOS asks for the
    microphone); it writes the peak level and when the meter first passed −40 dB."""
    if not path.is_file():
        return Result("mic_level", "SKIP", "needs runs/m5/mic_check.json — Dor: python -m lecture_copilot.cli "
                                           "miccheck (allows the microphone prompt)")
    r = json.loads(path.read_text(encoding="utf-8"))
    detail = f"peak {r['peak_db']} dB, −40 dB reached after {r['seconds_to_minus40']} s of {r['seconds']} s"
    if r["seconds_to_minus40"] is None or r["seconds_to_minus40"] > 10:
        return _fail("mic_level", detail)
    return _ok("mic_level", detail)


def check_course_html(courses_root: Path = COURSES_ROOT) -> Result:
    import re
    pages = sorted(Path(courses_root).glob("*/course.html"))
    if not pages:
        return _fail("course_html", f"no course.html under {courses_root}")
    page = pages[-1]
    html = page.read_text(encoding="utf-8")
    lectures_block = html.split('id="lectures"', 1)[1].split("</section>", 1)[0] if 'id="lectures"' in html else ""
    n_lectures = len(re.findall(r"<a\b", lectures_block))
    glossary = html.split('data-list="glossary"', 1)[1].split("</section>", 1)[0] if 'data-list="glossary"' in html \
        else ""
    n_glossary = len(re.findall(r'class="row', glossary))
    detail = f"{page.parent.name}: {n_lectures} lectures, {n_glossary} glossary rows"
    if '<html lang="he" dir="rtl">' not in html or n_lectures < 2 or n_glossary < 2:
        return _fail("course_html", detail)
    return _ok("course_html", detail)


def check_page_rtl(img_dir: Path = IMG_DIR) -> Result:
    html = (ROOT / "lecture_copilot" / "web" / "index.html").read_text(encoding="utf-8")
    problems = []
    if '<html lang="he" dir="rtl">' not in html:
        problems.append("index.html is not lang=he dir=rtl")
    for word in ("alert(", "Notification", "confirm(", "<audio", ".play()"):
        if word in html:
            problems.append(f"index.html contains {word} — nothing may pop up")
    missing = [n for n in ("m5_before.jpg", "m5_during.jpg", "m5_after.jpg")
               if not (img_dir / n).is_file() or (img_dir / n).stat().st_size < 1000]
    if missing:
        problems.append(f"no browser screenshot: {', '.join(missing)}")
    if problems:
        return _fail("page_rtl", "; ".join(problems))
    return _ok("page_rtl", "lang=he dir=rtl, nothing pops up; screenshots before / during / after")


# ---------- runner ----------

def _stage0() -> dict:
    return json.loads(STAGE0.read_text(encoding="utf-8")) if STAGE0.exists() else {}


# ---------- M6: Notion ----------

def _notion(env=None, client=None):
    """(sink, ids) from `.env`, or a Result saying why not: SKIP without a token (Dor's step), FAIL without
    the databases (`cli notion-init` was not run)."""
    from lecture_copilot.cli import _env_from
    from lecture_copilot.output.notion import NotionAPI, NotionIds, NotionSink
    env = _env_from(ROOT / ".env") if env is None else env
    if not env.get("NOTION_TOKEN"):
        return Result("notion", "SKIP", "NOTION_TOKEN is empty — Dor: integration token into .env, share the root "
                                        "page, then `python -m lecture_copilot.cli notion-init` (PLAN §7)")
    ids = NotionIds.from_env(env)
    if ids is None:
        return _fail("notion", "no database ids in .env — run `python -m lecture_copilot.cli notion-init`")
    return NotionSink(NotionAPI(env["NOTION_TOKEN"], client=client), None, ids), ids


def _synced_lecture(con: sqlite3.Connection) -> tuple[str, str] | None:
    """The latest lecture whose Notion sink wrote (lecture_id, course_id)."""
    row = con.execute("select d.lecture_id, l.course_id from decisions d join lectures l on l.id = d.lecture_id "
                      "where d.node = 'sink' and d.output_json like '%\"NotionSink\"%' and d.output_json like "
                      "'%\"ok\"%' and d.input_ref like '%:write_lecture' order by d.ts desc limit 1").fetchone()
    return (row[0], row[1]) if row else None


async def _rows(sink, ds: str, prefix: str, store) -> list[dict]:
    """Every row whose key starts with `prefix`, raw (duplicates included — that is what the gate looks for)."""
    sink.bind(store)
    out, cursor = [], None
    while True:
        body = {"filter": {"property": "key", "rich_text": {"starts_with": prefix}}, "page_size": 100}
        if cursor:
            body["start_cursor"] = cursor
        res, _ = await sink.api.request("POST", f"/v1/data_sources/{ds}/query", body)
        out += res["results"]
        if not res.get("has_more"):
            return out
        cursor = res["next_cursor"]


def _key(page: dict) -> str:
    return "".join(t.get("plain_text", "") for t in page["properties"].get("key", {}).get("rich_text", []))


def check_notion_init(env=None, client=None) -> Result:
    from lecture_copilot.output.notion import ENV_KEYS
    got = _notion(env, client)
    if isinstance(got, Result):
        return Result("notion_init", got.status, got.detail)
    sink, ids = got
    import asyncio

    async def probe():
        missing = []
        for name in ENV_KEYS:
            try:
                await sink.api.request("GET", f"/v1/data_sources/{getattr(ids, name)}")
            except Exception as e:  # noqa: BLE001 — any answer but 200 means the database is not usable
                missing.append(f"{name} ({type(e).__name__})")
        return missing
    missing = asyncio.run(probe())
    if missing:
        return _fail("notion_init", f"data sources that do not answer: {', '.join(missing)} — run notion-init again")
    return _ok("notion_init", "the five data sources answer (courses, lectures, glossary, claims, tasks)")


def check_notion_lecture_page(db: Path = DB_PATH, env=None, client=None) -> Result:
    import asyncio

    from lecture_copilot.output.digest import SECTIONS, section_headings
    from lecture_copilot.store.db import Store
    got = _notion(env, client)
    if isinstance(got, Result):
        return Result("notion_lecture_page", got.status, got.detail)
    sink, ids = got
    store = Store(db)
    try:
        found = _synced_lecture(store.con)
        if found is None:
            return _fail("notion_lecture_page", "no lecture was synced to Notion yet (NotionSink ok row)")
        lid, _ = found

        async def go():
            pages = [p for p in await _rows(sink, ids.lectures, lid, store) if _key(p) == lid]
            if not pages:
                return None
            res, _ = await sink.api.request("GET", f"/v1/pages/{pages[0]['id']}/markdown")
            return res["markdown"]
        md = asyncio.run(go())
    finally:
        store.close()
    if md is None:
        return _fail("notion_lecture_page", f"lecture {lid} has no row in the Lectures database")
    heads = section_headings(md)
    detail = f"lecture {lid}: {len(heads)} sections in Notion ({', '.join(heads[:3])}…)"
    return _ok("notion_lecture_page", detail) if heads == SECTIONS else _fail("notion_lecture_page", detail)


def check_notion_glossary(db: Path = DB_PATH, env=None, client=None) -> Result:
    import asyncio

    from lecture_copilot.store.db import Store
    got = _notion(env, client)
    if isinstance(got, Result):
        return Result("notion_glossary", got.status, got.detail)
    sink, ids = got
    store = Store(db)
    try:
        found = _synced_lecture(store.con)
        if found is None:
            return _fail("notion_glossary", "no lecture was synced to Notion yet")
        _, course_id = found
        synced = [r[0] for r in store.con.execute(
            "select distinct d.lecture_id from decisions d join lectures l on l.id = d.lecture_id where l.course_id = ? "
            "and d.node = 'sink' and d.output_json like '%\"NotionSink\"%' and d.output_json like '%\"ok\"%' "
            "and d.input_ref like '%:write_lecture'", (course_id,))]
        keys = {r[0] for r in store.con.execute(
            f"select distinct canonical_key from items where kind = 'concept' and canonical_key is not null "
            f"and lecture_id in ({','.join('?' * len(synced))})", synced)}
        rows = asyncio.run(_rows(sink, ids.glossary, f"{course_id}:", store))
    finally:
        store.close()
    notion_keys = [_key(p).split(":", 1)[1] for p in rows]
    all_rows = len(rows)
    detail = f"course {course_id}: {all_rows} rows in Notion, {len(keys)} distinct canonical keys in {len(synced)} " \
             f"synced lectures"
    if set(notion_keys) != keys or all_rows != len(keys):
        return _fail("notion_glossary", detail + f" (missing {sorted(keys - set(notion_keys))[:5]}, "
                                                 f"extra {sorted(set(notion_keys) - keys)[:5]})")
    return _ok("notion_glossary", detail)


def check_notion_resync(db: Path = DB_PATH, env=None, client=None) -> Result:
    """Sync the latest synced lecture again from its saved Digest: every database must keep its row count and the
    lecture page must keep its id."""
    import asyncio

    from lecture_copilot.cli import write_sink
    from lecture_copilot.output.digest import saved_digest
    from lecture_copilot.store.db import Store
    got = _notion(env, client)
    if isinstance(got, Result):
        return Result("notion_resync", got.status, got.detail)
    sink, ids = got
    store = Store(db)
    try:
        found = _synced_lecture(store.con)
        if found is None:
            return _fail("notion_resync", "no lecture was synced to Notion yet")
        lid, course_id = found
        doc = saved_digest(lid, store)
        if doc is None:
            return _fail("notion_resync", f"lecture {lid} has no saved Digest")

        async def counts():
            return {"lectures": len(await _rows(sink, ids.lectures, lid, store)),
                    "glossary": len(await _rows(sink, ids.glossary, f"{course_id}:", store)),
                    "claims": len(await _rows(sink, ids.claims, f"{lid}:", store)),
                    "tasks": len(await _rows(sink, ids.tasks, f"{lid}:", store)),
                    "page": [p["id"] for p in await _rows(sink, ids.lectures, lid, store)]}
        before = asyncio.run(counts())
        _, log = asyncio.run(write_sink(sink, "write_lecture", doc, store, lid))
        after = asyncio.run(counts())
    finally:
        store.close()
    if log["status"] != "ok":
        return _fail("notion_resync", f"second sync {log['status']}: {log.get('error', log.get('reason'))}")
    detail = ", ".join(f"{k} {before[k]}→{after[k]}" for k in ("lectures", "glossary", "claims", "tasks"))
    if before != after:
        return _fail("notion_resync", f"rows changed on re-sync: {detail}")
    return _ok("notion_resync", f"re-sync of {lid} left every count unchanged ({detail}), {log['calls']} calls")


def check_notion_skipped(db: Path = DB_PATH) -> Result:
    con = sqlite3.connect(db)
    row = con.execute("select lecture_id, output_json from decisions where node = 'sink' and output_json like "
                      "'%\"NotionSink\"%' and output_json like '%\"skipped\"%' order by ts desc limit 1").fetchone()
    if row is None:
        return _fail("notion_skipped", "no run ever skipped the Notion sink (start one with NOTION_TOKEN empty)")
    out = json.loads(row[1])
    folder = con.execute("select output_json from decisions where node = 'sink' and lecture_id = ? and output_json "
                         "like '%\"FolderSink\"%' and output_json like '%\"ok\"%'", (row[0],)).fetchone()
    if not out.get("reason") or folder is None:
        return _fail("notion_skipped", f"skipped row without a reason or without the folder write: {out}")
    return _ok("notion_skipped", f"lecture {row[0]}: Notion skipped — \"{out['reason']}\"; the folder sink wrote")


CHECKS: dict[int, list[tuple[str, Callable[[], Result]]]] = {
    0: [
        ("mw_version", check_mw_version),
        ("ollama_models", check_ollama_models),
        ("sqlite_vec", lambda: check_sqlite_vec(_stage0())),
        ("fixture", check_fixture),
        ("mw_bench", lambda: check_mw_bench(_stage0())),
        ("extract_bench", lambda: check_extract_bench(_stage0())),
        ("mic_seat", lambda: check_mic(_stage0())),
        ("secret_scan", check_secret_scan),
    ],
    6: [
        ("notion_init", check_notion_init),
        ("notion_lecture_page", check_notion_lecture_page),
        ("notion_glossary", check_notion_glossary),
        ("notion_resync", check_notion_resync),
        ("notion_skipped", check_notion_skipped),
        ("prompt_cache", check_prompt_cache),
        ("tests", check_tests),
        ("secret_scan", check_secret_scan),
    ],
    5: [
        ("live_simulation", check_live_simulation),
        ("crash_resume", check_crash_resume),
        ("mic_level", check_mic_level),
        ("course_html", check_course_html),
        ("page_rtl", check_page_rtl),
        ("prompt_cache", check_prompt_cache),
        ("tests", check_tests),
        ("secret_scan", check_secret_scan),
    ],
    4: [
        ("verdict_accuracy", check_verdict_accuracy),
        ("precision_at_5", check_precision),
        ("outage", check_outage),
        ("cost_per_lecture", check_cost),
        ("cache_hit", check_cache_hit),
        ("ci_offline", check_ci_offline),
        ("prompt_cache", check_prompt_cache),
        ("tests", check_tests),
        ("secret_scan", check_secret_scan),
    ],
    3: [
        ("shared_concepts", check_shared_concepts),
        ("contradiction", check_contradiction),
        ("search_top1", check_search_top1),
        ("continuation", check_continuation),
        ("memory_budget", check_memory_budget),
        ("numpy_fallback", check_numpy_fallback),
        ("prompt_cache", check_prompt_cache),
        ("tests", check_tests),
        ("secret_scan", check_secret_scan),
    ],
    2: [
        ("digest_sections", check_digest_sections),
        ("digest_time", check_digest_time),
        ("vtt_replay", check_vtt_replay),
        ("course_folder", check_course_folder),
        ("html_rtl", check_html_rtl),
        ("prompt_cache", check_prompt_cache),
        ("tests", check_tests),
        ("secret_scan", check_secret_scan),
    ],
    1: [
        ("fixture_replay", check_fixture_replay),
        ("rerun_upsert", check_rerun_upsert),
        ("chunk_budget", check_chunk_budget),
        ("asr_error", check_asr_error),
        ("peak_memory", check_peak_memory),
        ("prompt_cache", check_prompt_cache),
        ("tests", check_tests),
        ("secret_scan", check_secret_scan),
    ],
}


def summary(m: int, results: list[Result]) -> str:
    passed = sum(r.status == "PASS" for r in results)
    if any(r.status == "FAIL" for r in results):
        verdict = "FAIL"
    elif passed < len(results):
        verdict = "SKIP"
    else:
        verdict = "PASS"
    return f"GATE M{m}: {verdict} {passed}/{len(results)}"


def run(m: int) -> list[Result]:
    out = []
    for name, check in CHECKS[m]:
        try:
            r = check()
        except Exception as e:
            r = _fail(name, f"crashed: {type(e).__name__}: {e}")
        out.append(r)
        print(f"  {r.status:<4}  {r.name:<14} {r.detail}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="gate")
    ap.add_argument("--m", type=int, required=True)
    m = ap.parse_args(argv).m
    if m not in CHECKS:
        print(f"no gate defined for M{m} yet", file=sys.stderr)
        return 2
    line = summary(m, run(m))
    print(line)
    return 0 if line.startswith(f"GATE M{m}: PASS") else 1


if __name__ == "__main__":
    sys.exit(main())
