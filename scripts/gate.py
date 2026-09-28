"""Milestone gate: `python scripts/gate.py --m N` prints one line per check and `GATE M<N>: PASS k/k`.

Checks read real state (binaries, the Ollama server, files, eval/stage0.json). A crashing check is a FAIL,
never a crash of the gate. The secret scan reports file names only — it never prints key material.
"""

import argparse
import asyncio
import json
import math
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
    DB_PATH,
    DIGEST_MODEL,
    EMBED_MODEL,
    LIVE_MODEL,
    OLLAMA_URL,
    ROOT,
    Profile,
)

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
    lectures, runs, con = _fixture_runs(db, fixture)
    if len(lectures) != 1:
        return _fail("rerun_upsert", f"{len(lectures)} lecture rows for {fixture.name}, expected 1")
    if len(runs) < 2:
        return _fail("rerun_upsert", f"{len(runs)} complete replay(s) — {REPLAY_HINT}")
    a, b = runs[-2]["counts"], runs[-1]["counts"]
    if a != b:
        diff = ", ".join(f"{k} {a[k]}→{b[k]}" for k in a if a[k] != b[k])
        return _fail("rerun_upsert", f"last two replays differ: {diff}")
    table = _table_counts(con, lectures[0])
    if table != b:
        return _fail("rerun_upsert", f"tables hold {table}, the last replay wrote {b}")
    return _ok("rerun_upsert", f"1 lecture, {len(runs)} replays, last two identical: {b}")


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
    procs = ", ".join(f"{k} {v / 1024:.1f} GB" for k, v in sorted(m["procs_peak_mb"].items(), key=lambda kv: -kv[1]))
    pressure = PRESSURE.get(m["max_pressure"], str(m["max_pressure"]))
    detail = (f"peak {m['peak_used_gb']} of {m['total_gb']} GB, pressure {pressure}, swap {m['swap_growth_mb']:+.0f} MB"
              f" · {procs}")
    if missing:
        return _fail("peak_memory", f"not resident at the peak: {', '.join(missing)} — {detail}")
    if m["max_pressure"] > 1:
        return _fail("peak_memory", detail)
    return _ok("peak_memory", detail)


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
    detail = (f"corrupt chunk → {first.get('status')} ({first.get('error', '')[:70]}); "
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


# ---------- runner ----------

def _stage0() -> dict:
    return json.loads(STAGE0.read_text(encoding="utf-8")) if STAGE0.exists() else {}


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
    1: [
        ("fixture_replay", check_fixture_replay),
        ("rerun_upsert", check_rerun_upsert),
        ("chunk_budget", check_chunk_budget),
        ("asr_error", check_asr_error),
        ("peak_memory", check_peak_memory),
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
