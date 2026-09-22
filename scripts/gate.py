"""Milestone gate: `python scripts/gate.py --m N` prints one line per check and `GATE M<N>: PASS k/k`.

Checks read real state (binaries, the Ollama server, files, eval/stage0.json). A crashing check is a FAIL,
never a crash of the gate. The secret scan reports file names only — it never prints key material.
"""

import argparse
import json
import math
import sqlite3
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import httpx

from lecture_copilot.config import DIGEST_MODEL, EMBED_MODEL, LIVE_MODEL, OLLAMA_URL, ROOT

STAGE0 = ROOT / "eval" / "stage0.json"
FIXTURE = ROOT / "eval" / "fixture_10min.m4a"
SECRET_PATTERN = r"AIza[0-9A-Za-z_-]{30,}|ntn_[A-Za-z0-9]{20,}|secret_[A-Za-z0-9]{20,}"


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
                        required: tuple[str, ...] = (LIVE_MODEL, DIGEST_MODEL, EMBED_MODEL)) -> Result:
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


def check_secret_scan(root: Path = ROOT) -> Result:
    p = subprocess.run(["git", "-C", str(root), "grep", "--untracked", "-I", "-l", "-iE", SECRET_PATTERN],
                       capture_output=True, text=True)
    if p.returncode == 1:
        return _ok("secret_scan", "no key material in tracked or untracked files")
    if p.returncode == 0:
        return _fail("secret_scan", "key material in: " + ", ".join(p.stdout.split()))
    return _fail("secret_scan", f"git grep failed: {p.stderr.strip()[:120]}")


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
