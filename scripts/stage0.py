"""Stage-0 measurements (DESIGN_HE §checks). Every number lands in eval/stage0.json; the Hebrew page
docs/notes/STAGE0_HE.html reads them through <span data-metric="a.b.c"> placeholders (`report`), never typed.

    python scripts/stage0.py sqlite
    python scripts/stage0.py cut data/lectures/<7-6 lecture file>
    python scripts/stage0.py mw-bench [--busy]
    python scripts/stage0.py extract-bench
    python scripts/stage0.py mic data/mic_seat.m4a
    python scripts/stage0.py mic-verdict yes|no
    python scripts/stage0.py report

Raw transcripts and model outputs stay in runs/stage0/ (gitignored): they are course material.
"""

import argparse
import asyncio
import json
import math
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import httpx

from lecture_copilot import prompts
from lecture_copilot.agents.schemas import ExtractResult
from lecture_copilot.config import BUDGET_S, DIGEST_MODEL, LIVE_MODEL, OLLAMA_URL, ROOT
from lecture_copilot.llm import chat_json

RESULTS = ROOT / "eval" / "stage0.json"
WORK = ROOT / "runs" / "stage0"
REPORT = ROOT / "docs" / "notes" / "STAGE0_HE.html"
MW_TIMEOUT_S = BUDGET_S["asr"] * 3  # PLAN §3.3: a run longer than this is a hang
BENCH_COURSE, BENCH_LECTURE, BENCH_LANGUAGE = "AI Developers — Python", "7/6 Lecture", "he"


# ---------- pure helpers (tested) ----------

def percentile(xs: list[float], p: float) -> float:
    s = sorted(xs)
    return s[max(0, math.ceil(p / 100 * len(s)) - 1)]


def mw_verdict(runs_s: list[float | None], budget_s: float) -> dict:
    """runs_s[0] is the cold run; None is a hang (timed out). hot = the median run fits the per-chunk ASR budget."""
    if len(runs_s) < 5:
        raise ValueError("need 5 runs")
    done = [r for r in runs_s if r is not None]
    hangs = len(runs_s) - len(done)
    p50 = percentile(done, 50) if done else float("inf")
    hot = p50 <= budget_s
    return {
        "runs_s": runs_s, "cold_s": runs_s[0], "p50_s": p50, "hangs": hangs, "budget_s": budget_s,
        "verdict": "hot" if hot else "cold", "decision": "mw" if hot and hangs == 0 else "mlx",
    }


def hebrew_share(text: str) -> float:
    letters = [c for c in text if c.isalpha()]
    return sum("֐" <= c <= "׿" for c in letters) / len(letters) if letters else 0.0


def cut_offsets(duration_s: float, n: int, chunk_s: float, margin_s: float) -> list[float]:
    end = duration_s - margin_s - chunk_s
    if n * chunk_s > duration_s - 2 * margin_s:
        raise ValueError(f"{duration_s:.0f} s is too short for {n} × {chunk_s} s chunks with {margin_s} s margins")
    step = (end - margin_s) / (n - 1)
    return [round(margin_s + i * step, 1) for i in range(n)]


def load_results(path: Path = RESULTS) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def update_results(path: Path, key: str, value: object) -> None:
    data = load_results(path)
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


def summarize_extract(calls: list[dict]) -> dict:
    ms = [c["ms"] for c in calls]
    return {
        "n": len(calls),
        "valid_first": sum(c["first_valid"] for c in calls),
        "valid_after_retry": sum(c["valid"] for c in calls),
        "p50_ms": percentile(ms, 50), "p95_ms": percentile(ms, 95),
        "hebrew_summaries": sum(c["hebrew"] >= 0.5 for c in calls),
        "concepts": sum(c["concepts"] for c in calls), "claims": sum(c["claims"] for c in calls),
    }


_METRIC = re.compile(r'(<span data-metric="([^"]+)">)(.*?)(</span>)', re.DOTALL)


def _lookup(results: dict, path: str) -> object:
    node: object = results
    for part in path.split("."):
        if not isinstance(node, dict) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


def _fmt(v: object) -> str:
    if isinstance(v, float):
        return str(round(v, 2))
    if v is None:
        return "—"
    return str(v)


def fill_metrics(html: str, results: dict) -> str:
    return _METRIC.sub(lambda m: m.group(1) + _fmt(_lookup(results, m.group(2))) + m.group(4), html)


# ---------- providers (run for real, not in tests) ----------

def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ffmpeg_cut(src: Path, dst: Path, start_s: float, dur_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start_s), "-t", str(dur_s), "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", str(dst)], check=True)


def mw_cmd(audio: Path, out_dir: Path, language: str) -> list[str]:
    return ["mw", "transcribe", str(audio), "--format", "json", "--language", language, "-o", str(out_dir)]


def mw_text(out_dir: Path) -> str:
    files = sorted(out_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"mw wrote no json into {out_dir}")
    data = json.loads(files[-1].read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return " ".join(s.get("text", "").strip() for s in data["segments"]).strip()
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return data["text"].strip()
    raise RuntimeError(f"unknown mw json shape: {list(data)[:8] if isinstance(data, dict) else type(data)}")


def mw_run(audio: Path, out_dir: Path, language: str = BENCH_LANGUAGE) -> float | None:
    """Seconds for one `mw transcribe`, or None on a hang (timeout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.json"):
        old.unlink()
    t0 = time.perf_counter()
    try:
        subprocess.run(mw_cmd(audio, out_dir, language), capture_output=True, text=True, check=True,
                       timeout=MW_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return None
    return round(time.perf_counter() - t0, 2)


# ---------- subcommands ----------

def cmd_sqlite(_: argparse.Namespace) -> None:
    con = sqlite3.connect(":memory:")
    res = {"sqlite": sqlite3.sqlite_version, "python": sys.version.split()[0],
           "enable_load_extension": hasattr(con, "enable_load_extension"), "loaded": False, "vec_version": None,
           "fallback": None}
    try:
        import sqlite_vec
        con.enable_load_extension(True)
        sqlite_vec.load(con)
        con.enable_load_extension(False)
        res["vec_version"] = con.execute("select vec_version()").fetchone()[0]
        res["loaded"] = True
    except Exception as e:  # any failure means the numpy fallback from day one
        res["fallback"], res["error"] = "numpy", f"{type(e).__name__}: {e}"
    update_results(RESULTS, "sqlite_vec", res)
    print(json.dumps(res, indent=2))


def cmd_cut(a: argparse.Namespace) -> None:
    src = Path(a.lecture)
    dur = ffprobe_duration(src)
    ffmpeg_cut(src, WORK / "bench_45s.wav", a.bench_at, 45)
    offs = cut_offsets(dur, n=10, chunk_s=45, margin_s=300)
    for i, off in enumerate(offs, 1):
        ffmpeg_cut(src, WORK / "chunks" / f"chunk_{i:02d}.wav", off, 45)
    update_results(RESULTS, "cut", {"source": src.name, "duration_s": round(dur, 1), "bench_at_s": a.bench_at,
                                    "chunk_offsets_s": offs})
    print(f"bench_45s.wav + 10 chunks from {src.name} ({dur / 60:.0f} min)")


def cmd_mw_bench(a: argparse.Namespace) -> None:
    wav, out = WORK / "bench_45s.wav", WORK / "mw_out"
    res = load_results().get("mw", {})
    if a.busy:
        res["busy_s"] = mw_run(wav, out)
        print(f"busy run: {res['busy_s']} s")
    else:
        runs = []
        for i in range(5):
            runs.append(mw_run(wav, out))
            print(f"run {i + 1}: {runs[-1]} s")
        res.update(mw_verdict(runs, BUDGET_S["asr"]))
        dirs = [out / f"p{i}" for i in range(2)]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        procs = [subprocess.Popen(mw_cmd(wav, d, BENCH_LANGUAGE), stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) for d in dirs]
        codes = [p.wait(timeout=MW_TIMEOUT_S * 2) for p in procs]
        res["parallel2_wall_s"] = round(time.perf_counter() - t0, 2)
        res["parallel2_exit"] = codes
        res["words_45s"] = len(mw_text(out).split())
        print(f"2 in parallel: {res['parallel2_wall_s']} s wall, exit {codes}")
    update_results(RESULTS, "mw", res)
    print(json.dumps({k: v for k, v in res.items() if k != "runs_s"}, indent=2))


async def _extract_model(model: str, texts: list[str], client: httpx.AsyncClient) -> tuple[dict, list[dict]]:
    prompt = prompts.load("extract_v0")
    think = False if model == LIVE_MODEL else None

    def render(i: int, text: str) -> tuple[str, str]:
        return prompt.render(language=BENCH_LANGUAGE, course_name=BENCH_COURSE, lecture_title=BENCH_LECTURE,
                             idx=i, t0=0, t1=45, known_terms="(none yet)", previous_chunk_summary="", text=text)

    await chat_json(model, *render(0, texts[0]), ExtractResult, client=client, think=think, timeout_s=180)  # warm-up
    rows, raw = [], []
    for i, text in enumerate(texts, 1):
        call = await chat_json(model, *render(i, text), ExtractResult, client=client, think=think, timeout_s=180)
        v = call.value
        rows.append({"first_valid": call.first_valid, "valid": v is not None, "ms": round(call.ms[0]),
                     "hebrew": hebrew_share(v.chunk_summary) if v else 0.0,
                     "concepts": len(v.concepts) if v else 0, "claims": len(v.claims) if v else 0})
        raw.append({"chunk": i, "attempts": call.attempts, "error": call.error, "raw": call.raw,
                    "tokens_in": call.tokens_in, "tokens_out": call.tokens_out})
        print(f"{model} chunk {i:02d}: valid_first={call.first_valid} {call.ms[0] / 1000:.1f} s")
    await client.post("/api/generate", json={"model": model, "keep_alive": 0})  # unload before the next model
    return summarize_extract(rows), raw


def cmd_extract_bench(_: argparse.Namespace) -> None:
    chunks = sorted((WORK / "chunks").glob("chunk_*.wav"))
    if len(chunks) < 10:
        sys.exit("run `stage0.py cut <lecture>` first")
    texts, asr_s = [], []
    for wav in chunks:  # transcribe everything first: never an LLM loaded while Whisper runs
        out = WORK / "chunks" / wav.stem
        asr_s.append(mw_run(wav, out))
        texts.append(mw_text(out))
        (WORK / "chunks" / f"{wav.stem}.txt").write_text(texts[-1] + "\n", encoding="utf-8")
    res = {"asr_p50_s": percentile([s for s in asr_s if s], 50), "words_per_chunk_p50": percentile(
        [len(t.split()) for t in texts], 50)}

    async def go() -> None:
        async with httpx.AsyncClient(base_url=OLLAMA_URL) as client:
            for model in (LIVE_MODEL, DIGEST_MODEL):
                res[model], raw = await _extract_model(model, texts, client)
                with (WORK / f"extract_{model.replace(':', '_')}.jsonl").open("w", encoding="utf-8") as f:
                    f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in raw)

    asyncio.run(go())
    update_results(RESULTS, "extract", res)
    print(json.dumps(res, indent=2))


def cmd_mic(a: argparse.Namespace) -> None:
    src = Path(a.file)
    out = WORK / "mic"
    secs = mw_run(src, out)
    text = mw_text(out)
    (WORK / "mic_seat.txt").write_text(text + "\n", encoding="utf-8")
    res = {**load_results().get("mic", {}), "file": str(src.relative_to(ROOT)) if src.is_absolute() else str(src),
           "duration_s": round(ffprobe_duration(src), 1), "asr_s": secs, "words": len(text.split())}
    res.setdefault("readable", None)
    update_results(RESULTS, "mic", res)
    print(f"{res['words']} words → runs/stage0/mic_seat.txt — read it, then: stage0.py mic-verdict yes|no")


def cmd_mic_verdict(a: argparse.Namespace) -> None:
    res = load_results().get("mic")
    if not res:
        sys.exit("run `stage0.py mic <file>` first")
    res["readable"] = a.readable
    update_results(RESULTS, "mic", res)
    print(f"mic readable: {a.readable}")


def cmd_report(_: argparse.Namespace) -> None:
    html = REPORT.read_text(encoding="utf-8")
    REPORT.write_text(fill_metrics(html, load_results()), encoding="utf-8", newline="")
    print(f"filled {len(_METRIC.findall(html))} metrics in {REPORT.relative_to(ROOT)}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="stage0")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sqlite").set_defaults(fn=cmd_sqlite)
    p = sub.add_parser("cut")
    p.add_argument("lecture")
    p.add_argument("--bench-at", type=float, default=1200.0, help="start of the 45 s mw benchmark clip (s)")
    p.set_defaults(fn=cmd_cut)
    p = sub.add_parser("mw-bench")
    p.add_argument("--busy", action="store_true", help="one run while the MacWhisper app is transcribing")
    p.set_defaults(fn=cmd_mw_bench)
    sub.add_parser("extract-bench").set_defaults(fn=cmd_extract_bench)
    p = sub.add_parser("mic")
    p.add_argument("file")
    p.set_defaults(fn=cmd_mic)
    p = sub.add_parser("mic-verdict")
    p.add_argument("readable", choices=["yes", "no"])
    p.set_defaults(fn=cmd_mic_verdict)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
