"""Stage-0 measurements (DESIGN_HE §checks). Every number lands in eval/stage0.json; the Hebrew page
docs/notes/STAGE0_HE.html reads them through data-metric="a.b.c" placeholders (`report`), never typed.

    python scripts/stage0.py sqlite
    python scripts/stage0.py cut data/lectures/<7-6 lecture file>
    python scripts/stage0.py mw-bench [--busy] [--wav clip.wav --model engine:id --key mw_tts]
    python scripts/stage0.py extract-bench
    python scripts/stage0.py mic data/mic_seat.m4a [--model engine:id]
    python scripts/stage0.py mic-verdict yes|no
    python scripts/stage0.py report [--page docs/notes/<page>_HE.html]

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
import numpy as np

from lecture_copilot import prompts
from lecture_copilot.agents.schemas import ExtractResult
from lecture_copilot.config import BUDGET_S, DIGEST_MODEL, LIVE_MODEL, MW_BIN, MW_MODEL, OLLAMA_URL, ROOT, SILENCE_DB
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
    return sum("\u0590" <= c <= "\u05ff" for c in letters) / len(letters) if letters else 0.0


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


def segment_coverage(segments: list[dict]) -> float:
    """Seconds of audio covered by mw segments (start/end in ms)."""
    return round(sum(seg["end"] - seg["start"] for seg in segments) / 1000, 1)


def level_stats(samples: np.ndarray, sr: int, floor_db: float, frame_s: float = 0.25) -> dict:
    """Level distribution over 250 ms frames — the same window the live meter uses."""
    f = int(sr * frame_s)
    frames = [samples[i:i + f] for i in range(0, len(samples) - f + 1, f)]
    db = np.array([20 * np.log10(np.sqrt(np.mean(np.square(fr, dtype=np.float64))) + 1e-12) for fr in frames])
    p10, p50, p90 = (round(float(v), 1) for v in np.percentile(db, [10, 50, 90]))
    return {"p10_db": p10, "p50_db": p50, "p90_db": p90, "floor_db": floor_db,
            "share_above_floor": round(float(np.mean(db > floor_db)), 2)}


# any element carrying data-metric, attributes in any order: <td class="num" data-metric="a.b">…</td>
_METRIC = re.compile(r'(<(\w+)\b[^>]*\bdata-metric="([^"]+)"[^>]*>)(.*?)(</\2>)', re.DOTALL)


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
    return _METRIC.sub(lambda m: m.group(1) + _fmt(_lookup(results, m.group(3))) + m.group(5), html)


# ---------- providers (run for real, not in tests) ----------

def ffprobe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def ffmpeg_cut(src: Path, dst: Path, start_s: float, dur_s: float) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss", str(start_s), "-t", str(dur_s), "-i", str(src),
                    "-vn", "-ac", "1", "-ar", "16000", str(dst)], check=True)


def decode_pcm(path: Path, sr: int = 16000) -> np.ndarray:
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                         capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype=np.float32)


def mw_cmd(audio: Path, out_json: Path, language: str, model: str = MW_MODEL, binary: str = MW_BIN) -> list[str]:
    """mw 14.7: `-o` is one output file; `--model` defaults to the app's selection, so it is always passed."""
    return [binary, "transcribe", str(audio), "--model", model, "--language", language, "--format", "json",
            "--no-speakers", "-o", str(out_json), "--overwrite"]


def mw_text(out_json: Path) -> str:
    """mw 14.7 JSON: {"text": str, "segments": [{"id", "start", "end" (ms, int), "text", "words": [...]}]}."""
    if not out_json.exists():
        raise RuntimeError(f"mw wrote no json at {out_json}")
    data = json.loads(out_json.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        return " ".join(s.get("text", "").strip() for s in data["segments"]).strip()
    if isinstance(data, dict) and isinstance(data.get("text"), str):
        return data["text"].strip()
    raise RuntimeError(f"unknown mw json shape: {list(data)[:8] if isinstance(data, dict) else type(data)}")


def mw_run(audio: Path, out_json: Path, language: str = BENCH_LANGUAGE, model: str = MW_MODEL) -> float | None:
    """Seconds for one `mw transcribe`, or None on a hang (timeout)."""
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.unlink(missing_ok=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(mw_cmd(audio, out_json, language, model), capture_output=True, text=True, check=True,
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
    wav, out = Path(a.wav), WORK / "mw_out"
    res = load_results().get(a.key, {})
    res.update({"wav": wav.name, "model": a.model})
    if a.busy:
        res["busy_s"] = mw_run(wav, out / "busy.json", model=a.model)
        print(f"busy run: {res['busy_s']} s")
    else:
        runs = []
        for i in range(5):
            runs.append(mw_run(wav, out / "run.json", model=a.model))
            print(f"run {i + 1}: {runs[-1]} s")
        res.update(mw_verdict(runs, BUDGET_S["asr"]))
        res["words"] = len(mw_text(out / "run.json").split())
        outs = [out / f"p{i}.json" for i in range(2)]
        t0 = time.perf_counter()
        procs = [subprocess.Popen(mw_cmd(wav, o, BENCH_LANGUAGE, a.model), stdout=subprocess.DEVNULL,
                                  stderr=subprocess.DEVNULL) for o in outs]
        codes = [p.wait(timeout=MW_TIMEOUT_S * 2) for p in procs]
        res["parallel2_wall_s"] = round(time.perf_counter() - t0, 2)
        res["parallel2_exit"] = codes
        print(f"2 in parallel: {res['parallel2_wall_s']} s wall, exit {codes}")
    update_results(RESULTS, a.key, res)
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
        out = wav.with_suffix(".json")
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
    engine = a.model.split(":", 1)[0]
    out = WORK / f"mic_seat_{engine}.json"
    secs = mw_run(src, out, model=a.model)
    text = mw_text(out)
    (WORK / f"mic_seat_{engine}.txt").write_text(text + "\n", encoding="utf-8")
    res = load_results().get("mic", {})
    for legacy in ("asr_s", "words"):
        res.pop(legacy, None)
    res.update({"file": str(src.relative_to(ROOT)) if src.is_absolute() else str(src),
                "duration_s": round(ffprobe_duration(src), 1),
                "level": level_stats(decode_pcm(src), sr=16000, floor_db=SILENCE_DB)})
    segments = json.loads(out.read_text(encoding="utf-8"))["segments"]
    res.setdefault("by_model", {})[a.model] = {"asr_s": secs, "words": len(text.split()),
                                               "covered_s": segment_coverage(segments)}
    res.setdefault("readable", None)
    update_results(RESULTS, "mic", res)
    m = res["by_model"][a.model]
    print(f"{a.model}: {m['words']} words, {m['covered_s']}/{res['duration_s']} s covered, {secs} s "
          f"→ runs/stage0/mic_seat_{engine}.txt — then: stage0.py mic-verdict yes|no")


def cmd_mic_verdict(a: argparse.Namespace) -> None:
    res = load_results().get("mic")
    if not res:
        sys.exit("run `stage0.py mic <file>` first")
    res["readable"] = a.readable
    update_results(RESULTS, "mic", res)
    print(f"mic readable: {a.readable}")


def cmd_report(a: argparse.Namespace) -> None:
    page = Path(a.page)
    html = page.read_text(encoding="utf-8")
    page.write_text(fill_metrics(html, load_results()), encoding="utf-8", newline="")
    print(f"filled {len(_METRIC.findall(html))} metrics in {page}")


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
    p.add_argument("--wav", default=str(WORK / "bench_45s.wav"), help="clip to time (default: the lecture cut)")
    p.add_argument("--model", default=MW_MODEL)
    p.add_argument("--key", default="mw", help="results key; the gate reads only `mw` (the real lecture clip)")
    p.set_defaults(fn=cmd_mw_bench)
    sub.add_parser("extract-bench").set_defaults(fn=cmd_extract_bench)
    p = sub.add_parser("mic")
    p.add_argument("file")
    p.add_argument("--model", default=MW_MODEL)
    p.set_defaults(fn=cmd_mic)
    p = sub.add_parser("mic-verdict")
    p.add_argument("readable", choices=["yes", "no"])
    p.set_defaults(fn=cmd_mic_verdict)
    p = sub.add_parser("report")
    p.add_argument("--page", default=str(REPORT), help="any Hebrew page with <span data-metric> placeholders")
    p.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
