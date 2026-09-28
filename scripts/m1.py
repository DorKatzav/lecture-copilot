"""M1 measurements. Numbers land in eval/m1.json or come from db/copilot.sqlite; the Hebrew pages read them through
data-metric placeholders (`report`), never typed.

    python scripts/m1.py runner-footprint      # gemma3:12b runner memory: Ollama's default load vs use_mmap
    python scripts/m1.py prompt-cache          # runner memory over distinct prompts, cache on vs off (restarts Ollama)
    python scripts/m1.py asr-repeat            # the fixture's chunks through mw twice: is the transcript the same?
    python scripts/m1.py report --page docs/notes/MEMORY_HE.html
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx

from lecture_copilot.config import DB_PATH, LIVE_MODEL, OLLAMA_LOAD_OPTIONS, OLLAMA_NUM_CTX, OLLAMA_URL, ROOT
from lecture_copilot.metrics_page import fill_metrics

RESULTS = ROOT / "eval" / "m1.json"
FIXTURE = ROOT / "eval" / "fixture_10min.m4a"
UNIT_MB = {"B": 1 / 2**20, "KB": 1 / 1024, "MB": 1.0, "GB": 1024.0, "K": 1 / 1024, "M": 1.0, "G": 1024.0}


# ---------- pure helpers (tested) ----------

def parse_footprint(text: str) -> dict:
    """`footprint -p <pid>` → dirty MB per category, and their sum."""
    cats: dict[str, float] = {}
    for m in re.finditer(r"^\s*([\d.]+) (B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+[\d.]+ (?:B|KB|MB|GB)\s+\d+\s+(.+)$",
                         text, re.MULTILINE):
        name = m.group(3).strip()
        name = re.sub(r"^untagged \((.+)\)$", r"\1", name)
        cats[name] = cats.get(name, 0.0) + float(m.group(1)) * UNIT_MB[m.group(2)]
    return {"footprint_mb": round(sum(cats.values()), 1), "categories": {k: round(v, 1) for k, v in cats.items()}}


def parse_vmmap_swapped_mb(text: str, region: str) -> float:
    """`vmmap --summary` row: REGION VIRTUAL RESIDENT DIRTY SWAPPED … → SWAPPED in MB."""
    m = re.search(rf"^{re.escape(region)}\s+([\d.]+[KMG])\s+([\d.]+[KMG])\s+([\d.]+[KMG])\s+([\d.]+)([KMG])",
                  text, re.MULTILINE)
    return round(float(m.group(4)) * UNIT_MB[m.group(5)], 1) if m else 0.0


def load_mode(command: str) -> str:
    m = re.search(r"--load-mode (\S+)", command)
    return m.group(1) if m else "mmap (default)"


def fixture_metrics(db: Path, fixture: Path) -> dict:
    import sqlite3
    con = sqlite3.connect(db)
    lids = [r[0] for r in con.execute("select id from lectures where source = 'file' and audio_path = ?",
                                      (str(fixture.resolve()),))]
    runs = [{"run_id": ref, **json.loads(o)} for lid in lids for ref, o in con.execute(
        "select input_ref, output_json from decisions where node = 'run' and lecture_id = ? order by ts", (lid,))]
    for r in runs:
        calls = [json.loads(o) for (o,) in con.execute(
            "select output_json from decisions where node = 'extractor' and input_ref like ?", (r["run_id"] + "#0%",))]
        r["extract"] = {"calls": len(calls), "first_valid": sum(bool(c.get("first_valid")) for c in calls),
                        "retries": sum(c.get("attempts", 1) - 1 for c in calls)}
    kinds = {k: n for lid in lids for k, n in con.execute(
        "select kind, count(*) from items where lecture_id = ? group by kind", (lid,))}
    return {"n_runs": len(runs), "first": runs[0] if runs else {}, "last": runs[-1] if runs else {},
            "by_run": {f"r{i}": r for i, r in enumerate(runs, 1)}, "item_kinds": kinds}


def compare_transcripts(a: dict, b: dict) -> dict:
    """Two mw JSON outputs of the same wav: same words? same segment boundaries?"""
    def text(d: dict) -> str:
        return " ".join(" ".join(s["text"] for s in d["segments"]).split())

    def cuts(d: dict) -> list:
        return [(s["start"], s["end"]) for s in d["segments"]]
    return {"same_text": text(a) == text(b), "same_segments": cuts(a) == cuts(b)}


def summarize_repeat(rows: list[dict]) -> dict:
    return {"n": len(rows), "same_text": sum(r["same_text"] for r in rows),
            "same_segments": sum(r["same_segments"] for r in rows)}


def cache_growth(footprints_mb: list[float]) -> dict:
    steps = [b - a for a, b in zip(footprints_mb, footprints_mb[1:], strict=False)]
    return {"first_mb": round(footprints_mb[0]), "last_mb": round(footprints_mb[-1]),
            "per_prompt_mb": round(sum(steps) / len(steps)) if steps else 0}


# ---------- providers (run for real, not in tests) ----------

def _sh(*cmd: str) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def _runner_pid() -> str:
    """The llama-server that serves the vision-packaged gemma3 (the only runner started with --mmproj)."""
    out = subprocess.run(["pgrep", "-f", "llama-server.*--mmproj"], capture_output=True, text=True).stdout.split()
    if not out:
        raise RuntimeError("no gemma3 runner found")
    return out[0]


def _unload(client: httpx.Client) -> None:
    client.post("/api/generate", json={"model": LIVE_MODEL, "keep_alive": 0})
    for _ in range(30):
        if not any(m["name"].startswith(LIVE_MODEL) for m in client.get("/api/ps").json()["models"]):
            return
        time.sleep(1)


def cmd_runner_footprint(_: argparse.Namespace) -> None:
    res = {}
    with httpx.Client(base_url=OLLAMA_URL, timeout=300) as client:
        for variant, extra in (("default", {}), ("use_mmap", {"use_mmap": True})):
            _unload(client)
            time.sleep(2)
            client.post("/api/generate", json={"model": LIVE_MODEL, "prompt": "Reply with OK.", "stream": False,
                                               "keep_alive": "5m", "options": {"num_ctx": OLLAMA_NUM_CTX, **extra}})
            time.sleep(3)
            pid = _runner_pid()
            fp = parse_footprint(_sh("footprint", "-p", pid))
            res[variant] = {"load_mode": load_mode(_sh("ps", "-o", "command=", "-p", pid)),
                            "footprint_gb": round(fp["footprint_mb"] / 1024, 1),
                            "malloc_large_gb": round(fp["categories"].get("MALLOC_LARGE", 0) / 1024, 1),
                            "vm_allocate_gb": round(fp["categories"].get("VM_ALLOCATE", 0) / 1024, 1),
                            "malloc_large_swapped_gb": round(parse_vmmap_swapped_mb(
                                _sh("vmmap", "--summary", pid), "MALLOC_LARGE") / 1024, 1)}
            print(variant, res[variant])
        _unload(client)  # the next replay loads it with the configured options
    res["ollama_version"] = _sh("ollama", "--version").strip().splitlines()[-1].split()[-1]
    _save("runner", res)


def _save(key: str, value: object) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="")


def _restart_ollama(cache_off: bool) -> None:
    subprocess.run(["pkill", "-f", "^ollama serve"])
    time.sleep(3)
    if cache_off:
        subprocess.run([str(ROOT / "scripts" / "ollama_serve.sh")], check=True)
        return
    (ROOT / "runs").mkdir(exist_ok=True)
    with (ROOT / "runs" / "ollama.log").open("a") as log:
        subprocess.Popen(["ollama", "serve"], stdout=log, stderr=log, start_new_session=True)
    time.sleep(4)


def cmd_prompt_cache(a: argparse.Namespace) -> None:
    """D-M1-4. Distinct prompts of about one chunk's size; the runner's footprint after each."""
    import random
    words = "lecture customer acquisition cost python function variable loop market segment revenue".split()
    res = {"prompts": a.n}
    try:
        for variant, off in (("cache_on", False), ("cache_off", True)):
            _restart_ollama(cache_off=off)
            with httpx.Client(base_url=OLLAMA_URL, timeout=300) as client:
                opts = {**OLLAMA_LOAD_OPTIONS, "temperature": 0, "seed": 42, "num_predict": 20}
                mb, tokens, eval_s = [], 0, []
                for i in range(a.n):
                    random.seed(i)
                    text = " ".join(random.choice(words) for _ in range(1200))
                    d = client.post("/api/chat", json={
                        "model": LIVE_MODEL, "stream": False, "keep_alive": "5m", "options": opts,
                        "messages": [{"role": "user", "content": f"Request {i}. Summarize in one sentence:\n{text}"}],
                    }).json()
                    tokens = d["prompt_eval_count"]
                    eval_s.append(d["prompt_eval_duration"] / 1e9)
                    mb.append(parse_footprint(_sh("footprint", "-p", _runner_pid()))["footprint_mb"])
                    print(variant, i, round(mb[-1]), "MB")
            res[variant] = {**cache_growth(mb), "first_gb": round(mb[0] / 1024, 1), "last_gb": round(mb[-1] / 1024, 1),
                            "prompt_tokens": tokens, "prompt_eval_s": round(sorted(eval_s)[len(eval_s) // 2], 2)}
    finally:
        _restart_ollama(cache_off=True)  # always leave the server the way the project needs it
    res["ollama_version"] = _sh("ollama", "--version").strip().splitlines()[-1].split()[-1]
    _save("prompt_cache", res)
    print(json.dumps(res, indent=2))


def cmd_asr_repeat(a: argparse.Namespace) -> None:
    """D-M1-5. Every chunk wav of a replayed lecture through mw twice."""
    import asyncio
    import shutil
    import tempfile

    from lecture_copilot.asr.macwhisper import MacWhisperASR
    asr, rows = MacWhisperASR(), []
    with tempfile.TemporaryDirectory() as tmp:
        for wav in sorted(Path(a.dir).glob("chunk_*.wav")):
            outs = []
            for k in (1, 2):
                copy = Path(tmp) / f"{wav.stem}_{k}.wav"
                shutil.copy(wav, copy)
                asyncio.run(asr.transcribe(copy, a.language))
                outs.append(json.loads(copy.with_suffix(".json").read_text(encoding="utf-8")))
            rows.append(compare_transcripts(*outs))
            print(wav.name, rows[-1])
    _save("asr_repeat", {**summarize_repeat(rows), "model": asr.models[a.language],
                         "mw_version": _sh(asr.binary, "version").strip().splitlines()[0]})


def cmd_report(a: argparse.Namespace) -> None:
    data = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {}
    data["fixture"] = fixture_metrics(DB_PATH, FIXTURE)
    page = Path(a.page)
    page.write_text(fill_metrics(page.read_text(encoding="utf-8"), data), encoding="utf-8", newline="")
    print(f"filled {page}")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="m1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("runner-footprint").set_defaults(fn=cmd_runner_footprint)
    c = sub.add_parser("prompt-cache")
    c.add_argument("--n", type=int, default=6)
    c.set_defaults(fn=cmd_prompt_cache)
    r = sub.add_parser("asr-repeat")
    r.add_argument("dir", help="runs/<lecture_id> of a replayed lecture")
    r.add_argument("--language", default="he")
    r.set_defaults(fn=cmd_asr_repeat)
    r = sub.add_parser("report")
    r.add_argument("--page", required=True)
    r.set_defaults(fn=cmd_report)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
