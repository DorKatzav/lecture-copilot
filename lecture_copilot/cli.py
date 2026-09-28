"""Command line. M1: `replay <audio file> --course NAME [--pace fast|realtime]`. Terminal output is English only;
Hebrew content stays in the database and in the pages built from it."""

import argparse
import asyncio
import sys
from collections.abc import Callable
from datetime import date as date_cls
from pathlib import Path

import httpx

from lecture_copilot.asr.base import ASR
from lecture_copilot.asr.macwhisper import MacWhisperASR
from lecture_copilot.audio.sources import ChunkSource, FileSource
from lecture_copilot.config import CHUNK_BUDGET_S, DB_PATH, OLLAMA_URL, RUNS_DIR, Profile
from lecture_copilot.memprobe import MemoryProbe, total_gb
from lecture_copilot.pipeline import Ctx, run, warm_up
from lecture_copilot.scriptcheck import terminal_text
from lecture_copilot.store.db import Store, new_id

TRANSCRIPT_SUFFIXES = {".vtt", ".srt", ".json", ".txt"}
PRESSURE = {1: "normal", 2: "warning", 4: "critical"}


def fmt_chunk(o: dict) -> str:
    def s(v: float | None) -> str:
        return "  -  " if v is None else f"{v:5.1f}"
    line = (f"chunk {o['idx']:04d}  {o['t0']:7.1f}–{o['t1']:7.1f} s  {o['status']:<14} asr {s(o['asr_s'])} s · "
            f"extract {s(o['extract_s'])} s · total {s(o['total_s'])} s · {o['segments']} segments")
    if o["total_s"] > CHUNK_BUDGET_S:
        line += f"  OVER BUDGET ({CHUNK_BUDGET_S} s)"
    if o["status"] in ("asr_failed", "failed"):
        line += f"  [{terminal_text(o.get('error', ''), 120)}]"
    return line


def fmt_summary(lecture_id: str, s: dict) -> list[str]:
    status = ", ".join(f"{k} {v}" for k, v in sorted(s["status"].items()))
    t, c = s["timing"], s["counts"]
    lines = [f"lecture {lecture_id}: {s['chunks']} chunks ({status}) · {s['audio_s'] / 60:.1f} min audio",
             f"rows: segments {c['segments']} · items {c['items']} · claims {c['claims']}",
             "p95: " + " · ".join(f"{k.removesuffix('_s')} {t[k]['p95']} s" for k in ("asr_s", "extract_s", "total_s"))
             + f" (max total {t['total_s']['max']} s, budget {CHUNK_BUDGET_S} s)"]
    m = s.get("memory")
    if m and m.get("samples"):
        models = ", ".join(f"{k} {v} GB" for k, v in m["models_at_peak_gb"].items()) or "none"
        procs = ", ".join(f"{k} {v / 1024:.1f} GB" for k, v in m["procs_peak_mb"].items())
        lines.append(f"memory: peak {m['peak_used_gb']} GB of {m['total_gb']} GB (baseline {m['baseline_used_gb']}), "
                     f"pressure {PRESSURE.get(m['max_pressure'], m['max_pressure'])}, "
                     f"swap {m['swap_growth_mb']:+.0f} MB · models at peak: {models} · process peaks: {procs}")
    return lines


async def replay(file: Path, course: str, language: str, title: str | None, date: str, pace: str, db: Path,
                 fact_check: bool, *, asr: ASR, client: httpx.AsyncClient,
                 source_factory: Callable[[str, Path, str], ChunkSource], runs_dir: Path,
                 probe: MemoryProbe | None, echo: Callable[[str], None] = print) -> dict:
    file = Path(file).resolve()
    store = Store(db)
    try:
        course_id = store.upsert_course(course, language=language)
        lang = store.course(course_id)["language"]
        if lang != language:
            echo(f"note: course language is {lang} (set when the course was created); --language {language} ignored")
        lecture_id = store.upsert_lecture(course_id, audio_path=str(file), source="file", title=title or file.stem,
                                          date=date, fact_check=fact_check)
        run_id = new_id()
        echo(f"lecture {lecture_id} · run {run_id} · {file.name} · pace {pace} · language {lang}")

        def extra() -> dict:
            out = {"source": file.name, "pace": pace, "language": lang}
            if probe:
                probe.stop()
                out["memory"] = probe.summary()
            return out

        async with client:
            ctx = Ctx(lecture_id=lecture_id, course_id=course_id, course_name=course, lecture_title=title or file.stem,
                      profile=Profile(fact_check=fact_check, language=lang), store=store, asr=asr, ollama=client,
                      run_id=run_id, runs_dir=runs_dir)
            if probe:
                probe.start()
            try:
                await warm_up(ctx)
                summary = await run(source_factory(lecture_id, file, pace), ctx, extra=extra,
                                    on_chunk=lambda o: echo(fmt_chunk(o)))
            finally:
                if probe:
                    probe.stop()
        store.end_lecture(lecture_id)
        for line in fmt_summary(lecture_id, summary):
            echo(line)
        return summary
    finally:
        store.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m lecture_copilot.cli")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="run a recording through the pipeline")
    r.add_argument("file", type=Path)
    r.add_argument("--course", required=True)
    r.add_argument("--language", choices=["he", "en"], default="he", help="only used when the course is new")
    r.add_argument("--title")
    r.add_argument("--date", default=date_cls.today().isoformat())
    r.add_argument("--pace", choices=["fast", "realtime"], default="fast")
    r.add_argument("--db", type=Path, default=DB_PATH)
    r.add_argument("--no-fact-check", dest="fact_check", action="store_false")
    a = ap.parse_args(argv)

    if a.file.suffix.lower() in TRANSCRIPT_SUFFIXES:
        print(f"{a.file.name}: transcript replay (TranscriptSource) arrives in M2", file=sys.stderr)
        return 2
    if not a.file.is_file():
        print(f"{a.file}: not found", file=sys.stderr)
        return 2
    try:
        asyncio.run(replay(a.file, a.course, a.language, a.title, a.date, a.pace, a.db, a.fact_check,
                           asr=MacWhisperASR(), client=httpx.AsyncClient(base_url=OLLAMA_URL),
                           source_factory=lambda lid, f, pace: FileSource(lid, f, pace, runs_dir=RUNS_DIR),
                           runs_dir=RUNS_DIR, probe=MemoryProbe(total_gb=total_gb())))
    except RuntimeError as e:
        print(f"replay stopped: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
