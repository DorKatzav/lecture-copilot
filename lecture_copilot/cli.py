"""Command line. Terminal output is English only; Hebrew stays in the database and in the files built from it.

    replay <audio | .vtt | MacWhisper .json> --course NAME [--week N] [--pace fast|realtime]   pipeline, then Digest
    digest [--lecture ID]                                                        rebuild the Digest of a lecture
"""

import argparse
import asyncio
import os
import sys
import time
from collections.abc import Callable
from datetime import date as date_cls
from pathlib import Path

import httpx

from lecture_copilot.agents.verifier import GeminiAPI, Verifier, VerifierBackend, VerifierWorker
from lecture_copilot.asr.base import ASR
from lecture_copilot.asr.macwhisper import MacWhisperASR
from lecture_copilot.asr.transcript import TranscriptASR
from lecture_copilot.audio.sources import ChunkSource, FileSource
from lecture_copilot.audio.transcript import TranscriptSource
from lecture_copilot.config import (
    CHUNK_BUDGET_S,
    COST_BUDGET_USD,
    COURSES_ROOT,
    DB_PATH,
    DIGEST_BUDGET_S,
    OLLAMA_URL,
    ROOT,
    RUNS_DIR,
    VERIFY_MIN_IMPORTANCE,
    Profile,
    load_env,
)
from lecture_copilot.memprobe import MemoryProbe, total_gb
from lecture_copilot.output.digest import digest, render_markdown, section_headings
from lecture_copilot.output.sinks import FolderSink, Sink
from lecture_copilot.pipeline import Ctx, run, warm_up
from lecture_copilot.scriptcheck import terminal_text
from lecture_copilot.store.db import Store, new_id
from lecture_copilot.store.net import Net

ENV_FILE = ROOT / ".env"
TRANSCRIPT_SUFFIXES = {".vtt", ".json"}
UNSUPPORTED_SUFFIXES = {".srt", ".txt"}
PRESSURE = {1: "normal", 2: "warning", 4: "critical"}


def fmt_chunk(o: dict) -> str:
    def s(v: float | None) -> str:
        return "  -  " if v is None else f"{v:5.1f}"
    line = (f"chunk {o['idx']:04d}  {o['t0']:7.1f}–{o['t1']:7.1f} s  {o['status']:<14} asr {s(o['asr_s'])} s · "
            f"memory {s(o.get('memory_s'))} s · extract {s(o['extract_s'])} s · total {s(o['total_s'])} s · "
            f"{o['segments']} segments")
    if o.get("already_said") or o.get("contradictions"):
        line += f" · already said {o['already_said']}, contradictions {o['contradictions']}"
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
             "p95: " + " · ".join(f"{k.removesuffix('_s')} {t[k]['p95']} s"
                                 for k in ("asr_s", "memory_s", "extract_s", "total_s") if k in t)
             + f" (max total {t['total_s']['max']} s, budget {CHUNK_BUDGET_S} s)"]
    if s.get("verifier") is not None:
        v = s["verifier"]
        over = f"  OVER BUDGET (${COST_BUDGET_USD})" if s["cost_usd"] > COST_BUDGET_USD else ""
        lines.append(f"verifier: {v['verified']} verified · {v['unchecked']} unchecked · {v['skipped']} skipped · "
                     f"{v['retried']} retried at stop · {s['net_calls']} calls · ${s['cost_usd']:.4f}{over}")
    else:
        lines.append("verifier: fact-checking off — every claim skipped, nothing left the machine")
    m = s.get("memory")
    if m and m.get("samples"):
        models = ", ".join(f"{k} {v} GB" for k, v in m["models_at_peak_gb"].items()) or "none"
        procs = ", ".join(f"{k} {v / 1024:.1f} GB" for k, v in m["procs_peak_mb"].items())
        lines.append(f"memory: peak {m['peak_used_gb']} GB of {m['total_gb']} GB (baseline {m['baseline_used_gb']}), "
                     f"pressure {PRESSURE.get(m['max_pressure'], m['max_pressure'])}, "
                     f"swap {m['swap_growth_mb']:+.0f} MB · models at peak: {models} · process peaks: {procs}")
    return lines


async def make_digest(lecture_id: str, store: Store, client: httpx.AsyncClient, sink: Sink,
                      echo: Callable[[str], None]) -> dict:
    """ "סיום": the Digest, then the sink. A failing sink never loses the Digest — it is already in the database."""
    t = time.perf_counter()
    doc = await digest(lecture_id, store=store, client=client)
    digest_s = round(time.perf_counter() - t, 1)
    out = {"folder": None, "digest_s": digest_s, "degraded": doc.degraded,
           "sections": len(section_headings(render_markdown(doc)))}
    t = time.perf_counter()
    try:
        folder = sink.write_lecture(doc)
        out["folder"] = str(folder)
        log = {"status": "ok", "folder": str(folder), "files": sum(1 for p in folder.iterdir() if p.is_file())}
    except OSError as e:
        log = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
    store.log("sink", lecture_id=lecture_id, input_ref=lecture_id, ms=(time.perf_counter() - t) * 1000,
              output={"sink": type(sink).__name__, **log})
    over = f"  OVER BUDGET ({DIGEST_BUDGET_S} s)" if digest_s > DIGEST_BUDGET_S else ""
    degraded = f" · degraded: {', '.join(doc.degraded)}" if doc.degraded else ""
    echo(f"digest: {out['sections']} sections · {doc.minutes} min lecture · {digest_s} s{over}{degraded}")
    echo(terminal_text(f"folder: {out['folder']}" if out["folder"] else f"sink failed: {log['error']}", 300))
    return out


async def verify_lecture(lecture_id: str | None, *, db: Path, gemini: VerifierBackend, client: httpx.AsyncClient,
                         sink: Sink, echo: Callable[[str], None] = print) -> dict:
    """Fact-check a lecture that was replayed without it (or whose claims went unchecked), then rebuild its Digest.
    Claims below the threshold stay skipped; verified ones are not checked again."""
    from lecture_copilot.agents.verifier import material
    store = Store(db)
    try:
        row = store.con.execute("select id from lectures where (? is null or id = ?) and status != 'recording' "
                                "order by ended_at desc limit 1", (lecture_id, lecture_id)).fetchone()
        if row is None:
            raise RuntimeError("no lecture to verify" if lecture_id is None else f"no lecture {lecture_id}")
        lid = row[0]
        store.con.execute("update claims set status = 'pending' where lecture_id = ? and status in ('skipped', "
                          "'unchecked') and importance >= ?", (lid, VERIFY_MIN_IMPORTANCE))
        store.con.commit()
        worker = VerifierWorker(Verifier(store, Net(store), gemini), lid)
        task = asyncio.create_task(worker.run())
        todo = [c["id"] for c in material(store.claims(lid)) if c["status"] == "pending"]
        for claim_id in todo:
            worker.enqueue(claim_id)
        out = {"verifier": await worker.finish(), "queued": len(todo)}
        await task
        net = store.con.execute("select coalesce(sum(cost_usd), 0), count(*) from decisions where node = 'net' "
                                "and lecture_id = ?", (lid,)).fetchone()
        v = out["verifier"]
        echo(f"verifier: {len(todo)} queued · {v['verified']} verified · {v['unchecked']} unchecked · "
             f"{v['skipped']} skipped · {net[1]} calls so far · ${float(net[0]):.4f}")
        async with client:
            out["digest"] = await make_digest(lid, store, client, sink, echo)
        return out
    finally:
        store.close()


async def rebuild_digest(lecture_id: str | None, *, db: Path, client: httpx.AsyncClient, sink: Sink,
                         echo: Callable[[str], None] = print) -> dict:
    store = Store(db)
    try:
        row = store.con.execute("select id from lectures where (? is null or id = ?) and status != 'recording' "
                                "order by ended_at desc limit 1", (lecture_id, lecture_id)).fetchone()
        if row is None:
            raise RuntimeError("no lecture to digest" if lecture_id is None else f"no lecture {lecture_id}")
        async with client:
            return await make_digest(row[0], store, client, sink, echo)
    finally:
        store.close()


async def replay(file: Path, course: str, language: str, title: str | None, date: str, pace: str, db: Path,
                 fact_check: bool, *, asr: ASR, client: httpx.AsyncClient,
                 source_factory: Callable[[str, Path, str], ChunkSource], runs_dir: Path,
                 probe: MemoryProbe | None, sink: Sink, week: int | None = None, source: str = "file",
                 gemini: VerifierBackend | None = None, echo: Callable[[str], None] = print) -> dict:
    file = Path(file).resolve()
    store = Store(db)
    try:
        course_id = store.upsert_course(course, language=language)
        lang = store.course(course_id)["language"]
        if lang != language:
            echo(f"note: course language is {lang} (set when the course was created); --language {language} ignored")
        lecture_id = store.upsert_lecture(course_id, audio_path=str(file), source=source, title=title or file.stem,
                                          date=date, fact_check=fact_check, week=week)
        run_id = new_id()
        echo(terminal_text(f"lecture {lecture_id} · run {run_id} · {file.name} · pace {pace} · language {lang}", 300))

        def extra() -> dict:
            out = {"source": file.name, "pace": pace, "language": lang}
            if probe:
                probe.stop()
                out["memory"] = probe.summary()
            return out

        async with client:
            verifier = None
            if fact_check and gemini is not None:
                verifier = VerifierWorker(Verifier(store, Net(store), gemini), lecture_id)
            ctx = Ctx(lecture_id=lecture_id, course_id=course_id, course_name=course, lecture_title=title or file.stem,
                      profile=Profile(fact_check=fact_check and gemini is not None, language=lang), store=store,
                      asr=asr, ollama=client, run_id=run_id, runs_dir=runs_dir, verifier=verifier)
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
            summary["digest"] = await make_digest(lecture_id, store, client, sink, echo)
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
    r.add_argument("--week", type=int)
    r.add_argument("--no-fact-check", dest="fact_check", action="store_false")
    d = sub.add_parser("digest", help="rebuild the Digest of a lecture (default: the last one that ended)")
    d.add_argument("--lecture")
    e = sub.add_parser("eval", help="verdict accuracy, Precision@5, cache, cost on eval/benchmark.json")
    e.add_argument("--benchmark", type=Path, default=ROOT / "eval" / "benchmark.json")
    v = sub.add_parser("verify", help="fact-check a lecture's material claims (default: the last one), then its Digest")
    v.add_argument("--lecture")
    c = sub.add_parser("copilot", help="the launcher: Ollama, MacWhisper check, the page in the browser")
    c.add_argument("--no-browser", action="store_true")
    c.add_argument("--port", type=int, default=8770)
    for p in (r, d, e, v):
        p.add_argument("--db", type=Path, default=DB_PATH)
        p.add_argument("--courses-root", type=Path, default=COURSES_ROOT)
    a = ap.parse_args(argv)
    if a.cmd == "copilot":
        from lecture_copilot.web.launcher import main as launch
        return launch(open_browser=not a.no_browser, port=a.port)
    client = httpx.AsyncClient(base_url=OLLAMA_URL)
    sink = FolderSink(a.courses_root)
    gemini = None
    if a.cmd == "eval":
        from lecture_copilot.eval import run_eval
        try:
            load_env(ENV_FILE, fact_check=True)
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            return 2
        if not a.benchmark.is_file():
            print(f"{a.benchmark}: not found — label it first (PLAN §7)", file=sys.stderr)
            return 2
        out = asyncio.run(run_eval(a.db, a.benchmark, GeminiAPI(os.environ["GEMINI_API_KEY"])))
        v = out["verdicts"]
        p_at_k = ", ".join(f"{d} {p['precision']:.0%} (k={p['k']})" for d, p in out["precision_at_5"].items())
        print(f"eval ({out['labels_by'][:40]}): {v['n']} claims · accuracy {v['accuracy']:.0%} · injected caught "
              f"{v['injected_caught']}/{v['injected']} · unchecked {out['unchecked']} · P@k {p_at_k} "
              f"· cache hit {out['cache_hit']} · ${out['cost_usd']:.4f} · {out['seconds']} s")
        return 0
    if a.cmd == "verify":
        try:
            load_env(ENV_FILE, fact_check=True)
        except RuntimeError as err:
            print(str(err), file=sys.stderr)
            return 2
        asyncio.run(verify_lecture(a.lecture, db=a.db, gemini=GeminiAPI(os.environ["GEMINI_API_KEY"]), client=client,
                                   sink=sink))
        return 0
    if a.cmd == "replay":
        suffix = a.file.suffix.lower()
        if suffix in UNSUPPORTED_SUFFIXES:
            print(f"{a.file.name}: supported transcripts are .vtt and MacWhisper .json", file=sys.stderr)
            return 2
        if not a.file.is_file():
            print(f"{a.file}: not found", file=sys.stderr)
            return 2
        try:
            load_env(ENV_FILE, fact_check=a.fact_check)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 2
        if a.fact_check:
            gemini = GeminiAPI(os.environ["GEMINI_API_KEY"])

    try:
        if a.cmd == "digest":
            asyncio.run(rebuild_digest(a.lecture, db=a.db, client=client, sink=sink))
            return 0
        transcript = suffix in TRANSCRIPT_SUFFIXES
        asyncio.run(replay(
            a.file, a.course, a.language, a.title, a.date, a.pace, a.db, a.fact_check, week=a.week, sink=sink,
            source="transcript" if transcript else "file", asr=TranscriptASR() if transcript else MacWhisperASR(),
            source_factory=(lambda lid, f, pace: TranscriptSource(lid, f, runs_dir=RUNS_DIR)) if transcript
            else (lambda lid, f, pace: FileSource(lid, f, pace, runs_dir=RUNS_DIR)),
            client=client, runs_dir=RUNS_DIR, probe=MemoryProbe(total_gb=total_gb()), gemini=gemini))
    except RuntimeError as e:
        print(terminal_text(f"stopped: {e}", 300), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
