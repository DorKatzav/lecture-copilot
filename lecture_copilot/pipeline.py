"""One pipeline, many sources: `run(source, ctx)` consumes any ChunkSource and never knows where a chunk came from.

A producer task fills a queue from the source; the worker takes one chunk at a time: ASR → extract → one store
transaction. Every step is a `decisions` row; a failing chunk is marked and the run goes on. At the end a `run`
row holds the counts, the per-stage timing percentiles and whatever the caller adds (e.g. peak memory).
"""

import asyncio
import time
from collections import Counter
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from lecture_copilot.agents.extractor import ExtractError, extract
from lecture_copilot.agents.memory import recall, remember
from lecture_copilot.agents.verifier import VerifierWorker, material
from lecture_copilot.asr.base import ASR, ASRError
from lecture_copilot.audio.sources import SR, AudioChunk, write_wav
from lecture_copilot.config import EMBED_MODEL, LIVE_MODEL, OLLAMA_KEEP_ALIVE, OLLAMA_LOAD_OPTIONS, RUNS_DIR, Profile
from lecture_copilot.stats import percentile
from lecture_copilot.store.db import Store


@dataclass
class Ctx:
    lecture_id: str
    course_id: str
    course_name: str
    lecture_title: str
    profile: Profile
    store: Store
    asr: ASR
    ollama: httpx.AsyncClient
    run_id: str
    runs_dir: Path = RUNS_DIR
    backoff_s: float = 1.0
    verifier: VerifierWorker | None = None    # M4: fact-checking beside the loop; None when it is off


def _since(t: float) -> float:
    return round(time.perf_counter() - t, 2)


async def _process(chunk: AudioChunk, ctx: Ctx, ref: str, out: dict) -> str:
    t = time.perf_counter()
    try:
        raw = await ctx.asr.transcribe(chunk.path, ctx.profile.language)
    except ASRError as e:
        out["asr_s"], out["error"] = _since(t), str(e)
        ctx.store.log("asr", lecture_id=ctx.lecture_id, input_ref=ref, ms=out["asr_s"] * 1000,
                      output={"status": "failed", "asr": ctx.asr.name, "error": str(e)})
        ctx.store.write_chunk(ctx.lecture_id, chunk.idx, [], asr=ctx.asr.name, result=None)  # no stale rows
        return "asr_failed"
    out["asr_s"] = _since(t)
    segments = [s.model_copy(update={"t0": s.t0 + chunk.t0, "t1": s.t1 + chunk.t0}) for s in raw]
    out["segments"] = len(segments)
    ctx.store.log("asr", lecture_id=ctx.lecture_id, input_ref=ref, ms=out["asr_s"] * 1000,
                  output={"status": "ok", "asr": ctx.asr.name, "segments": len(segments),
                          "words": sum(len(s.text.split()) for s in segments)})
    result, status = None, "empty"
    if segments:
        t = time.perf_counter()
        memory = await recall(segments, ctx, ref)
        out["memory_s"] = _since(t)
        t = time.perf_counter()
        try:
            result, status = await extract(chunk, segments, ctx, ref, memory=memory), "ok"
        except ExtractError as e:
            status, out["error"] = "extract_failed", str(e)
        out["extract_s"] = _since(t)
    ctx.store.write_chunk(ctx.lecture_id, chunk.idx, segments, asr=ctx.asr.name, result=result)
    if result is not None:
        t = time.perf_counter()
        mem = await remember(result, chunk, ctx, ref, memory=memory)
        out["memory_s"] = round(out["memory_s"] + _since(t), 2)
        out["already_said"], out["contradictions"] = mem["already_said"], mem["contradictions"]
        if ctx.verifier is not None:
            seg = ctx.store.con.execute("select id from segments where lecture_id = ? and chunk_id = ? order by t0 "
                                        "limit 1", (ctx.lecture_id, chunk.idx)).fetchone()
            new = [c for c in ctx.store.claims(ctx.lecture_id) if seg and c["segment_id"] == seg[0]]
            for c in material(new):
                ctx.verifier.enqueue(c["id"])
            out["enqueued"] = len(material(new))
    return status


async def process_chunk(chunk: AudioChunk, ctx: Ctx, queue_depth: int = 0) -> dict:
    ref = f"{ctx.run_id}#{chunk.idx:04d}"
    out = {"idx": chunk.idx, "t0": round(chunk.t0, 2), "t1": round(chunk.t1, 2), "queue_depth": queue_depth,
           "segments": 0, "asr_s": None, "memory_s": None, "extract_s": None, "already_said": 0, "contradictions": 0}
    t = time.perf_counter()
    try:
        out["status"] = await _process(chunk, ctx, ref, out)
    except Exception as e:  # a bug in one chunk must not stop the lecture
        out["status"], out["error"] = "failed", f"{type(e).__name__}: {e}"
    out["total_s"] = _since(t)
    ctx.store.log("chunk", lecture_id=ctx.lecture_id, input_ref=ref, ms=out["total_s"] * 1000, output=out)
    return out


def _timing(outcomes: list[dict], key: str) -> dict:
    xs = [o[key] for o in outcomes if o[key] is not None]
    if not xs:
        return {"n": 0, "p50": None, "p95": None, "max": None}
    return {"n": len(xs), "p50": percentile(xs, 50), "p95": percentile(xs, 95), "max": max(xs)}


async def run(source: AsyncIterable[AudioChunk], ctx: Ctx, extra: Callable[[], dict] | None = None,
              on_chunk: Callable[[dict], None] | None = None) -> dict:
    queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
    failure: list[Exception] = []

    async def produce() -> None:
        try:
            async for c in source:
                await queue.put(c)
        except Exception as e:
            failure.append(e)
        finally:
            await queue.put(None)

    t = time.perf_counter()
    producer = asyncio.create_task(produce())
    worker = asyncio.create_task(ctx.verifier.run()) if ctx.verifier else None
    outcomes = []
    while (chunk := await queue.get()) is not None:
        outcomes.append(await process_chunk(chunk, ctx, queue_depth=queue.qsize()))
        if on_chunk:
            on_chunk(outcomes[-1])
    await producer
    if not failure:
        ctx.store.prune_chunks(ctx.lecture_id, max((o["idx"] for o in outcomes), default=0))
    verifier_out = None
    if ctx.verifier:
        verifier_out = await ctx.verifier.finish()          # "סיום": drain, retry unchecked, skip the rest
        await worker
    else:
        ctx.store.set_claim_status(ctx.lecture_id, "pending", "skipped")
    net = ctx.store.con.execute("select coalesce(sum(cost_usd), 0), count(*) from decisions where node = 'net' "
                                "and lecture_id = ?", (ctx.lecture_id,)).fetchone()
    summary = {
        "chunks": len(outcomes), "status": dict(Counter(o["status"] for o in outcomes)),
        "counts": ctx.store.counts(ctx.lecture_id), "audio_s": round(sum(o["t1"] - o["t0"] for o in outcomes), 1),
        "timing": {k: _timing(outcomes, k) for k in ("asr_s", "memory_s", "extract_s", "total_s")},
        "already_said": sum(o["already_said"] for o in outcomes),
        "contradictions": sum(o["contradictions"] for o in outcomes),
        "max_queue_depth": max((o["queue_depth"] for o in outcomes), default=0),
        "verifier": verifier_out, "cost_usd": round(float(net[0]), 4), "net_calls": net[1],
    }
    if failure:
        summary["source_error"] = f"{type(failure[0]).__name__}: {failure[0]}"
    if extra:
        summary.update(extra())
    ctx.store.log("run", lecture_id=ctx.lecture_id, input_ref=ctx.run_id, ms=_since(t) * 1000, output=summary)
    if failure:
        raise failure[0]
    return summary


async def warm_up(ctx: Ctx) -> None:
    """Pay the cold starts before the first chunk (stage 0: a cold mw run is over the ASR budget): one mw run on
    a second of silence, and the LLM + embedding model loaded and kept resident. Failures are logged, not raised."""
    ref = f"{ctx.run_id}#warmup"
    if getattr(ctx.asr, "needs_warm_up", True):
        wav = ctx.runs_dir / ctx.lecture_id / "warmup.wav"
        write_wav(wav, np.zeros(SR, dtype=np.float32))
        t = time.perf_counter()
        try:
            await ctx.asr.transcribe(wav, ctx.profile.language)
            out = {"status": "ok"}
        except Exception as e:
            out = {"status": "failed", "error": f"{type(e).__name__}: {e}"}
        ctx.store.log("asr", lecture_id=ctx.lecture_id, input_ref=ref, ms=_since(t) * 1000, output=out)
    loads = (("extractor", LIVE_MODEL, "/api/generate", {}), ("memory", EMBED_MODEL, "/api/embed", {"input": "warm"}))
    for node, model, path, extra_body in loads:
        t = time.perf_counter()
        try:
            body = {"model": model, "keep_alive": OLLAMA_KEEP_ALIVE, "options": OLLAMA_LOAD_OPTIONS, **extra_body}
            r = await ctx.ollama.post(path, json=body, timeout=180)
            r.raise_for_status()
            out = {"status": "ok", "model": model}
        except httpx.HTTPError as e:
            out = {"status": "failed", "model": model, "error": f"{type(e).__name__}: {e}"}
        ctx.store.log(node, lecture_id=ctx.lecture_id, input_ref=ref, ms=_since(t) * 1000, output=out)
