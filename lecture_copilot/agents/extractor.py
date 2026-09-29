"""Extractor (PLAN.md §3.5): one Ollama call per chunk → concepts, claims, questions/tasks/highlights.

prompts/extract_v0.md, gemma3:12b (D-M0-10), pydantic validation + the foreign-script check, one retry.
A failure is a logged `extractor` row and an ExtractError — the chunk keeps its transcript, the lecture goes on.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from lecture_copilot import prompts
from lecture_copilot.agents.schemas import ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.config import (
    EXTRACT_OPTIONS,
    EXTRACT_PROMPT,
    EXTRACT_TIMEOUT_S,
    LIVE_MODEL,
    OLLAMA_KEEP_ALIVE,
)
from lecture_copilot.llm import chat_json
from lecture_copilot.scriptcheck import forbidden_scripts

if TYPE_CHECKING:
    from lecture_copilot.audio.sources import AudioChunk
    from lecture_copilot.pipeline import Ctx



class ExtractError(RuntimeError):
    pass


def result_texts(r: ExtractResult) -> list[str]:
    texts = [r.chunk_summary]
    texts += [t for c in r.concepts for t in (c.term, c.explanation, c.canonical_key)]
    texts += [t for c in r.claims for t in (c.text, c.normalized)]
    texts += [t for it in r.items for t in (it.text, it.owner, it.due) if t]
    return texts


def script_problem(r: ExtractResult) -> str | None:
    found = set().union(*(forbidden_scripts(t) for t in result_texts(r)))
    if found:
        return f"The output contains {', '.join(sorted(found))} letters. Write only Hebrew and English."
    return None


async def extract(chunk: AudioChunk, segments: list[Segment], ctx: Ctx, ref: str,
                  prompt: str = EXTRACT_PROMPT) -> ExtractResult:
    prev = ctx.store.previous_chunk_summary(ctx.lecture_id, chunk.idx)
    system, user = prompts.load(prompt).render(
        language=ctx.profile.language, course_name=ctx.course_name, lecture_title=ctx.lecture_title,
        idx=chunk.idx, t0=round(chunk.t0), t1=round(chunk.t1), known_terms="(none yet)",
        previous_chunk_summary=f"Previous chunk: {prev}" if prev else "",
        text="\n".join(s.text for s in segments))
    call = await chat_json(LIVE_MODEL, system, user, ExtractResult, client=ctx.ollama, options=EXTRACT_OPTIONS,
                           check=script_problem, keep_alive=OLLAMA_KEEP_ALIVE, timeout_s=EXTRACT_TIMEOUT_S,
                           backoff_s=ctx.backoff_s)
    v = call.value
    ctx.store.log("extractor", lecture_id=ctx.lecture_id, input_ref=ref, ms=round(sum(call.ms), 1),
                  tokens_in=call.tokens_in, tokens_out=call.tokens_out, output={
                      "status": "ok" if v else "failed", "model": LIVE_MODEL, "prompt": prompt,
                      "attempts": call.attempts, "first_valid": call.first_valid, "error": call.error,
                      "load_ms": round(call.load_ms, 1), "attempt_ms": [round(m) for m in call.ms],
                      "concepts": len(v.concepts) if v else 0, "claims": len(v.claims) if v else 0,
                      "items": len(v.items) if v else 0})
    if v is None:
        raise ExtractError(call.error or "no valid output")
    return v
