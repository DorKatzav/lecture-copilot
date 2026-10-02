""""מה פספסתי" (PLAN.md §3.5): the last N minutes of chunk summaries → one Ollama call → up to three lines."""

import time

import httpx

from lecture_copilot import prompts
from lecture_copilot.agents.schemas import Recap
from lecture_copilot.config import DIGEST_OPTIONS, EXTRACT_TIMEOUT_S, LIVE_MODEL, OLLAMA_KEEP_ALIVE, RECAP_PROMPT
from lecture_copilot.llm import chat_json
from lecture_copilot.scriptcheck import forbidden_scripts
from lecture_copilot.store.db import Store


async def recap(lecture_id: str, *, store: Store, client: httpx.AsyncClient, minutes: int = 5,
                backoff_s: float = 1.0) -> dict:
    t0 = time.perf_counter()
    segs = store.segments(lecture_id)
    if not segs:
        return {"bullets": [], "minutes": minutes, "note": "אין עדיין קטעים מתומללים."}
    since = max(s["t1"] for s in segs) - minutes * 60
    recent = sorted({s["chunk_id"] for s in segs if s["t1"] > since})
    lines = [t for cid, t in store.chunk_summaries(lecture_id) if cid in recent]
    items = [it for it in store.items(lecture_id) if it["chunk_id"] in recent]
    concepts = [it["text"] for it in items if it["kind"] == "concept"]
    highlights = [it["text"] for it in items if it["kind"] == "highlight"]
    if not lines:
        return {"bullets": [], "minutes": minutes, "note": "אין עדיין תקצירים לדקות האחרונות."}
    system, user = prompts.load(RECAP_PROMPT).render(
        minutes=minutes, chunk_summaries="\n".join(f"- {t}" for t in lines),
        concepts=", ".join(concepts) or "(none)", highlights=" · ".join(highlights) or "(none)")

    def check(v: Recap) -> str | None:
        found = set().union(*(forbidden_scripts(b) for b in v.bullets))
        if found:
            return f"The output contains {', '.join(sorted(found))} letters. Write only Hebrew and English."
        return None
    c = await chat_json(LIVE_MODEL, system, user, Recap, client=client, options=DIGEST_OPTIONS, check=check,
                        keep_alive=OLLAMA_KEEP_ALIVE, timeout_s=EXTRACT_TIMEOUT_S, backoff_s=backoff_s)
    store.log("recap", lecture_id=lecture_id, input_ref=f"{lecture_id}#{minutes}m",
              ms=(time.perf_counter() - t0) * 1000, tokens_in=c.tokens_in, tokens_out=c.tokens_out,
              output={"status": "ok" if c.value else "failed", "chunks": len(recent), "attempts": c.attempts,
                      "error": c.error})
    if c.value is None:
        return {"bullets": [], "minutes": minutes, "note": "המודל לא החזיר סיכום הפעם; נסה שוב בעוד רגע."}
    return {"bullets": c.value.bullets, "minutes": minutes, "note": ""}
