"""Verifier (PLAN.md §3.5, DESIGN_HE §flow 7): Gemini checks one material claim at a time, in its own worker,
never in the chunk loop. The only things that leave the machine are the claim's text and what the course said
about it earlier (D-M4-1: injected, not a tool — the API needs a special mode to mix search with functions).

    pending ──verify──► verified            (verdict, confidence, explanation, sources; cached by normalized text)
            ──offline/failed──► unchecked   (retried once when the lecture ends)
    below the importance threshold, or fact-checking off ──► skipped
"""

import asyncio
import hashlib
from typing import Protocol

from pydantic import ValidationError

from lecture_copilot import prompts
from lecture_copilot.agents.schemas import Verdict
from lecture_copilot.config import GEMINI_HOST, VERIFIER_MODEL, VERIFY_CONCURRENCY, VERIFY_MIN_IMPORTANCE
from lecture_copilot.store.db import Store
from lecture_copilot.store.net import Net, NetError, Offline, gemini_cost

PROMPT = "verifier_v1"


class VerifierBackend(Protocol):
    async def generate(self, system: str, user: str) -> tuple[str, dict]:
        """Returns (json text, usage) — usage: tokens_in, tokens_out, bytes_out, bytes_in, grounding: [url]."""


def cache_key(normalized: str) -> str:
    return hashlib.sha256(" ".join(normalized.lower().split()).encode()).hexdigest()


class Verifier:
    def __init__(self, store: Store, net: Net, backend: VerifierBackend, model: str = VERIFIER_MODEL):
        self.store, self.net, self.backend, self.model = store, net, backend, model

    def _context(self, claim: dict, course_id: str) -> str:
        hits = self.store.search(claim["normalized"] or claim["text"], course_id, k=3, before=claim["lecture_id"])
        lines = [f"- {h.text}" + (f" (W{h.week:02d})" if h.week is not None else "") for h in hits]
        return "\n".join(lines) if lines else "(nothing relevant was said earlier)"

    async def verify(self, claim_id: str) -> Verdict | None:
        claim = self.store.claim(claim_id)
        lec = self.store.lecture(claim["lecture_id"])
        course = self.store.course(lec["course_id"])
        key = cache_key(claim["normalized"] or claim["text"])
        cached = self.store.cached_verdict(key)
        if cached:
            v = Verdict.model_validate(cached)
            self._store(claim_id, v, key)
            self.store.log("verifier", lecture_id=claim["lecture_id"], input_ref=claim_id,
                           output={"status": "verified", "cache": True, "verdict": v.verdict})
            return v
        system, user = prompts.load(PROMPT).render(
            text=claim["text"], normalized=claim["normalized"] or claim["text"], course_name=course["name"],
            lecture_title=lec["title"], language=course["language"], course_context=self._context(claim, course["id"]))
        last_error = None
        for attempt in (1, 2):
            async def call():
                text, usage = await self.backend.generate(system, user)
                usage["cost_usd"] = gemini_cost(usage.get("tokens_in", 0), usage.get("tokens_out", 0))
                return (text, usage.get("grounding", [])), usage
            try:
                text, grounding = await self.net.call(GEMINI_HOST, call, lecture_id=claim["lecture_id"], ref=claim_id)
            except Offline as e:
                return self._unchecked(claim, "offline", str(e))
            except NetError as e:
                return self._unchecked(claim, "failed", str(e))
            try:
                v = Verdict.model_validate_json(text)
            except ValidationError as e:
                last_error = f"invalid verdict: {str(e)[:120]}"
                continue
            v.sources = list(dict.fromkeys([*v.sources, *grounding]))[:6]
            self.store.cache_verdict(key, v.model_dump())
            self._store(claim_id, v, key)
            self.store.log("verifier", lecture_id=claim["lecture_id"], input_ref=claim_id,
                           output={"status": "verified", "cache": False, "verdict": v.verdict, "attempts": attempt,
                                   "model": self.model, "prompt": PROMPT})
            return v
        return self._unchecked(claim, "failed", last_error or "no answer")

    def _store(self, claim_id: str, v: Verdict, key: str) -> None:
        self.store.set_verdict(claim_id, status="verified", verdict=v.verdict, confidence=v.confidence,
                               explanation=v.explanation, sources=v.sources, cache_key=key)

    def _unchecked(self, claim: dict, why: str, error: str) -> None:
        self.store.set_verdict(claim["id"], status="unchecked")
        self.store.log("verifier", lecture_id=claim["lecture_id"], input_ref=claim["id"],
                       output={"status": "unchecked", "why": why, "error": error[:200]})
        return None


class VerifierWorker:
    """Consumes claim ids as the lecture goes; `finish()` at "סיום": drain, retry what went unchecked once,
    and mark what was never material as skipped."""

    def __init__(self, verifier: Verifier, lecture_id: str, concurrency: int = VERIFY_CONCURRENCY):
        self.verifier, self.lecture_id = verifier, lecture_id
        self.queue: asyncio.Queue[str | None] = asyncio.Queue()
        self.sem = asyncio.Semaphore(concurrency)
        self.tasks: set[asyncio.Task] = set()

    def enqueue(self, claim_id: str) -> None:
        self.queue.put_nowait(claim_id)

    async def _one(self, claim_id: str) -> None:
        async with self.sem:
            try:
                await self.verifier.verify(claim_id)
            except Exception as e:  # a bug in one claim must not kill the worker
                self.verifier.store.log("verifier", lecture_id=self.lecture_id, input_ref=claim_id,
                                        output={"status": "crashed", "error": f"{type(e).__name__}: {e}"})
            finally:
                self.queue.task_done()

    async def run(self) -> None:
        while (claim_id := await self.queue.get()) is not None:
            task = asyncio.create_task(self._one(claim_id))
            self.tasks.add(task)
            task.add_done_callback(self.tasks.discard)
        self.queue.task_done()

    async def finish(self) -> dict:
        await self.queue.join()
        if self.tasks:
            await asyncio.gather(*self.tasks)
        store = self.verifier.store
        unchecked = [c["id"] for c in store.claims(self.lecture_id) if c["status"] == "unchecked"]
        for claim_id in unchecked:                      # batch at stop — the network may be back
            await self.verifier.verify(claim_id)
        store.set_claim_status(self.lecture_id, "pending", "skipped")
        self.queue.put_nowait(None)
        rows = store.claims(self.lecture_id)
        return {"verified": sum(r["status"] == "verified" for r in rows),
                "unchecked": sum(r["status"] == "unchecked" for r in rows),
                "skipped": sum(r["status"] == "skipped" for r in rows), "retried": len(unchecked)}


def material(claims: list[dict]) -> list[dict]:
    return [c for c in claims if (c["importance"] or 0) >= VERIFY_MIN_IMPORTANCE]
