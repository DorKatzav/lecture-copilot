"""Memory (PLAN.md §3.5, DESIGN_HE §memory): the course remembers what was said.

Per chunk, inside the budget (embeddings ≤ 2 s, search ≤ 1 s), both steps answer with one batched embedding call:
  recall   — before extraction: the chunk's text is embedded and searched (FTS5 + vectors, RRF, top-5) over the
             course's earlier lectures; the hits go into the extraction prompt as known terms and earlier claims,
             so the extractor can mark a claim that contradicts one of them.
  remember — after extraction: the chunk's concepts and claims are embedded and indexed; a concept that an earlier
             lecture explained (same key, same term, or close enough in meaning) keeps that lecture as
             `first_seen_lecture_id`; a claim the extractor marked `contradicts` gains importance and links to
             the earlier claim.
Nothing here raises: an embedding failure is logged, text search still works, and the lecture goes on.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from lecture_copilot.agents.schemas import ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.config import ALREADY_SAID_COSINE, CONTRADICTION_BONUS, MEMORY_K
from lecture_copilot.store.embed import EmbedError, cosine, embed, unpack
from lecture_copilot.store.search import MemoryHit

if TYPE_CHECKING:
    from lecture_copilot.audio.sources import AudioChunk
    from lecture_copilot.pipeline import Ctx


@dataclass
class MemoryContext:
    hits: list[MemoryHit] = field(default_factory=list)
    embed_error: str | None = None

    @property
    def known_terms(self) -> str:
        rows = [f"{h.text} — {h.explanation or ''} ({_where(h)})" for h in self.hits if h.kind == "concept"]
        return "\n".join(rows) if rows else "(none yet)"

    @property
    def previous_claims(self) -> str:
        rows = [f"{h.text} ({_where(h)})" for h in self.hits if h.kind == "claim"]
        return "\n".join(rows) if rows else "(none)"


def _where(h: MemoryHit) -> str:
    return f"W{h.week:02d}" if h.week is not None else (h.title or "earlier")


def _norm(s: str | None) -> str:
    return " ".join((s or "").lower().split())


async def recall(segments: list[Segment], ctx: Ctx, ref: str) -> MemoryContext:
    t0 = time.perf_counter()
    text = " ".join(s.text for s in segments)
    mem = MemoryContext()
    query_vec = None
    try:
        (query_vec,) = await embed([text], client=ctx.ollama, backoff_s=ctx.backoff_s)
    except EmbedError as e:
        mem.embed_error = str(e)
    embed_ms = (time.perf_counter() - t0) * 1000
    mem.hits = ctx.store.search(text, ctx.course_id, k=MEMORY_K, query_vec=query_vec, before=ctx.lecture_id)
    ctx.store.log("memory", lecture_id=ctx.lecture_id, input_ref=ref, ms=(time.perf_counter() - t0) * 1000,
                  output={"step": "recall", "hits": len(mem.hits), "embed_ms": round(embed_ms, 1),
                          "embed_error": mem.embed_error,
                          "top": [(h.kind, h.text[:40], h.score) for h in mem.hits[:3]]})
    return mem


async def remember(result: ExtractResult, chunk: AudioChunk, ctx: Ctx, ref: str,
                   memory: MemoryContext | None = None) -> dict:
    t0 = time.perf_counter()
    store = ctx.store
    concepts = [r for r in store.items(ctx.lecture_id, kind="concept") if r["chunk_id"] == chunk.idx]
    claims = [r for r in store.claims(ctx.lecture_id) if r["segment_id"] == (concepts[0]["segment_id"]
                                                                               if concepts else None)]
    if not concepts:
        seg = store.con.execute("select id from segments where lecture_id = ? and chunk_id = ? order by t0 limit 1",
                                (ctx.lecture_id, chunk.idx)).fetchone()
        claims = [r for r in store.claims(ctx.lecture_id) if seg and r["segment_id"] == seg[0]]
    texts = [f"{r['text']} — {r['explanation'] or ''}" for r in concepts] + [r["normalized"] or r["text"]
                                                                            for r in claims]
    vecs: list[list[float]] = []
    embed_error = None
    try:
        vecs = await embed(texts, client=ctx.ollama, backoff_s=ctx.backoff_s) if texts else []
    except EmbedError as e:
        embed_error = str(e)
    embed_ms = (time.perf_counter() - t0) * 1000
    for row, vec in zip(concepts + claims, vecs, strict=False):
        store.set_embedding("items" if "kind" in row else "claims", row["id"], vec)

    # already said: same key, same term, or close enough in meaning to a concept of an earlier lecture
    earlier = store.earlier_concepts(ctx.course_id, ctx.lecture_id)
    by_key = {_norm(e["canonical_key"]): e for e in reversed(earlier) if e["canonical_key"]}
    by_term = {_norm(e["text"]): e for e in reversed(earlier)}
    already = 0
    for i, row in enumerate(concepts):
        match = by_key.get(_norm(row["canonical_key"])) or by_term.get(_norm(row["text"]))
        if match is None and i < len(vecs):
            hits = store.search("", ctx.course_id, k=1, query_vec=vecs[i], before=ctx.lecture_id)
            if hits and hits[0].kind == "concept":
                hit_vec = store.con.execute("select embedding from items where id = ?", (hits[0].id,)).fetchone()[0]
                if hit_vec and cosine(vecs[i], unpack(hit_vec)) >= ALREADY_SAID_COSINE:
                    match = {"first_seen_lecture_id": hits[0].lecture_id, "lecture_id": hits[0].lecture_id}
        if match:
            store.set_first_seen(row["id"], match["first_seen_lecture_id"] or match["lecture_id"])
            already += 1

    # contradictions: the extractor marked them against the claims it was shown
    recalled = {_norm(h.text): h for h in (memory.hits if memory else []) if h.kind == "claim"}
    contradictions = 0
    by_text = {r["text"]: r for r in claims}
    for c in result.claims:
        row = by_text.get(c.text)
        if c.contradicts and row:
            hit = recalled.get(_norm(c.contradicts))
            store.set_contradiction(row["id"], hit.id if hit else None, CONTRADICTION_BONUS)
            contradictions += 1
    out = {"step": "remember", "embedded": len(vecs), "embed_ms": round(embed_ms, 1), "embed_error": embed_error,
           "already_said": already, "new_concepts": len(concepts) - already, "contradictions": contradictions}
    store.log("memory", lecture_id=ctx.lecture_id, input_ref=ref, ms=(time.perf_counter() - t0) * 1000, output=out)
    return out
