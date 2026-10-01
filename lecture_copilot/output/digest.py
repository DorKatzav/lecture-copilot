"""The Digest (PLAN.md §3.7, DESIGN_HE §Digest): nine sections in a fixed order, the same in every lecture.

Map-reduce over what the extractor already wrote, never over the transcript (D-M2-2). Gemma runs with a 4,096-token
context and a one-line chunk summary costs about 38 tokens, so a full lecture does not fit one call:
  map    — the chunk summaries are split into even blocks; each block (+ the concepts explained in it) becomes
           paragraphs of the full summary                                        prompts/digest_sections_v0.md
  reduce — the full summary (+ highlights, previous lecture) becomes the five executive bullets
                                                                                 prompts/digest_exec_v0.md
Everything else is a template over SQLite. A failed call never costs the student the Digest: a failed block falls
back to its chunk summaries, a failed executive summary is said in its section, and `degraded` names what failed.
"""

import json
import math
import re
import time
from dataclasses import dataclass, field

import httpx
from jinja2 import Environment, PackageLoader, StrictUndefined

from lecture_copilot import prompts
from lecture_copilot.agents.schemas import Continuation, DigestExec, DigestExecWithPrevious, DigestSection
from lecture_copilot.config import (
    DIGEST_EXEC_PROMPT,
    DIGEST_FULL_WORDS,
    DIGEST_MAP_INPUT_TOKENS,
    DIGEST_MAP_PROMPT,
    DIGEST_OPTIONS,
    DIGEST_TIMEOUT_S,
    OLLAMA_KEEP_ALIVE,
    TOKENS_PER_WORD,
    VERIFY_MIN_IMPORTANCE,
)
from lecture_copilot.config import DIGEST_MODEL as MODEL
from lecture_copilot.llm import chat_json
from lecture_copilot.scriptcheck import forbidden_scripts
from lecture_copilot.store.db import Store, new_id

SECTIONS = ["סיכום מנהלים", "סיכום מלא", "★ למבחן / הודגש", "המשך מ", "מושגים", "טענות מסומנות", "שאלות פתוחות",
            "משימות", "ההערות שלי"]
CLAIM_LABELS = {"pending": "עדיין לא נבדק", "unchecked": "לא נבדק — אין רשת", "skipped": "לא נבדק"}

_env = Environment(loader=PackageLoader("lecture_copilot.output", "templates"), undefined=StrictUndefined,
                   trim_blocks=True, lstrip_blocks=True, keep_trailing_newline=True, autoescape=False)


@dataclass(frozen=True)
class ConceptRow:
    term: str
    explanation: str
    canonical_key: str
    first_seen: str | None = None    # "W04" when an earlier lecture explained it (M3)


@dataclass(frozen=True)
class ClaimRow:
    text: str
    importance: int
    status: str
    label: str


@dataclass(frozen=True)
class TaskRow:
    text: str
    owner: str | None
    due: str | None


@dataclass
class DigestDoc:
    lecture_id: str
    course_name: str
    title: str
    date: str
    week: int | None
    minutes: int
    language: str
    exec_summary: list[str] = field(default_factory=list)
    full_summary: list[str] = field(default_factory=list)
    highlights: list[str] = field(default_factory=list)
    prev_title: str | None = None
    prev_lecture_id: str | None = None
    prev_bullets: list[str] = field(default_factory=list)
    continuation: Continuation | None = None
    concepts: list[ConceptRow] = field(default_factory=list)
    claims: list[ClaimRow] = field(default_factory=list)
    all_claims: list[dict] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    tasks: list[TaskRow] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    segments: list[dict] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)

    @property
    def returned(self) -> list[ConceptRow]:
        return [c for c in self.concepts if c.first_seen]

    @property
    def concepts_heading(self) -> str:
        """'מושגים (13, +2 שחזרו מ-W04)' — the count, and how many came back from which earlier lecture."""
        n = len(self.concepts)
        back = self.returned
        if not back:
            return f"מושגים ({n})"
        weeks = sorted({c.first_seen for c in back})
        verb = "שחזר" if len(back) == 1 else "שחזרו"
        return f"מושגים ({n}, +{len(back)} {verb} מ-{', '.join(weeks)})"

    @property
    def heading(self) -> str:
        y, m, d = (self.date.split("-") + ["", ""])[:3]
        date = f"{int(d)}.{int(m)}.{y}" if y.isdigit() and m.isdigit() and d.isdigit() else self.date
        week = f"W{self.week:02d} · " if self.week is not None else ""
        return f"{self.course_name} — {week}{self.title} · {date}"


@dataclass
class Block:
    lines: list[tuple[int, str]]
    concepts: list[tuple[str, str]]
    tokens: int


# ---------- planning ----------

def estimate_tokens(text: str) -> int:
    """Measured on the fixture with gemma3:12b: 2.86 tokens per Hebrew word; rounded up."""
    return len(text.split()) * TOKENS_PER_WORD


def target_words(words_in: int) -> int:
    """About 0.9 of the material, inside the spec's range for a full lecture (a short replay gets a short text:
    asking for 400 words about 150 words of input would make the model invent)."""
    lo, hi = DIGEST_FULL_WORDS
    return max(lo, min(hi, round(words_in * 0.9)))


def plan_blocks(lines: list[tuple[int, str]], concepts_by_chunk: dict[int, list[tuple[str, str]]],
                budget_tokens: int = DIGEST_MAP_INPUT_TOKENS) -> list[Block]:
    line_tokens = sum(estimate_tokens(f"{i}. {t}") for i, t in lines)
    n = max(1, math.ceil(line_tokens / (budget_tokens * 0.75)))        # a quarter is kept for the concepts
    size = math.ceil(len(lines) / n)
    blocks = []
    for k in range(n):
        part = lines[k * size:(k + 1) * size]
        if not part:
            continue
        used = sum(estimate_tokens(f"{i}. {t}") for i, t in part)
        concepts, seen = [], set()
        for i, _ in part:
            for term, expl in concepts_by_chunk.get(i, []):
                cost = estimate_tokens(f"{term} — {expl}")
                if term not in seen and used + cost <= budget_tokens:
                    concepts.append((term, expl))
                    seen.add(term)
                    used += cost
        blocks.append(Block(part, concepts, used))
    return blocks


# ---------- rendering ----------

def render_markdown(doc: DigestDoc) -> str:
    """The template separates sections with `@@`; they are joined by exactly one blank line."""
    parts = _env.get_template("digest.md.j2").render(doc=doc).split("@@")
    return "\n\n".join(p.strip() for p in parts) + "\n"


def section_headings(markdown: str) -> list[str]:
    """The `##` headings of a Digest, each reduced to the fixed section name it starts with."""
    out = []
    for h in re.findall(r"^## (.+)$", markdown, re.MULTILINE):
        out.append(next((s for s in SECTIONS if h.startswith(s)), h))
    return out


# ---------- the document ----------

_PLACEHOLDERS = {"none", "-", "—", "n/a", "אין", "אין.", "אף אחת", "אף אחד", "אין סתירות", "לא"}


def _clean(items: list[str]) -> list[str]:
    return [i.strip() for i in items if i.strip().lower() not in _PLACEHOLDERS]


def _script_check(texts, hebrew=None):
    """`texts`: every free-text field (no foreign scripts); `hebrew`: fields that must be written in Hebrew."""
    def check(value) -> str | None:
        found = set().union(*(forbidden_scripts(t) for t in texts(value)))
        if found:
            return f"The output contains {', '.join(sorted(found))} letters. Write only Hebrew and English."
        if hebrew:
            # a bare term (CAC, Git branch) is fine; a sentence must be Hebrew
            latin = [t for t in _clean(hebrew(value))
                     if len(t.split()) > 3 and not any("\u0590" <= c <= "\u05ff" for c in t)]
            if latin:
                return f"These items are not in Hebrew: {' | '.join(latin[:3])}. Write every item in Hebrew."
        return None
    return check


def lecture_label(row: dict | None) -> str | None:
    if row is None:
        return None
    return f"W{row['week']:02d} · {row['title']}" if row["week"] is not None else row["title"]


def _collect(lecture_id: str, store: Store) -> tuple[DigestDoc, dict[int, list[tuple[str, str]]]]:
    lec = store.lecture(lecture_id)
    course = store.course(lec["course_id"])
    segments = store.segments(lecture_id)
    doc = DigestDoc(lecture_id=lecture_id, course_name=course["name"], title=lec["title"], date=lec["date"],
                    week=lec["week"], language=course["language"], segments=segments,
                    minutes=round((max(s["t1"] for s in segments) - min(s["t0"] for s in segments)) / 60)
                    if segments else 0)
    prev_id = lec["continues_id"] or store.previous_lecture(lec["course_id"], lecture_id)
    if prev_id:
        prev = store.lecture(prev_id)
        store.set_continues(lecture_id, prev_id)
        doc.prev_lecture_id, doc.prev_title = prev_id, lecture_label(prev)
        row = store.con.execute("select bullets_json from lecture_summaries where lecture_id = ?",
                                (prev_id,)).fetchone()
        doc.prev_bullets = json.loads(row[0]) if row and row[0] else []
    weeks: dict[str, str] = {}
    by_chunk: dict[int, list[tuple[str, str]]] = {}
    seen = set()
    for it in store.items(lecture_id):
        if it["kind"] == "concept":
            by_chunk.setdefault(it["chunk_id"], []).append((it["text"], it["explanation"] or ""))
            key = it["canonical_key"] or it["text"]
            if key not in seen:
                seen.add(key)
                first = it["first_seen_lecture_id"]
                label = None
                if first and first != lecture_id:
                    if first not in weeks:
                        other = store.lecture(first)
                        weeks[first] = (f"W{other['week']:02d}" if other and other["week"] is not None
                                        else (other["title"] if other else "קודם"))
                    label = weeks[first]
                doc.concepts.append(ConceptRow(it["text"], it["explanation"] or "", key, label))
        elif it["kind"] == "highlight":
            doc.highlights.append(it["text"])
        elif it["kind"] == "question":
            doc.questions.append(it["text"])
        elif it["kind"] in ("action", "decision"):
            doc.tasks.append(TaskRow(it["text"], it["owner"], it["due"]))
        elif it["kind"] == "note":
            doc.notes.append(it["text"])
    doc.all_claims = store.claims(lecture_id)
    doc.claims = [ClaimRow(c["text"], c["importance"], c["status"], CLAIM_LABELS.get(c["status"], c["status"]))
                  for c in doc.all_claims if (c["importance"] or 0) >= VERIFY_MIN_IMPORTANCE]
    return doc, by_chunk


async def digest(lecture_id: str, *, store: Store, client: httpx.AsyncClient, backoff_s: float = 1.0,
                 map_prompt: str = DIGEST_MAP_PROMPT, exec_prompt: str = DIGEST_EXEC_PROMPT,
                 save: bool = True) -> DigestDoc:
    t0 = time.perf_counter()
    run_id = new_id()
    doc, by_chunk = _collect(lecture_id, store)
    lines = store.chunk_summaries(lecture_id)
    blocks = plan_blocks(lines, by_chunk) if lines else []
    common = dict(language=doc.language, course_name=doc.course_name, lecture_title=doc.title)

    async def call(step: str, prompt: str, schema, texts, fallback=None, hebrew=None, **values):
        """`fallback`: a looser schema tried on the last raw answer when the strict one failed twice."""
        system, user = prompts.load(prompt).render(**common, **values)
        c = await chat_json(MODEL, system, user, schema, client=client, options=DIGEST_OPTIONS,
                            check=_script_check(texts, hebrew), keep_alive=OLLAMA_KEEP_ALIVE,
                            timeout_s=DIGEST_TIMEOUT_S, backoff_s=backoff_s)
        value, degraded = c.value, step
        if value is None and fallback and c.raw:
            try:
                value, degraded = fallback.model_validate_json(c.raw[-1]), "continuation"
            except ValueError:
                pass
        store.log("digest", lecture_id=lecture_id, input_ref=f"{run_id}#{step}", ms=round(sum(c.ms), 1),
                  tokens_in=c.tokens_in, tokens_out=c.tokens_out,
                  output={"status": "ok" if c.value else ("degraded" if value else "failed"), "model": MODEL,
                          "prompt": prompt, "attempts": c.attempts, "error": c.error})
        if c.value is None:
            doc.degraded.append(degraded)
        return value

    if not lines:
        doc.degraded.append("empty")
    total_words = target_words(sum(len(t.split()) for _, t in lines))
    for k, b in enumerate(blocks, 1):
        section = await call(
            f"map-{k}", map_prompt, DigestSection, lambda v: v.paragraphs, part=k, parts=len(blocks),
            chunk_summaries="\n".join(f"{i}. {t}" for i, t in b.lines),
            concepts="\n".join(f"{term} — {expl}" for term, expl in b.concepts) or "(none)",
            words=max(40, round(total_words * len(b.lines) / len(lines))))
        doc.full_summary += section.paragraphs if section else [" ".join(t for _, t in b.lines)]
    if lines:
        schema = DigestExecWithPrevious if doc.prev_title else DigestExec
        def cont_items(v):
            return (v.continuation.new + v.continuation.repeated + v.continuation.contradicts) if v.continuation else []
        ex = await call("exec", exec_prompt, schema, lambda v: v.exec_summary + cont_items(v), hebrew=cont_items,
                        fallback=DigestExec if doc.prev_title else None, date=doc.date,
                        minutes=doc.minutes, full_summary="\n\n".join(doc.full_summary),
                        highlights=" · ".join(doc.highlights) or "(none)", prev_title=doc.prev_title or "none",
                        prev_bullets="\n".join(f"- {b}" for b in doc.prev_bullets) if doc.prev_bullets else "(none)")
        doc.exec_summary = ex.exec_summary if ex else []
        doc.continuation = ex.continuation if ex else None
        if doc.continuation:
            c = doc.continuation
            doc.continuation = Continuation(new=_clean(c.new), repeated=_clean(c.repeated),
                                            contradicts=_clean(c.contradicts))
    markdown = render_markdown(doc)
    if save:
        store.save_digest(lecture_id, bullets=doc.exec_summary, digest_md=markdown)
    store.log("digest", lecture_id=lecture_id, input_ref=f"{run_id}#digest", ms=(time.perf_counter() - t0) * 1000,
              output={"status": "ok" if not doc.degraded else "degraded", "degraded": doc.degraded,
                      "blocks": len(blocks), "chunks": len(lines), "minutes": doc.minutes,
                      "words": len(markdown.split()), "total_s": round(time.perf_counter() - t0, 2),
                      "sections": section_headings(markdown)})
    return doc
