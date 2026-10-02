"""The one running lecture (DESIGN_HE §screens). One process, asyncio tasks: the pipeline consumes a source
(microphone or file), the verifier works beside it, and "סיום" runs the Digest and the sinks. The web page only
ever reads this state and SQLite; nothing here notifies anyone.

    idle ──record/replay──► recording ──stop──► digesting ──► digested (= idle again, with the result shown)
A lecture left in `recording` by a crash is listed as interrupted; `resume` finishes its Digest from the chunks
that reached the disk.
"""

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from lecture_copilot.agents.memory import MemoryHit
from lecture_copilot.agents.ranker import rank
from lecture_copilot.agents.recap import recap as recap_agent
from lecture_copilot.agents.verifier import Verifier, VerifierWorker
from lecture_copilot.cli import make_digest
from lecture_copilot.config import COURSES_ROOT, DB_PATH, OLLAMA_URL, ROOT, RUNS_DIR, Profile, load_env
from lecture_copilot.output.course_page import course_page
from lecture_copilot.output.sinks import FolderSink
from lecture_copilot.pipeline import Ctx, run, warm_up
from lecture_copilot.store.db import Store, new_id
from lecture_copilot.store.embed import EmbedError, embed
from lecture_copilot.store.net import Net

MARK_WINDOW_S = 30


def _dedupe(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for r in rows:
        if r["text"] not in seen:
            seen.add(r["text"])
            out.append(r)
    return out


@dataclass
class Current:
    lecture_id: str
    course_id: str
    course: str
    title: str
    source: str
    fact_check: bool
    started: float
    status: str = "recording"          # recording | digesting | digested | failed
    chunks: int = 0
    last_chunk: dict | None = None
    queue_depth: int = 0
    summary: dict | None = None
    digest: dict | None = None
    error: str | None = None
    task: asyncio.Task | None = None
    live: object | None = None         # the LiveSource, for the meter and stop()
    log: list[str] = field(default_factory=list)


class Session:
    def __init__(self, *, db: Path = DB_PATH, courses_root: Path = COURSES_ROOT, runs_dir: Path = RUNS_DIR,
                 ollama_client_factory: Callable | None = None, asr_factory: Callable | None = None,
                 source_factory: Callable | None = None, gemini_factory: Callable | None = None,
                 course_name: str = "", backoff_s: float = 1.0, env_file: Path = ROOT / ".env"):
        self.store = Store(db)
        self.sink = FolderSink(courses_root)
        self.runs_dir = Path(runs_dir)
        self.ollama_client_factory = ollama_client_factory or self._default_client
        self.asr_factory = asr_factory or self._default_asr
        self.source_factory = source_factory or self._default_source
        self.gemini_factory = gemini_factory
        self.env_file = env_file
        self.backoff_s = backoff_s
        self.current: Current | None = None
        self.listeners: set[asyncio.Queue] = set()
        self.checks: dict = {"ollama": None, "mw": None}
        if course_name:
            self.store.upsert_course(course_name, language="he")

    # ---------- defaults (the real providers) ----------

    @staticmethod
    def _default_client():
        import httpx
        return httpx.AsyncClient(base_url=OLLAMA_URL)

    @staticmethod
    def _default_asr(kind: str):
        """The ASR follows the source: a transcript's pseudo-chunks are read back, audio goes to MacWhisper."""
        if kind == "transcript":
            from lecture_copilot.asr.transcript import TranscriptASR
            return TranscriptASR()
        from lecture_copilot.asr.macwhisper import MacWhisperASR
        return MacWhisperASR()

    def _default_source(self, lecture_id: str, kind: str, file: Path | None = None, pace: str = "fast"):
        from lecture_copilot.audio.sources import FileSource, LiveSource
        from lecture_copilot.audio.transcript import TranscriptSource
        if kind == "mic":
            return LiveSource(lecture_id, runs_dir=self.runs_dir)
        if kind == "transcript":
            return TranscriptSource(lecture_id, file, runs_dir=self.runs_dir, pace=pace)
        return FileSource(lecture_id, file, pace, runs_dir=self.runs_dir)

    def _gemini(self, fact_check: bool):
        if not fact_check:
            return None
        if self.gemini_factory:
            return self.gemini_factory()
        load_env(self.env_file, fact_check=True)
        from lecture_copilot.agents.verifier import GeminiAPI
        return GeminiAPI(os.environ["GEMINI_API_KEY"])

    # ---------- events ----------

    def _notify(self, kind: str, **data) -> None:
        for q in list(self.listeners):
            q.put_nowait({"event": kind, **data})

    # ---------- state ----------

    def courses(self) -> list[str]:
        return [r[0] for r in self.store.con.execute("select name from courses order by name")]

    def interrupted(self) -> list[dict]:
        rows = self.store.con.execute("select id, title, started_at from lectures where status = 'recording' "
                                      "order by started_at").fetchall()
        out = []
        for r in rows:
            if self.current and self.current.lecture_id == r["id"] and self.current.status == "recording":
                continue
            out.append({"lecture_id": r["id"], "title": r["title"], "started_at": r["started_at"],
                        "chunks": self.store.con.execute("select count(distinct chunk_id) from segments "
                                                         "where lecture_id = ?", (r["id"],)).fetchone()[0]})
        return out

    def dashboard(self, lecture_id: str) -> dict:
        items = self.store.items(lecture_id)
        concepts, seen = [], set()
        for it in items:
            if it["kind"] == "concept":
                key = (it["canonical_key"] or it["text"]).lower()
                if key in seen:
                    continue
                seen.add(key)
                first = it["first_seen_lecture_id"]
                concepts.append({"term": it["text"], "explanation": it["explanation"],
                                 "returned": bool(first and first != lecture_id)})
        return {
            "concepts": concepts,
            "claims": [{"text": c.text, "importance": c.importance, "label": c.label, "status": c.status,
                        "verdict": c.verdict, "verdict_he": c.verdict_he, "explanation": c.explanation,
                        "sources": c.sources[:2]} for c in rank(lecture_id, self.store) if c.importance >= 70],
            "highlights": _dedupe([{"text": it["text"], "user": it["owner"] == "user"} for it in items
                                   if it["kind"] == "highlight"]),
            "questions": list(dict.fromkeys(it["text"] for it in items if it["kind"] == "question")),
            "tasks": [{"text": it["text"], "due": it["due"]} for it in items if it["kind"] in ("action", "decision")],
            "notes": [{"text": it["text"], "t": it["due"]} for it in items if it["kind"] == "note"],
        }

    def previous_bullets(self, course_id: str) -> list[dict]:
        rows = self.store.con.execute(
            "select l.id, l.title, l.week, s.bullets_json from lectures l join lecture_summaries s "
            "on s.lecture_id = l.id where l.course_id = ? order by l.date desc, l.started_at desc limit 1",
            (course_id,)).fetchall()
        import json
        return [{"lecture_id": r["id"], "title": r["title"], "week": r["week"],
                 "bullets": json.loads(r["bullets_json"] or "[]")} for r in rows]

    def state(self) -> dict:
        cur = self.current
        current = None
        if cur:
            net = self.store.con.execute("select coalesce(sum(cost_usd), 0) from decisions where node = 'net' "
                                         "and lecture_id = ?", (cur.lecture_id,)).fetchone()[0]
            current = {"lecture_id": cur.lecture_id, "course": cur.course, "title": cur.title, "source": cur.source,
                       "fact_check": cur.fact_check, "status": cur.status, "chunks": cur.chunks,
                       "elapsed_s": round(time.time() - cur.started), "last_chunk": cur.last_chunk,
                       "queue_depth": cur.queue_depth, "level": getattr(cur.live, "level", None),
                       "digest": cur.digest, "verifier": (cur.summary or {}).get("verifier"),
                       "cost_usd": round(float(net), 4), "error": cur.error, "log": cur.log[-6:]}
        course_id = cur.course_id if cur else (self.store.con.execute("select id from courses order by name limit 1")
                                               .fetchone() or [None])[0]
        return {"ready": True, "checks": self.checks, "courses": self.courses(), "current": current,
                "dashboard": self.dashboard(cur.lecture_id) if cur else None,
                "previous": self.previous_bullets(course_id) if course_id else [],
                "interrupted": self.interrupted()}

    # ---------- actions ----------

    async def start(self, course: str, title: str, fact_check: bool, source: str = "mic",
                    file: Path | None = None, pace: str = "fast") -> str:
        if self.current and self.current.status in ("recording", "digesting"):
            raise RuntimeError("a lecture is already running")
        course_id = self.store.upsert_course(course, language="he")
        lang = self.store.course(course_id)["language"]
        audio_path = str(file) if file else f"mic:{datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%S')}"
        src = "transcript" if source == "transcript" else ("mic" if source == "mic" else "file")
        lecture_id = self.store.upsert_lecture(course_id, audio_path=audio_path, source=src, title=title,
                                               date=datetime.now(UTC).date().isoformat(), fact_check=fact_check)
        gemini = self._gemini(fact_check)
        cur = Current(lecture_id, course_id, course, title, src, bool(gemini), time.time())
        self.current = cur
        cur.task = asyncio.create_task(self._run(cur, lang, gemini, source, file, pace))
        self._notify("changed")
        return lecture_id

    async def _run(self, cur: Current, lang: str, gemini, source: str, file: Path | None, pace: str) -> None:
        try:
            async with self.ollama_client_factory() as client:
                verifier = VerifierWorker(Verifier(self.store, Net(self.store), gemini), cur.lecture_id) if gemini \
                    else None
                ctx = Ctx(lecture_id=cur.lecture_id, course_id=cur.course_id, course_name=cur.course,
                          lecture_title=cur.title, profile=Profile(fact_check=bool(gemini), language=lang),
                          store=self.store, asr=self.asr_factory(source), ollama=client, run_id=new_id(),
                          runs_dir=self.runs_dir, backoff_s=self.backoff_s, verifier=verifier)
                await warm_up(ctx)
                src = self.source_factory(cur.lecture_id, source, file, pace)
                cur.live = src

                def on_chunk(o: dict) -> None:
                    cur.chunks += 1
                    cur.last_chunk, cur.queue_depth = o, o.get("queue_depth", 0)
                    self._notify("changed")
                cur.summary = await run(src, ctx, on_chunk=on_chunk,
                                        extra=lambda: {"via": "web", "source_kind": source, "pace": pace,
                                                       "file": file.name if file else None})
                self.store.end_lecture(cur.lecture_id)
                cur.status = "digesting"
                self._notify("changed")
                cur.digest = await make_digest(cur.lecture_id, self.store, client, self.sink, cur.log.append)
                self.sink.write_course(course_page(cur.course_id, self.store))
                cur.status = "digested"
        except Exception as e:                        # the page shows it; nothing pops up
            cur.status, cur.error = "failed", f"{type(e).__name__}: {e}"
            self.store.log("run", lecture_id=cur.lecture_id, input_ref="session", output={"status": "failed",
                                                                                           "error": cur.error})
        finally:
            self._notify("changed")

    async def stop(self) -> None:
        cur = self.current
        if not cur or cur.status != "recording":
            raise RuntimeError("no lecture is recording")
        if cur.live is not None and hasattr(cur.live, "stop"):
            cur.live.stop()
        self._notify("changed")

    async def resume(self, lecture_id: str) -> None:
        """A lecture the process died on: finish the Digest from the chunks that reached the disk."""
        lec = self.store.lecture(lecture_id)
        if lec is None or lec["status"] != "recording":
            raise RuntimeError("nothing to resume")
        course = self.store.course(lec["course_id"])
        cur = Current(lecture_id, lec["course_id"], course["name"], lec["title"], lec["source"], False, time.time(),
                      status="digesting")
        cur.chunks = self.store.con.execute("select count(distinct chunk_id) from segments where lecture_id = ?",
                                            (lecture_id,)).fetchone()[0]
        self.current = cur

        async def finish() -> None:
            try:
                self.store.set_claim_status(lecture_id, "pending", "skipped")
                self.store.end_lecture(lecture_id)
                self.store.log("run", lecture_id=lecture_id, input_ref="resume",
                               output={"status": "resumed", "chunks": cur.chunks, "via": "web"})
                async with self.ollama_client_factory() as client:
                    cur.digest = await make_digest(lecture_id, self.store, client, self.sink, cur.log.append)
                    self.sink.write_course(course_page(cur.course_id, self.store))
                cur.status = "digested"
            except Exception as e:
                cur.status, cur.error = "failed", f"{type(e).__name__}: {e}"
            finally:
                self._notify("changed")
        cur.task = asyncio.create_task(finish())
        self._notify("changed")

    def mark(self) -> bool:
        """★: the last 30 s become a highlight of the student's own (kind=highlight, owner=user)."""
        cur = self.current
        if not cur or cur.status != "recording":
            return False
        seg = self.store.con.execute("select id, chunk_id, chunk_summary, t1 from segments where lecture_id = ? "
                                     "order by t1 desc limit 1", (cur.lecture_id,)).fetchone()
        if seg is None:
            return False
        since = seg["t1"] - MARK_WINDOW_S
        text = seg["chunk_summary"] or " ".join(
            r[0] for r in self.store.con.execute("select text from segments where lecture_id = ? and t1 > ? "
                                                 "order by t0", (cur.lecture_id, since)))
        self.store.add_item(cur.lecture_id, seg["id"], "highlight", text or "★", owner="user", t0=since)
        self._notify("changed")
        return True

    def note(self, text: str) -> bool:
        cur = self.current
        if not cur or cur.status != "recording" or not text.strip():
            return False
        seg = self.store.con.execute("select id, t1 from segments where lecture_id = ? order by t1 desc limit 1",
                                     (cur.lecture_id,)).fetchone()
        t = round(time.time() - cur.started)
        stamp = f"{t // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"
        self.store.add_item(cur.lecture_id, seg["id"] if seg else None, "note", text.strip(), owner="user",
                            due=stamp, t0=seg["t1"] if seg else 0.0)
        self._notify("changed")
        return True

    async def recap(self, minutes: int = 5) -> dict:
        cur = self.current
        if not cur:
            return {"bullets": [], "minutes": minutes, "note": "אין הרצאה פעילה."}
        async with self.ollama_client_factory() as client:
            return await recap_agent(cur.lecture_id, store=self.store, client=client, minutes=minutes,
                                     backoff_s=self.backoff_s)

    async def search(self, query: str, k: int = 8) -> list[dict]:
        cur = self.current
        course_id = cur.course_id if cur else (self.store.con.execute("select id from courses order by name limit 1")
                                               .fetchone() or [None])[0]
        if not course_id or not query.strip():
            return []
        vec = None
        try:
            async with self.ollama_client_factory() as client:
                (vec,) = await embed([query], client=client, backoff_s=self.backoff_s)
        except EmbedError:
            pass
        hits: list[MemoryHit] = self.store.search(query, course_id, k=k, query_vec=vec)
        return [{"kind": h.kind, "text": h.text, "explanation": h.explanation, "week": h.week, "title": h.title,
                 "score": h.score} for h in hits]

    def close(self) -> None:
        self.store.close()
