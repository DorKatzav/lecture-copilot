"""The one SQLite file (PLAN.md §3.6, schema = DESIGN_HE §data model). It is the only state; the UI hydrates from it.

IDs are ULIDs minted in Python before any write. Replaying the same file is an upsert: the lecture is found by its
natural key (course, source, audio path) and keeps its id; each chunk's rows are replaced in one transaction, so a
replay never duplicates and a crash loses at most the chunk in flight. `decisions` is an append-only log.
"""

import json
import secrets
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from lecture_copilot.agents.schemas import ExtractResult
from lecture_copilot.asr.base import Segment

SCHEMA_VERSION = 2   # 2: decisions.node gains digest, sink (M2)
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
# spec nodes + asr / chunk / run (D-M1-1: per-chunk status and timing live in the log, not in a new table)
NODES = ("extractor", "memory", "verifier", "ranker", "net", "asr", "chunk", "run", "digest", "sink")

DECISIONS = f"""CREATE TABLE IF NOT EXISTS decisions (
    id TEXT PRIMARY KEY, lecture_id TEXT,
    node TEXT NOT NULL CHECK (node IN ({", ".join(f"'{n}'" for n in NODES)})),
    input_ref TEXT, output_json TEXT, ms REAL, tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL,
    ts TEXT NOT NULL);"""

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS courses (
    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, folder TEXT,
    language TEXT NOT NULL CHECK (language IN ('he', 'en')), is_meetings INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS lectures (
    id TEXT PRIMARY KEY, course_id TEXT NOT NULL, week INTEGER, date TEXT, title TEXT,
    source TEXT NOT NULL CHECK (source IN ('mic', 'zoom', 'file', 'transcript')),
    fact_check INTEGER NOT NULL DEFAULT 1, continues_id TEXT, started_at TEXT, ended_at TEXT, audio_path TEXT,
    status TEXT NOT NULL CHECK (status IN ('recording', 'ended', 'digested')),
    UNIQUE (course_id, source, audio_path));
CREATE TABLE IF NOT EXISTS segments (
    id TEXT PRIMARY KEY, lecture_id TEXT NOT NULL, chunk_id INTEGER NOT NULL, t0 REAL NOT NULL, t1 REAL NOT NULL,
    text TEXT NOT NULL, speaker TEXT, asr TEXT, chunk_summary TEXT);
CREATE INDEX IF NOT EXISTS segments_chunk ON segments (lecture_id, chunk_id);
CREATE TABLE IF NOT EXISTS items (
    id TEXT PRIMARY KEY, lecture_id TEXT NOT NULL, segment_id TEXT,
    kind TEXT NOT NULL CHECK (kind IN ('concept', 'question', 'action', 'highlight', 'note', 'decision')),
    text TEXT NOT NULL, explanation TEXT, canonical_key TEXT, owner TEXT, due TEXT, first_seen_lecture_id TEXT,
    t0 REAL, embedding BLOB);
CREATE INDEX IF NOT EXISTS items_lecture ON items (lecture_id);
CREATE INDEX IF NOT EXISTS items_segment ON items (segment_id);
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY, lecture_id TEXT NOT NULL, segment_id TEXT, text TEXT NOT NULL, normalized TEXT,
    importance INTEGER, status TEXT NOT NULL CHECK (status IN ('pending', 'verified', 'skipped', 'unchecked')),
    verdict TEXT, confidence REAL, sources_json TEXT, cache_key TEXT, embedding BLOB);
CREATE INDEX IF NOT EXISTS claims_lecture ON claims (lecture_id);
CREATE INDEX IF NOT EXISTS claims_segment ON claims (segment_id);
CREATE TABLE IF NOT EXISTS lecture_summaries (
    lecture_id TEXT PRIMARY KEY, bullets_json TEXT, digest_md TEXT, embedding BLOB);
CREATE TABLE IF NOT EXISTS fact_cache (
    cache_key TEXT PRIMARY KEY, verdict TEXT, sources_json TEXT, checked_at TEXT);
{DECISIONS}
CREATE INDEX IF NOT EXISTS decisions_lecture ON decisions (lecture_id, node);
"""


_last = [0, 0]  # (ms, random part) of the last id minted from the clock


def new_id(now_ms: int | None = None) -> str:
    """ULID: 48-bit millisecond time + 80 random bits, Crockford base32, 26 chars. Ids minted from the clock are
    monotonic: within one millisecond the random part counts up, so ids sort in the order they were minted."""
    if now_ms is None:
        ms, rand = int(time.time() * 1000), secrets.randbits(80)
        if ms <= _last[0]:
            ms, rand = _last[0], _last[1] + 1
        _last[:] = ms, rand
    else:
        ms, rand = now_ms, secrets.randbits(80)
    n = (ms << 80) | rand
    return "".join(CROCKFORD[(n >> (5 * i)) & 31] for i in reversed(range(26)))


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds")


class Store:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.con = sqlite3.connect(path)
        self.con.row_factory = sqlite3.Row
        self.con.execute("pragma journal_mode = wal")
        self._migrate()
        self.con.executescript(SCHEMA)
        self.con.execute(f"pragma user_version = {SCHEMA_VERSION}")

    def _migrate(self) -> None:
        """SQLite cannot change a CHECK in place: when the allowed nodes grew, rebuild `decisions` and keep rows."""
        version = self.con.execute("pragma user_version").fetchone()[0]
        exists = self.con.execute("select 1 from sqlite_master where name = 'decisions'").fetchone()
        if not exists or version >= SCHEMA_VERSION:
            return
        self.con.executescript(
            "BEGIN; DROP INDEX IF EXISTS decisions_lecture; ALTER TABLE decisions RENAME TO decisions_old; "
            + DECISIONS + " INSERT INTO decisions SELECT * FROM decisions_old; DROP TABLE decisions_old; COMMIT;")

    def close(self) -> None:
        self.con.close()

    # ---------- courses / lectures ----------

    def upsert_course(self, name: str, language: str, folder: str | None = None, is_meetings: bool = False) -> str:
        with self.con:
            self.con.execute("insert into courses (id, name, folder, language, is_meetings) values (?, ?, ?, ?, ?) "
                             "on conflict (name) do nothing", (new_id(), name, folder, language, int(is_meetings)))
        return self.con.execute("select id from courses where name = ?", (name,)).fetchone()[0]

    def course(self, course_id: str) -> dict | None:
        row = self.con.execute("select * from courses where id = ?", (course_id,)).fetchone()
        return dict(row) if row else None

    def upsert_lecture(self, course_id: str, *, audio_path: str, source: str, title: str, date: str,
                       fact_check: bool, week: int | None = None) -> str:
        with self.con:
            return self.con.execute(
                "insert into lectures (id, course_id, week, date, title, source, fact_check, started_at, audio_path, "
                "status) values (?, ?, ?, ?, ?, ?, ?, ?, ?, 'recording') "
                "on conflict (course_id, source, audio_path) do update set week = excluded.week, "
                "date = excluded.date, title = excluded.title, fact_check = excluded.fact_check, "
                "started_at = excluded.started_at, ended_at = null, status = 'recording' returning id",
                (new_id(), course_id, week, date, title, source, int(fact_check), now_iso(),
                 audio_path)).fetchone()[0]

    def lecture(self, lecture_id: str) -> dict | None:
        row = self.con.execute("select * from lectures where id = ?", (lecture_id,)).fetchone()
        return dict(row) if row else None

    def end_lecture(self, lecture_id: str) -> None:
        with self.con:
            self.con.execute("update lectures set status = 'ended', ended_at = ? where id = ?", (now_iso(), lecture_id))

    # ---------- chunks ----------

    def _delete_chunks(self, lecture_id: str, where: str, arg: int) -> None:
        segs = f"select id from segments where lecture_id = ? and chunk_id {where} ?"
        for table in ("items", "claims"):
            self.con.execute(f"delete from {table} where segment_id in ({segs})", (lecture_id, arg))
        self.con.execute(f"delete from segments where lecture_id = ? and chunk_id {where} ?", (lecture_id, arg))

    def write_chunk(self, lecture_id: str, idx: int, segments: list[Segment], asr: str,
                    result: ExtractResult | None) -> None:
        """Replace everything chunk `idx` produced, in one transaction."""
        summary = result.chunk_summary if result else None
        seg_rows = [(new_id(), lecture_id, idx, s.t0, s.t1, s.text, s.speaker, asr, summary) for s in segments]
        with self.con:
            self._delete_chunks(lecture_id, "=", idx)
            self.con.executemany("insert into segments values (?, ?, ?, ?, ?, ?, ?, ?, ?)", seg_rows)
            if result is None or not seg_rows:
                return
            seg_id, t0 = seg_rows[0][0], seg_rows[0][3]
            items = [(new_id(), lecture_id, seg_id, "concept", c.term, c.explanation, c.canonical_key, None, None,
                      lecture_id, t0) for c in result.concepts]
            items += [(new_id(), lecture_id, seg_id, it.kind, it.text, None, None, it.owner, it.due, lecture_id, t0)
                      for it in result.items]
            self.con.executemany(
                "insert into items (id, lecture_id, segment_id, kind, text, explanation, canonical_key, owner, due, "
                "first_seen_lecture_id, t0) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", items)
            self.con.executemany(
                "insert into claims (id, lecture_id, segment_id, text, normalized, importance, status) "
                "values (?, ?, ?, ?, ?, ?, 'pending')",
                [(new_id(), lecture_id, seg_id, c.text, c.normalized, c.importance) for c in result.claims])

    def prune_chunks(self, lecture_id: str, last_idx: int) -> None:
        """After a complete run: drop chunks a previous run of the same lecture produced beyond the last one."""
        with self.con:
            self._delete_chunks(lecture_id, ">", last_idx)

    def previous_chunk_summary(self, lecture_id: str, idx: int) -> str | None:
        row = self.con.execute("select chunk_summary from segments where lecture_id = ? and chunk_id < ? "
                               "and chunk_summary is not null order by chunk_id desc limit 1",
                               (lecture_id, idx)).fetchone()
        return row[0] if row else None

    def counts(self, lecture_id: str) -> dict[str, int]:
        return {t: self.con.execute(f"select count(*) from {t} where lecture_id = ?", (lecture_id,)).fetchone()[0]
                for t in ("segments", "items", "claims")}

    # ---------- what the Digest reads and writes ----------

    def chunk_summaries(self, lecture_id: str) -> list[tuple[int, str]]:
        return [tuple(r) for r in self.con.execute(
            "select chunk_id, chunk_summary from segments where lecture_id = ? and chunk_summary is not null "
            "group by chunk_id order by chunk_id", (lecture_id,))]

    def items(self, lecture_id: str, kind: str | None = None) -> list[dict]:
        rows = self.con.execute(
            "select i.*, s.chunk_id from items i left join segments s on s.id = i.segment_id "
            "where i.lecture_id = ? and (? is null or i.kind = ?) order by s.chunk_id, i.id",
            (lecture_id, kind, kind))
        return [dict(r) for r in rows]

    def claims(self, lecture_id: str) -> list[dict]:
        rows = self.con.execute("select * from claims where lecture_id = ? order by importance desc, id",
                                (lecture_id,))
        return [dict(r) for r in rows]

    def segments(self, lecture_id: str) -> list[dict]:
        rows = self.con.execute("select * from segments where lecture_id = ? order by t0, id", (lecture_id,))
        return [dict(r) for r in rows]

    def save_digest(self, lecture_id: str, bullets: list[str], digest_md: str) -> None:
        with self.con:
            self.con.execute(
                "insert into lecture_summaries (lecture_id, bullets_json, digest_md) values (?, ?, ?) "
                "on conflict (lecture_id) do update set bullets_json = excluded.bullets_json, "
                "digest_md = excluded.digest_md", (lecture_id, json.dumps(bullets, ensure_ascii=False), digest_md))
            self.con.execute("update lectures set status = 'digested' where id = ?", (lecture_id,))

    # ---------- decisions ----------

    def log(self, node: str, *, lecture_id: str | None, input_ref: str, output: dict, ms: float | None = None,
            tokens_in: int | None = None, tokens_out: int | None = None, cost_usd: float | None = None) -> None:
        with self.con:
            self.con.execute("insert into decisions values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                             (new_id(), lecture_id, node, input_ref, json.dumps(output, ensure_ascii=False), ms,
                              tokens_in, tokens_out, cost_usd, now_iso()))
