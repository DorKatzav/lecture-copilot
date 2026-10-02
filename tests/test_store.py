import json
import sqlite3

import pytest

from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult, Item
from lecture_copilot.asr.base import Segment
from lecture_copilot.store.db import SCHEMA_VERSION, Store, new_id

CROCKFORD = set("0123456789ABCDEFGHJKMNPQRSTVWXYZ")


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "db" / "copilot.sqlite")
    yield s
    s.close()


def segs(*texts, t0=0.0):
    return [Segment(t0=t0 + 10 * i, t1=t0 + 10 * i + 9, text=t) for i, t in enumerate(texts)]


def result(n_concepts=1, n_claims=1, n_items=1, summary="סיכום"):
    return ExtractResult(
        chunk_summary=summary,
        concepts=[Concept(term=f"T{i}", explanation="הסבר", canonical_key=f"k{i}") for i in range(n_concepts)],
        claims=[Claim(text=f"c{i}", normalized=f"n{i}", importance=70) for i in range(n_claims)],
        items=[Item(kind="question", text=f"q{i}") for i in range(n_items)],
    )


def lecture(store, path="/x/fixture.m4a"):
    course = store.upsert_course("AI Developers — Python", language="he")
    return store.upsert_lecture(course, audio_path=path, source="file", title="Tirgul", date="2026-06-19",
                                fact_check=True)


# ---------- ids ----------

def test_new_id_is_a_ulid():
    i = new_id()
    assert len(i) == 26 and set(i) <= CROCKFORD


def test_new_ids_sort_by_time():
    assert new_id(now_ms=1_000) < new_id(now_ms=2_000) < new_id(now_ms=2**40)


def test_ids_minted_in_the_same_millisecond_keep_their_order():
    ids = [new_id() for _ in range(2000)]
    assert ids == sorted(ids)


def test_new_ids_are_unique():
    assert len({new_id() for _ in range(1000)}) == 1000


# ---------- schema ----------

def test_schema_has_the_spec_tables(store):
    names = {r[0] for r in store.con.execute("select name from sqlite_master where type='table'")}
    assert {"courses", "lectures", "segments", "items", "claims", "lecture_summaries", "fact_cache",
            "decisions"} <= names
    assert store.con.execute("pragma user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopening_keeps_rows(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    lid = lecture(s)
    s.close()
    s2 = Store(tmp_path / "c.sqlite")
    assert s2.con.execute("select id from lectures").fetchone()[0] == lid
    s2.close()


# ---------- courses / lectures ----------

def test_course_upsert_by_name_keeps_id_and_language(store):
    a = store.upsert_course("Entrepreneurship", language="he")
    b = store.upsert_course("Entrepreneurship", language="he")
    assert a == b and store.course(a)["language"] == "he"


def test_course_language_is_not_changed_by_a_later_call(store):
    a = store.upsert_course("Entrepreneurship", language="he")
    store.upsert_course("Entrepreneurship", language="en")
    assert store.course(a)["language"] == "he"  # language is per course, set once


def test_replaying_the_same_file_is_the_same_lecture(store):
    a = lecture(store)
    store.end_lecture(a)
    b = lecture(store)
    assert a == b
    row = store.lecture(a)
    assert row["status"] == "recording" and row["ended_at"] is None
    assert store.con.execute("select count(*) from lectures").fetchone()[0] == 1


def test_another_file_is_another_lecture(store):
    assert lecture(store, "/x/a.m4a") != lecture(store, "/x/b.m4a")


def test_end_lecture_sets_status_and_time(store):
    lid = lecture(store)
    store.end_lecture(lid)
    row = store.lecture(lid)
    assert row["status"] == "ended" and row["ended_at"]


# ---------- chunks ----------

def test_write_chunk_stores_segments_items_claims(store):
    lid = lecture(store)
    store.write_chunk(lid, 1, segs("a", "b", t0=30), asr="mw:ivrit", result=result(2, 1, 1))
    assert store.counts(lid) == {"segments": 2, "items": 3, "claims": 1}
    rows = store.con.execute("select chunk_id, t0, asr, chunk_summary from segments order by t0").fetchall()
    assert [tuple(r) for r in rows] == [(1, 30.0, "mw:ivrit", "סיכום"), (1, 40.0, "mw:ivrit", "סיכום")]


def test_extracted_rows_link_to_the_chunks_first_segment(store):
    lid = lecture(store)
    store.write_chunk(lid, 1, segs("a", "b", t0=30), asr="mw", result=result(1, 1, 1))
    first = store.con.execute("select id from segments where t0 = 30").fetchone()[0]
    concept = store.con.execute("select * from items where kind='concept'").fetchone()
    assert concept["segment_id"] == first and concept["t0"] == 30.0
    assert (concept["text"], concept["explanation"], concept["canonical_key"]) == ("T0", "הסבר", "k0")
    assert concept["first_seen_lecture_id"] == lid
    claim = store.con.execute("select * from claims").fetchone()
    assert claim["segment_id"] == first and claim["status"] == "pending" and claim["importance"] == 70


def test_rewriting_a_chunk_replaces_it(store):
    lid = lecture(store)
    store.write_chunk(lid, 1, segs("a", "b"), asr="mw", result=result(3, 2, 2))
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=result(1, 0, 0))
    assert store.counts(lid) == {"segments": 1, "items": 1, "claims": 0}


def test_rewriting_a_chunk_leaves_other_chunks_and_lectures(store):
    lid, other = lecture(store, "/x/a.m4a"), lecture(store, "/x/b.m4a")
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=result())
    store.write_chunk(lid, 2, segs("b", t0=60), asr="mw", result=result())
    store.write_chunk(other, 1, segs("c"), asr="mw", result=result())
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=result())
    assert store.counts(lid) == {"segments": 2, "items": 4, "claims": 2}
    assert store.counts(other) == {"segments": 1, "items": 2, "claims": 1}


def test_chunk_without_extraction_keeps_its_transcript(store):
    lid = lecture(store)
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=None)
    assert store.counts(lid) == {"segments": 1, "items": 0, "claims": 0}


def test_ids_are_minted_ulids(store):
    lid = lecture(store)
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=result())
    for table in ("segments", "items", "claims"):
        (i,) = store.con.execute(f"select id from {table}").fetchone()
        assert len(i) == 26 and set(i) <= CROCKFORD


def test_prune_drops_chunks_after_the_last_one(store):
    lid = lecture(store)
    for idx in (1, 2, 3):
        store.write_chunk(lid, idx, segs("a", t0=60 * idx), asr="mw", result=result())
    store.prune_chunks(lid, last_idx=2)
    assert store.counts(lid) == {"segments": 2, "items": 4, "claims": 2}


def test_previous_chunk_summary(store):
    lid = lecture(store)
    assert store.previous_chunk_summary(lid, 1) is None
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=result(summary="ראשון"))
    store.write_chunk(lid, 2, segs("b", t0=60), asr="mw", result=None)
    assert store.previous_chunk_summary(lid, 3) == "ראשון"  # skips a chunk whose extraction failed


# ---------- decisions ----------

def test_log_appends_a_decision_row(store):
    lid = lecture(store)
    store.log("extractor", lecture_id=lid, input_ref="run1#0003", output={"status": "ok", "claims": 1},
              ms=812.5, tokens_in=900, tokens_out=300)
    store.log("extractor", lecture_id=lid, input_ref="run1#0003", output={"status": "ok"}, ms=1.0)
    rows = store.con.execute("select * from decisions order by ts").fetchall()
    assert len(rows) == 2
    assert rows[0]["node"] == "extractor" and rows[0]["ms"] == 812.5 and rows[0]["tokens_in"] == 900
    assert json.loads(rows[0]["output_json"]) == {"status": "ok", "claims": 1}


def test_log_rejects_an_unknown_node(store):
    with pytest.raises(sqlite3.IntegrityError):
        store.log("oracle", lecture_id=None, input_ref="x", output={})


# ---------- M2: what the Digest reads and writes ----------

def test_week_is_stored_and_updated_on_replay(store):
    course = store.upsert_course("c", language="he")
    kw = dict(audio_path="/x/a.m4a", source="file", title="t", date="2026-11-04", fact_check=True)
    lid = store.upsert_lecture(course, week=5, **kw)
    assert store.lecture(lid)["week"] == 5
    store.upsert_lecture(course, week=6, **kw)
    assert store.lecture(lid)["week"] == 6


def test_chunk_summaries_in_order_without_failed_chunks(store):
    lid = lecture(store)
    store.write_chunk(lid, 2, segs("b", t0=60), asr="mw", result=result(summary="שני"))
    store.write_chunk(lid, 1, segs("a", "a2"), asr="mw", result=result(summary="ראשון"))
    store.write_chunk(lid, 3, segs("c", t0=120), asr="mw", result=None)
    assert store.chunk_summaries(lid) == [(1, "ראשון"), (2, "שני")]


def test_items_carry_their_chunk(store):
    lid = lecture(store)
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=result(1, 0, 1))
    store.write_chunk(lid, 2, segs("b", t0=60), asr="mw", result=result(1, 0, 0))
    rows = store.items(lid)
    assert [(r["chunk_id"], r["kind"], r["text"]) for r in rows] == [
        (1, "concept", "T0"), (1, "question", "q0"), (2, "concept", "T0")]
    assert [r["text"] for r in store.items(lid, kind="question")] == ["q0"]


def test_claims_come_most_important_first(store):
    lid = lecture(store)
    res = ExtractResult(chunk_summary="s", concepts=[], items=[], claims=[
        Claim(text="minor", normalized="n", importance=40), Claim(text="major", normalized="n", importance=90)])
    store.write_chunk(lid, 1, segs("a"), asr="mw", result=res)
    assert [(c["text"], c["importance"], c["status"]) for c in store.claims(lid)] == [
        ("major", 90, "pending"), ("minor", 40, "pending")]


def test_segments_in_time_order(store):
    lid = lecture(store)
    store.write_chunk(lid, 2, segs("c", t0=60), asr="mw", result=None)
    store.write_chunk(lid, 1, segs("a", "b"), asr="mw", result=None)
    assert [s["text"] for s in store.segments(lid)] == ["a", "b", "c"]


def test_saving_a_digest_marks_the_lecture_digested_and_replaces_the_old_one(store):
    lid = lecture(store)
    store.save_digest(lid, bullets=["א", "ב"], digest_md="# first")
    store.save_digest(lid, bullets=["ג"], digest_md="# second")
    rows = store.con.execute("select * from lecture_summaries").fetchall()
    assert len(rows) == 1 and rows[0]["digest_md"] == "# second" and json.loads(rows[0]["bullets_json"]) == ["ג"]
    assert store.lecture(lid)["status"] == "digested"


def test_digest_and_sink_are_loggable_nodes(store):
    store.log("digest", lecture_id=None, input_ref="x", output={})
    store.log("sink", lecture_id=None, input_ref="x", output={})


def test_a_database_from_m1_is_migrated_and_keeps_its_rows(tmp_path):
    path = tmp_path / "old.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE decisions (id TEXT PRIMARY KEY, lecture_id TEXT,
            node TEXT NOT NULL CHECK (node IN ('extractor', 'memory', 'verifier', 'ranker', 'net', 'asr', 'chunk',
                                               'run')),
            input_ref TEXT, output_json TEXT, ms REAL, tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL,
            ts TEXT NOT NULL);
        INSERT INTO decisions VALUES ('01A', 'L', 'run', 'R1', '{"chunks": 14}', 1.5, 1, 2, null, '2026-09-28');
        PRAGMA user_version = 1;""")
    con.close()
    s = Store(path)
    s.log("digest", lecture_id="L", input_ref="R2", output={"ok": True})
    rows = s.con.execute("select id, node, output_json, ms from decisions order by ts").fetchall()
    assert [tuple(r) for r in rows][0] == ("01A", "run", '{"chunks": 14}', 1.5) and rows[1]["node"] == "digest"
    assert s.con.execute("pragma user_version").fetchone()[0] == SCHEMA_VERSION == 7
    s.close()


def test_a_v6_database_gets_the_llm_json_column(tmp_path):
    path = tmp_path / "v6.sqlite"
    con = sqlite3.connect(path)
    con.executescript("""
        CREATE TABLE lecture_summaries (lecture_id TEXT PRIMARY KEY, bullets_json TEXT, digest_md TEXT,
                                        embedding BLOB);
        INSERT INTO lecture_summaries VALUES ('L', '["a"]', '# x', null);
        CREATE TABLE decisions (id TEXT PRIMARY KEY, lecture_id TEXT,
            node TEXT NOT NULL CHECK (node IN ('extractor', 'memory', 'verifier', 'ranker', 'net', 'asr', 'chunk',
                                               'run', 'digest', 'sink', 'recap')),
            input_ref TEXT, output_json TEXT, ms REAL, tokens_in INTEGER, tokens_out INTEGER, cost_usd REAL,
            ts TEXT NOT NULL);
        PRAGMA user_version = 6;""")
    con.close()
    s = Store(path)
    row = s.con.execute("select bullets_json, llm_json from lecture_summaries").fetchone()
    assert tuple(row) == ('["a"]', None)
    s.save_digest("L", ["b"], "# y", llm={"full_summary": ["p"], "continuation": None, "degraded": []})
    assert json.loads(s.con.execute("select llm_json from lecture_summaries").fetchone()[0])["full_summary"] == ["p"]
    s.close()
