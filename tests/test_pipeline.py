import asyncio
import json

import pytest

from lecture_copilot.asr.base import ASRError, Segment
from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.config import (
    EMBED_MODEL,
    EXTRACT_OPTIONS,
    LIVE_MODEL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_LOAD_OPTIONS,
    Profile,
)
from lecture_copilot.pipeline import Ctx, run, warm_up
from lecture_copilot.store.db import Store
from tests.stubs import FakeASR, FakeOllama, ListSource


def reply(summary="סיכום", concepts=1, claims=1, items=0, term="CAC"):
    return json.dumps({
        "chunk_summary": summary,
        "concepts": [{"term": term, "explanation": "עלות רכישת לקוח", "canonical_key": "cac"}] * concepts,
        "claims": [{"text": "CAC ירד ב-2024", "normalized": "CAC fell in 2024", "importance": 60}] * claims,
        "items": [{"kind": "question", "text": "למה?", "owner": None, "due": None}] * items,
    }, ensure_ascii=False)


def chunks(tmp_path, n, length=45.0):
    return [AudioChunk("L", i, tmp_path / f"chunk_{i:04d}.wav", (i - 1) * length, i * length) for i in range(1, n + 1)]


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "copilot.sqlite")
    yield s
    s.close()


def go(store, tmp_path, source, asr, fake, lecture_id=None, **kw):
    async def main():
        async with fake.async_client() as client:
            ctx = make_ctx(store, asr, client, lecture_id, tmp_path)
            return ctx, await run(source, ctx, **kw)
    return asyncio.run(main())


def make_ctx(store, asr, client, lecture_id, tmp_path):
    course = store.upsert_course("AI Developers — Python", language="he")
    lid = lecture_id or store.upsert_lecture(course, audio_path=str(tmp_path / "lecture.m4a"), source="file",
                                             title="Tirgul", date="2026-06-19", fact_check=True)
    return Ctx(lecture_id=lid, course_id=course, course_name="AI Developers — Python", lecture_title="Tirgul",
               profile=Profile(language="he"), store=store, asr=asr, ollama=client, run_id="RUN1",
               runs_dir=tmp_path / "runs", backoff_s=0)


def decisions(store, node):
    return [dict(r, output=json.loads(r["output_json"]))
            for r in store.con.execute("select * from decisions where node = ? order by ts", (node,))]


# ---------- the happy path ----------

def test_every_chunk_lands_in_the_store(store, tmp_path):
    ctx, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 3)), FakeASR(),
                      FakeOllama([reply(), reply(items=1), reply(concepts=2)]))
    assert store.counts(ctx.lecture_id) == {"segments": 3, "items": 5, "claims": 3}
    assert summary["chunks"] == 3 and summary["status"] == {"ok": 3}
    assert summary["counts"] == store.counts(ctx.lecture_id)


def test_segment_times_are_lecture_times(store, tmp_path):
    asr = FakeASR(default=[Segment(t0=1.0, t1=4.0, text="א"), Segment(t0=4.0, t1=9.5, text="ב")])
    go(store, tmp_path, ListSource(chunks(tmp_path, 2)), asr, FakeOllama([reply(), reply()]))
    rows = store.con.execute("select chunk_id, t0, t1 from segments order by t0").fetchall()
    assert [tuple(r) for r in rows] == [(1, 1.0, 4.0), (1, 4.0, 9.5), (2, 46.0, 49.0), (2, 49.0, 54.5)]


def test_asr_gets_the_course_language(store, tmp_path):
    asr = FakeASR()
    go(store, tmp_path, ListSource(chunks(tmp_path, 1)), asr, FakeOllama([reply()]))
    assert asr.calls[0][1] == "he"


def test_extraction_request_uses_the_live_model_and_prompt(store, tmp_path):
    fake = FakeOllama([reply(summary="פרק ראשון"), reply()])
    go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(), fake)
    first, second = fake.requests
    assert first["model"] == LIVE_MODEL and first["keep_alive"] == OLLAMA_KEEP_ALIVE
    assert {k: first["options"][k] for k in EXTRACT_OPTIONS} == EXTRACT_OPTIONS
    assert {k: first["options"][k] for k in OLLAMA_LOAD_OPTIONS} == OLLAMA_LOAD_OPTIONS
    user = second["messages"][1]["content"]
    assert "AI Developers — Python" in user and "Chunk 2 (45–90 s)" in user
    assert "פרק ראשון" in user            # the previous chunk's summary
    assert "שלום, היום נדבר על CAC" in user


def test_every_step_is_logged(store, tmp_path):
    go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(), FakeOllama([reply(), reply()]))
    asr, ext, chunk = decisions(store, "asr"), decisions(store, "extractor"), decisions(store, "chunk")
    assert [d["input_ref"] for d in chunk] == ["RUN1#0001", "RUN1#0002"]
    assert all(d["output"]["status"] == "ok" and d["ms"] is not None for d in asr + ext + chunk)
    assert ext[0]["tokens_in"] == 100 and ext[0]["output"]["attempts"] == 1
    assert {"asr_s", "extract_s", "queue_depth", "segments"} <= chunk[0]["output"].keys()
    (run_row,) = decisions(store, "run")
    assert run_row["input_ref"] == "RUN1" and run_row["output"]["counts"]["claims"] == 2


# ---------- failures never stop the lecture ----------

def test_asr_error_marks_the_chunk_failed_and_the_run_continues(store, tmp_path):
    asr = FakeASR(script={2: ASRError("mw exit 1: could not decode audio")})
    ctx, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 3)), asr, FakeOllama([reply(), reply()]))
    chunk = decisions(store, "chunk")
    assert [d["output"]["status"] for d in chunk] == ["ok", "asr_failed", "ok"]
    assert "could not decode" in chunk[1]["output"]["error"]
    assert summary["status"] == {"ok": 2, "asr_failed": 1}
    assert [r[0] for r in store.con.execute("select distinct chunk_id from segments order by 1")] == [1, 3]


def test_unexpected_exception_in_a_chunk_is_logged_not_raised(store, tmp_path):
    asr = FakeASR(script={1: KeyError("boom")})
    _, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 2)), asr, FakeOllama([reply()]))
    assert summary["status"] == {"failed": 1, "ok": 1}
    assert "KeyError" in decisions(store, "chunk")[0]["output"]["error"]


def test_failed_extraction_keeps_the_transcript(store, tmp_path):
    fake = FakeOllama(["not json", "{still not", reply()])
    ctx, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(), fake)
    assert summary["status"] == {"extract_failed": 1, "ok": 1}
    assert store.con.execute("select count(*) from segments where chunk_id = 1").fetchone()[0] == 1
    assert decisions(store, "extractor")[0]["output"]["status"] == "failed"


def test_foreign_script_is_retried_and_never_stored(store, tmp_path):
    fake = FakeOllama([reply(term="бизнес"), reply(term="CAC")])
    ctx, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 1)), FakeASR(), fake)
    assert "Cyrillic" in fake.requests[1]["messages"][-1]["content"]
    assert [r[0] for r in store.con.execute("select text from items")] == ["CAC"]
    assert decisions(store, "extractor")[0]["output"]["attempts"] == 2


def test_foreign_script_twice_fails_the_extraction(store, tmp_path):
    fake = FakeOllama([reply(summary="סיכום هو"), reply(summary="סיכום هو")])
    _, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 1)), FakeASR(), fake)
    assert summary["status"] == {"extract_failed": 1}
    assert store.con.execute("select count(*) from items").fetchone()[0] == 0


def test_silent_chunk_skips_extraction(store, tmp_path):
    fake = FakeOllama([reply()])
    _, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(script={1: []}), fake)
    assert summary["status"] == {"empty": 1, "ok": 1} and len(fake.requests) == 1


def test_a_broken_source_keeps_what_was_done_and_raises(store, tmp_path):
    with pytest.raises(RuntimeError, match="ffmpeg"):
        go(store, tmp_path, ListSource(chunks(tmp_path, 3), fail_after=2), FakeASR(), FakeOllama([reply(), reply()]))
    (run_row,) = decisions(store, "run")
    assert "ffmpeg" in run_row["output"]["source_error"] and run_row["output"]["chunks"] == 2


# ---------- replay is an upsert ----------

def test_replaying_gives_identical_counts(store, tmp_path):
    ctx1, s1 = go(store, tmp_path, ListSource(chunks(tmp_path, 3)), FakeASR(), FakeOllama([reply()] * 3))
    ctx2, s2 = go(store, tmp_path, ListSource(chunks(tmp_path, 3)), FakeASR(), FakeOllama([reply()] * 3))
    assert ctx1.lecture_id == ctx2.lecture_id and s1["counts"] == s2["counts"] == store.counts(ctx1.lecture_id)


def test_a_shorter_replay_drops_the_old_tail(store, tmp_path):
    ctx, _ = go(store, tmp_path, ListSource(chunks(tmp_path, 3)), FakeASR(), FakeOllama([reply()] * 3))
    go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(), FakeOllama([reply()] * 2))
    assert store.counts(ctx.lecture_id) == {"segments": 2, "items": 2, "claims": 2}


def test_summary_has_budget_percentiles(store, tmp_path):
    _, summary = go(store, tmp_path, ListSource(chunks(tmp_path, 3)), FakeASR(), FakeOllama([reply()] * 3))
    for stage in ("asr_s", "extract_s", "total_s"):
        assert {"p50", "p95", "max"} <= summary["timing"][stage].keys()


def test_extra_run_fields_are_logged(store, tmp_path):
    go(store, tmp_path, ListSource(chunks(tmp_path, 1)), FakeASR(), FakeOllama([reply()]),
       extra=lambda: {"memory": {"peak_used_gb": 14.2}})
    assert decisions(store, "run")[0]["output"]["memory"] == {"peak_used_gb": 14.2}


# ---------- warm-up ----------

def test_warm_up_loads_asr_llm_and_embedding_model(store, tmp_path):
    fake, asr = FakeOllama(), FakeASR()

    async def main():
        async with fake.async_client() as client:
            await warm_up(make_ctx(store, asr, client, None, tmp_path))
    asyncio.run(main())
    assert len(asr.calls) == 1 and asr.calls[0][0].endswith("warmup.wav")
    loaded = {body["model"]: body["keep_alive"] for _, body in fake.loads}
    assert loaded == {LIVE_MODEL: OLLAMA_KEEP_ALIVE, EMBED_MODEL: OLLAMA_KEEP_ALIVE}
    # the same load options as every later call, so Ollama never reloads the model mid-lecture
    assert all(body["options"] == OLLAMA_LOAD_OPTIONS for _, body in fake.loads)
    assert {d["input_ref"] for d in decisions(store, "asr") + decisions(store, "extractor")} == {"RUN1#warmup"}


def test_warm_up_failures_are_logged_not_raised(store, tmp_path):
    fake, asr = FakeOllama(fail_loads=True), FakeASR(default=ASRError("mw not found"))

    async def main():
        async with fake.async_client() as client:
            await warm_up(make_ctx(store, asr, client, None, tmp_path))
    asyncio.run(main())
    assert decisions(store, "asr")[0]["output"]["status"] == "failed"
    assert decisions(store, "extractor")[0]["output"]["status"] == "failed"


def test_a_chunk_that_fails_on_replay_leaves_no_stale_rows(store, tmp_path):
    ctx, _ = go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(), FakeOllama([reply()] * 2))
    go(store, tmp_path, ListSource(chunks(tmp_path, 2)), FakeASR(script={2: ASRError("mw timed out")}),
       FakeOllama([reply()]))
    assert [r[0] for r in store.con.execute("select distinct chunk_id from segments")] == [1]
