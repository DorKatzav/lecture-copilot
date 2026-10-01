import asyncio
import json

import pytest

from lecture_copilot.agents.memory import MemoryContext, recall, remember
from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.config import ALREADY_SAID_COSINE, EMBED_MODEL, Profile
from lecture_copilot.pipeline import Ctx
from lecture_copilot.store.db import Store
from tests.stubs import EMBED_DIMS, FakeASR, FakeOllama


@pytest.fixture(params=["sqlite-vec", "numpy"])
def store(tmp_path, request):
    s = Store(tmp_path / "c.sqlite", vec_backend=request.param, dims=EMBED_DIMS)
    yield s
    s.close()


def concept(term, explanation, key):
    return Concept(term=term, explanation=explanation, canonical_key=key)


def ctx_for(store, lid, course, client):
    return Ctx(lecture_id=lid, course_id=course, course_name="יזמות", lecture_title="t", profile=Profile(),
               store=store, asr=FakeASR(), ollama=client, run_id="R", backoff_s=0)


def run(coro_fn, fake):
    async def go():
        async with fake.async_client() as client:
            return await coro_fn(client)
    return asyncio.run(go())


def seed_previous_lecture(store, fake):
    """W4 taught CAC and Churn and claimed something about Dropbox; its rows carry embeddings."""
    course = store.upsert_course("יזמות", language="he")
    w4 = store.upsert_lecture(course, audio_path="/w4", source="file", title="מודלים א'", date="2026-10-28",
                              fact_check=True, week=4)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[
        concept("CAC", "עלות רכישת לקוח: הוצאות שיווק חלקי לקוחות חדשים", "cac"),
        concept("Churn", "שיעור הלקוחות שעוזבים בחודש", "churn")],
        claims=[Claim(text="Dropbox הגיעה ל-4% משלמים", normalized="Dropbox: 4% paying users", importance=80)])
    chunk = AudioChunk(w4, 1, store.path.parent / "c.wav", 0.0, 40.0)
    store.write_chunk(w4, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    run(lambda c: remember(res, chunk, ctx_for(store, w4, course, c), "R#0001"), fake)
    w5 = store.upsert_lecture(course, audio_path="/w5", source="file", title="מודלים ב'", date="2026-11-04",
                              fact_check=True, week=5)
    return course, w4, w5


# ---------- recall: what the course already knows about this chunk ----------

def test_recall_returns_hits_from_earlier_lectures_only(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    own = ExtractResult(chunk_summary="s", items=[], claims=[], concepts=[concept("CAC", "עלות רכישת לקוח", "cac")])
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=own)   # this lecture's own row
    run(lambda c: remember(own, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c), "R#0001"), fake)
    segs = [Segment(t0=0, t1=30, text="נחזור על עלות רכישת לקוח, CAC, ומה זה אומר על השיווק")]
    mem = run(lambda c: recall(segs, ctx_for(store, w5, course, c), "R#0001"), fake)
    assert isinstance(mem, MemoryContext) and mem.hits
    assert mem.hits[0].text == "CAC" and all(h.lecture_id == w4 for h in mem.hits)
    assert fake.embed_calls[-1] == ["נחזור על עלות רכישת לקוח, CAC, ומה זה אומר על השיווק"]


def test_recall_renders_known_terms_and_previous_claims_for_the_prompt(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    segs = [Segment(t0=0, t1=30, text="Dropbox ו-CAC משלמים")]
    mem = run(lambda c: recall(segs, ctx_for(store, w5, course, c), "R#0001"), fake)
    assert "CAC — עלות רכישת לקוח" in mem.known_terms and "(W04)" in mem.known_terms
    assert "Dropbox הגיעה ל-4% משלמים" in mem.previous_claims


def test_recall_with_an_empty_course_memory_is_empty(store):
    fake = FakeOllama()
    course = store.upsert_course("c", language="he")
    lid = store.upsert_lecture(course, audio_path="/a", source="file", title="t", date="d", fact_check=True)
    mem = run(lambda c: recall([Segment(t0=0, t1=1, text="x")], ctx_for(store, lid, course, c), "R#0001"), fake)
    assert mem.hits == [] and mem.known_terms == "(none yet)" and mem.previous_claims == "(none)"


def test_recall_survives_an_embedding_failure(store):
    fake = FakeOllama(fail_loads=True)
    course, w4, w5 = seed_previous_lecture(store, FakeOllama())
    mem = run(lambda c: recall([Segment(t0=0, t1=1, text="CAC")], ctx_for(store, w5, course, c), "R#0001"), fake)
    assert mem.hits and mem.hits[0].text == "CAC"     # text search still works
    row = json.loads(store.con.execute("select output_json from decisions where node = 'memory' "
                                       "order by ts desc limit 1").fetchone()[0])
    assert row["step"] == "recall" and row["embed_error"]


def test_recall_is_logged_with_its_timing(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    run(lambda c: recall([Segment(t0=0, t1=1, text="CAC")], ctx_for(store, w5, course, c), "R#0007"), fake)
    row = store.con.execute("select input_ref, ms, output_json from decisions where node = 'memory' "
                            "order by ts desc limit 1").fetchone()
    assert row["input_ref"] == "R#0007" and row["ms"] is not None and json.loads(row["output_json"])["hits"] >= 1


# ---------- remember: embeddings, already said, contradictions ----------

def test_remember_embeds_every_concept_and_claim_in_one_batch(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    assert len(fake.embed_calls) == 1 and len(fake.embed_calls[0]) == 3
    assert fake.loads[0][1]["model"] == EMBED_MODEL
    assert store.con.execute("select count(*) from items where embedding is null").fetchone()[0] == 0
    assert store.con.execute("select count(*) from claims where embedding is null").fetchone()[0] == 0


def test_a_concept_with_a_known_key_was_already_said(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    res = ExtractResult(chunk_summary="s", items=[], claims=[],
                        concepts=[concept("עלות רכישת לקוח", "כמה עולה להביא לקוח", "CAC"),   # key differs in case only
                                  concept("LTV", "ערך חיי לקוח", "ltv")])
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    out = run(lambda c: remember(res, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c),
                                 "R#0001"), fake)
    rows = {r["text"]: r["first_seen_lecture_id"] for r in store.items(w5, kind="concept")}
    assert rows["עלות רכישת לקוח"] == w4 and rows["LTV"] == w5
    assert out["already_said"] == 1 and out["new_concepts"] == 1


def test_a_concept_with_the_same_term_was_already_said(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    res = ExtractResult(chunk_summary="s", items=[], claims=[],
                        concepts=[concept("churn", "נטישה", "customer_churn_rate")])      # other key, same term
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    run(lambda c: remember(res, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c), "R#0001"), fake)
    assert store.items(w5, kind="concept")[0]["first_seen_lecture_id"] == w4


def test_a_concept_whose_meaning_is_close_enough_was_already_said(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    res = ExtractResult(chunk_summary="s", items=[], claims=[],
                        concepts=[concept("עלות רכישה", "עלות רכישת לקוח: הוצאות שיווק חלקי לקוחות חדשים", "acq")])
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    run(lambda c: remember(res, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c), "R#0001"), fake)
    assert store.items(w5, kind="concept")[0]["first_seen_lecture_id"] == w4 and 0.5 < ALREADY_SAID_COSINE < 1


def test_a_concept_repeated_inside_the_same_lecture_is_not_already_said(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    for idx in (1, 2):
        res = ExtractResult(chunk_summary="s", items=[], claims=[], concepts=[concept("LTV", "ערך חיי לקוח", "ltv")])
        store.write_chunk(w5, idx, [Segment(t0=40 * idx, t1=40 * idx + 40, text="x")], asr="mw", result=res)
        run(lambda c, r=res, i=idx: remember(r, AudioChunk(w5, i, store.path, 0, 40), ctx_for(store, w5, course, c),
                                             f"R#{i:04d}"), fake)
    assert all(r["first_seen_lecture_id"] == w5 for r in store.items(w5, kind="concept"))


def test_a_claim_the_model_marked_as_contradicting_gains_20_importance(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[], claims=[
        Claim(text="Dropbox הגיעה ל-10% משלמים", normalized="Dropbox: 10% paying", importance=70,
              contradicts="Dropbox הגיעה ל-4% משלמים"),
        Claim(text="LTV צריך להיות פי 3", normalized="LTV/CAC = 3", importance=95, contradicts=None)])
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    mem = run(lambda c: recall([Segment(t0=0, t1=1, text="Dropbox משלמים")], ctx_for(store, w5, course, c),
                               "R#0001"), fake)
    out = run(lambda c: remember(res, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c),
                                 "R#0001", memory=mem), fake)
    rows = {r["text"]: r for r in store.claims(w5)}
    assert rows["Dropbox הגיעה ל-10% משלמים"]["importance"] == 90
    assert rows["Dropbox הגיעה ל-10% משלמים"]["contradicts_id"] == store.claims(w4)[0]["id"]
    assert rows["LTV צריך להיות פי 3"]["importance"] == 95 and rows["LTV צריך להיות פי 3"]["contradicts_id"] is None
    assert out["contradictions"] == 1


def test_a_contradiction_that_matches_no_recalled_claim_is_kept_but_not_linked(store):
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[], claims=[
        Claim(text="x", normalized="x", importance=50, contradicts="something the model made up")])
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    out = run(lambda c: remember(res, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c),
                                 "R#0001", memory=MemoryContext()), fake)
    row = store.claims(w5)[0]
    assert row["importance"] == 70 and row["contradicts_id"] is None and out["contradictions"] == 1


def test_remember_survives_an_embedding_failure(store):
    course, w4, w5 = seed_previous_lecture(store, FakeOllama())
    res = ExtractResult(chunk_summary="s", items=[], claims=[], concepts=[concept("CAC", "x", "cac")])
    store.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    out = run(lambda c: remember(res, AudioChunk(w5, 1, store.path, 0, 40), ctx_for(store, w5, course, c),
                                 "R#0001"), FakeOllama(fail_loads=True))
    assert out["embed_error"] and store.items(w5, kind="concept")[0]["first_seen_lecture_id"] == w4  # key still works


def test_memory_only_looks_at_lectures_that_started_earlier(store):
    """Replaying an old lecture after a newer one: the newer one is not 'already said'."""
    fake = FakeOllama()
    course, w4, w5 = seed_previous_lecture(store, fake)
    w3 = store.upsert_lecture(course, audio_path="/w3", source="file", title="W3", date="2026-10-21",
                              fact_check=True, week=3)
    res = ExtractResult(chunk_summary="s", items=[], claims=[], concepts=[concept("CAC", "עלות רכישת לקוח", "cac")])
    store.write_chunk(w3, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    mem = run(lambda c: recall([Segment(t0=0, t1=1, text="CAC")], ctx_for(store, w3, course, c), "R#0001"), fake)
    out = run(lambda c: remember(res, AudioChunk(w3, 1, store.path, 0, 40), ctx_for(store, w3, course, c),
                                 "R#0001", memory=mem), fake)
    assert mem.hits == [] and out["already_said"] == 0
    assert store.items(w3, kind="concept")[0]["first_seen_lecture_id"] == w3
