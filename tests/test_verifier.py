import asyncio
import json

import httpx
import pytest

from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult
from lecture_copilot.agents.verifier import Verifier, VerifierWorker, cache_key
from lecture_copilot.asr.base import Segment
from lecture_copilot.config import GEMINI_HOST, VERIFY_MIN_IMPORTANCE
from lecture_copilot.store.db import Store
from lecture_copilot.store.net import Net
from tests.stubs import FakeGemini

CORRECT = {"verdict": "correct", "confidence": 0.95, "explanation": "נכון, לפי התיעוד.", "sources": ["https://a"]}
WRONG = {"verdict": "incorrect", "confidence": 0.9, "explanation": "הפונקציה מחזירה None.", "sources": ["https://b"],
         "grounding": ["https://docs.python.org/3/library/random.html"]}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    yield s
    s.close()


def lecture(store, claims):
    course = store.upsert_course("AI Developers — Python", language="he")
    lid = store.upsert_lecture(course, audio_path="/x.m4a", source="file", title="t", date="2026-06-09",
                               fact_check=True)
    res = ExtractResult(chunk_summary="s", items=[],
                        concepts=[Concept(term="shuffle", explanation="x", canonical_key="s")],
                        claims=[Claim(text=t, normalized=n, importance=i) for t, n, i in claims])
    store.write_chunk(lid, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    return course, lid


def verifier(store, fake, **kw):
    return Verifier(store, Net(store, backoff_s=0), fake, **kw)


def test_a_claim_is_verified_stored_cached_and_logged(store):
    course, lid = lecture(store, [("shuffle מחזירה רשימה", "random.shuffle returns the list", 80)])
    fake = FakeGemini([WRONG])
    claim = store.claims(lid)[0]
    v = asyncio.run(verifier(store, fake).verify(claim["id"]))
    assert v.verdict == "incorrect"
    row = store.claims(lid)[0]
    assert row["status"] == "verified" and row["verdict"] == "incorrect" and row["confidence"] == 0.9
    assert row["explanation"] == "הפונקציה מחזירה None."
    assert json.loads(row["sources_json"]) == ["https://b", "https://docs.python.org/3/library/random.html"]
    assert row["cache_key"] == cache_key("random.shuffle returns the list")
    cached = store.con.execute("select verdict, sources_json from fact_cache").fetchone()
    assert cached["verdict"] == "incorrect"
    net = store.con.execute("select * from decisions where node = 'net'").fetchone()
    assert json.loads(net["output_json"])["host"] == GEMINI_HOST and net["cost_usd"] > 0 and net["lecture_id"] == lid
    system, user = fake.calls[0]
    assert "shuffle מחזירה רשימה" in user and "AI Developers — Python" in user


def test_the_course_context_goes_into_the_prompt(store):
    course, lid = lecture(store, [("shuffle מחזירה רשימה", "random.shuffle returns the list", 80)])
    earlier = store.upsert_lecture(course, audio_path="/e.m4a", source="file", title="W1", date="2026-06-07",
                                   fact_check=True, week=1)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[], claims=[
        Claim(text="shuffle לא מחזירה כלום", normalized="random.shuffle returns None", importance=90)])
    store.write_chunk(earlier, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    fake = FakeGemini([WRONG])
    asyncio.run(verifier(store, fake).verify(store.claims(lid)[0]["id"]))
    assert "shuffle לא מחזירה כלום" in fake.calls[0][1] and "(W01)" in fake.calls[0][1]


def test_a_repeated_claim_is_answered_from_the_cache_without_a_call(store):
    course, lid = lecture(store, [("א", "random.shuffle returns the list", 80),
                                  ("ב", "Random.shuffle  returns the list ", 75)])
    fake = FakeGemini([WRONG])
    ver = verifier(store, fake)
    for c in store.claims(lid):
        asyncio.run(ver.verify(c["id"]))
    assert len(fake.calls) == 1
    rows = store.claims(lid)
    assert all(r["status"] == "verified" and r["verdict"] == "incorrect" for r in rows)
    assert store.con.execute("select count(*) from decisions where node = 'net'").fetchone()[0] == 1
    ver_rows = [json.loads(o) for (o,) in store.con.execute(
        "select output_json from decisions where node = 'verifier'")]
    assert [r["cache"] for r in ver_rows] == [False, True]


def test_offline_marks_the_claim_unchecked_and_keeps_going(store):
    course, lid = lecture(store, [("א", "n1", 80)])
    fake = FakeGemini([httpx.ConnectError("no route")])
    v = asyncio.run(verifier(store, fake).verify(store.claims(lid)[0]["id"]))
    assert v is None and store.claims(lid)[0]["status"] == "unchecked"


def test_a_failing_provider_marks_the_claim_unchecked_after_one_retry(store):
    course, lid = lecture(store, [("א", "n1", 80)])
    fake = FakeGemini([RuntimeError("429"), RuntimeError("429")])
    asyncio.run(verifier(store, fake).verify(store.claims(lid)[0]["id"]))
    assert store.claims(lid)[0]["status"] == "unchecked" and len(fake.calls) == 2


def test_an_invalid_answer_is_retried_then_unchecked(store):
    course, lid = lecture(store, [("א", "n1", 80)])
    fake = FakeGemini([{"verdict": "maybe"}, {"verdict": "maybe"}])
    asyncio.run(verifier(store, fake).verify(store.claims(lid)[0]["id"]))
    assert store.claims(lid)[0]["status"] == "unchecked" and len(fake.calls) == 2


def test_cache_key_normalizes_case_and_spaces():
    assert cache_key("Random.shuffle  returns the list ") == cache_key("random.shuffle returns the list")
    assert cache_key("a") != cache_key("b") and len(cache_key("a")) == 64


# ---------- the worker ----------

def run_worker(store, lid, fake, importances, finish=True):
    course, lid = lecture(store, [(f"c{i}", f"n{i}", imp) for i, imp in enumerate(importances)])
    ver = verifier(store, fake)

    async def go():
        worker = VerifierWorker(ver, lid)
        task = asyncio.create_task(worker.run())
        for c in store.claims(lid):
            if c["importance"] >= VERIFY_MIN_IMPORTANCE:
                worker.enqueue(c["id"])
        out = await worker.finish()
        await task
        return out
    return lid, asyncio.run(go())


def test_worker_verifies_material_claims_two_at_a_time(store):
    fake = FakeGemini([CORRECT] * 6)
    lid, out = run_worker(store, None, fake, [90, 85, 80, 75, 70, 60])
    rows = {r["text"]: r["status"] for r in store.claims(lid)}
    assert [rows[f"c{i}"] for i in range(6)] == ["verified"] * 5 + ["skipped"]
    assert fake.max_in_flight == 2 and out == {"verified": 5, "unchecked": 0, "skipped": 1, "retried": 0}


def test_finish_retries_what_went_unchecked_when_the_network_is_back(store):
    fake = FakeGemini([httpx.ConnectError("down"), CORRECT, CORRECT])
    lid, out = run_worker(store, None, fake, [90, 80])
    assert out["verified"] == 2 and out["retried"] == 1 and all(r["status"] == "verified" for r in store.claims(lid))


def test_finish_leaves_unchecked_when_still_offline(store):
    fake = FakeGemini([httpx.ConnectError("down")] * 4)
    lid, out = run_worker(store, None, fake, [90])
    assert out["unchecked"] == 1 and store.claims(lid)[0]["status"] == "unchecked"


# ---------- the real backend's response parsing ----------

def test_gemini_response_is_parsed_into_text_usage_and_grounding_urls():
    from types import SimpleNamespace as NS

    from lecture_copilot.agents.verifier import parse_gemini_response
    resp = NS(text='{"verdict": "correct"}',
              usage_metadata=NS(prompt_token_count=170, candidates_token_count=150),
              candidates=[NS(grounding_metadata=NS(grounding_chunks=[
                  NS(web=NS(uri="https://docs.python.org/3/library/random.html")), NS(web=None)]))])
    text, usage = parse_gemini_response(resp, bytes_out=900)
    assert text == '{"verdict": "correct"}' and usage["tokens_in"] == 170 and usage["tokens_out"] == 150
    assert usage["grounding"] == ["https://docs.python.org/3/library/random.html"] and usage["bytes_out"] == 900


def test_gemini_response_without_metadata_still_parses():
    from types import SimpleNamespace as NS

    from lecture_copilot.agents.verifier import parse_gemini_response
    text, usage = parse_gemini_response(NS(text="{}", usage_metadata=None, candidates=[]), bytes_out=1)
    assert usage["tokens_in"] == 0 and usage["grounding"] == []
