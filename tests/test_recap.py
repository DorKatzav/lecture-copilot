import asyncio
import json

from lecture_copilot.agents.recap import recap
from lecture_copilot.agents.schemas import Concept, ExtractResult, Item
from lecture_copilot.asr.base import Segment
from lecture_copilot.config import LIVE_MODEL
from lecture_copilot.store.db import Store
from tests.stubs import FakeOllama

THREE = json.dumps({"bullets": ["הוסבר CAC", "הוצג LTV", "נאמר שזה במבחן"]}, ensure_ascii=False)


def lecture(tmp_path, minutes=12):
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("c", language="he")
    lid = s.upsert_lecture(course, audio_path="/a", source="mic", title="t", date="d", fact_check=True)
    for i in range(1, minutes + 1):
        concepts = [Concept(term=f"מושג{i}", explanation="x", canonical_key=f"k{i}")] if i % 4 == 0 else []
        res = ExtractResult(chunk_summary=f"תקציר דקה {i}", claims=[], concepts=concepts,
                            items=[Item(kind="highlight", text="זה במבחן")] if i == minutes else [])
        s.write_chunk(lid, i, [Segment(t0=60.0 * (i - 1), t1=60.0 * i, text="x")], asr="mw", result=res)
    return s, lid


def test_recap_summarizes_only_the_last_minutes(tmp_path):
    s, lid = lecture(tmp_path)
    fake = FakeOllama([THREE])

    async def go():
        async with fake.async_client() as c:
            return await recap(lid, store=s, client=c, minutes=5, backoff_s=0)
    out = asyncio.run(go())
    assert out["bullets"] == ["הוסבר CAC", "הוצג LTV", "נאמר שזה במבחן"] and out["minutes"] == 5
    user = fake.requests[0]["messages"][1]["content"]
    assert "תקציר דקה 12" in user and "תקציר דקה 8" in user and "תקציר דקה 7" not in user
    assert "מושג12" in user and "מושג8" in user and "מושג4" not in user and "זה במבחן" in user
    assert fake.requests[0]["model"] == LIVE_MODEL
    (row,) = s.con.execute("select output_json from decisions where node = 'recap'").fetchall()
    assert json.loads(row[0])["status"] == "ok"
    s.close()


def test_recap_without_chunks_says_so_without_a_call(tmp_path):
    s, lid = lecture(tmp_path, minutes=0)
    fake = FakeOllama([])

    async def go():
        async with fake.async_client() as c:
            return await recap(lid, store=s, client=c, backoff_s=0)
    out = asyncio.run(go())
    assert out["bullets"] == [] and "אין" in out["note"] and fake.requests == []
    s.close()


def test_recap_failure_is_a_note_not_an_error(tmp_path):
    s, lid = lecture(tmp_path)
    fake = FakeOllama(["nope", "nope"])

    async def go():
        async with fake.async_client() as c:
            return await recap(lid, store=s, client=c, backoff_s=0)
    out = asyncio.run(go())
    assert out["bullets"] == [] and out["note"]
    s.close()
