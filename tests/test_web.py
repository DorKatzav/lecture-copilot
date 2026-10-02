import asyncio
import json

from fastapi.testclient import TestClient

from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.web.app import create_app
from lecture_copilot.web.session import Session
from tests.stubs import FakeASR, FakeGemini, FakeOllama, ListSource

REPLY = json.dumps({"chunk_summary": "הוסבר CAC",
                    "items": [{"kind": "question", "text": "למה?", "owner": None, "due": None}],
                    "concepts": [{"term": "CAC", "explanation": "עלות רכישת לקוח", "canonical_key": "cac"}],
                    "claims": [{"text": "CAC ירד", "normalized": "CAC fell", "importance": 90}]}, ensure_ascii=False)
SECTION = json.dumps({"paragraphs": ["פסקה."]}, ensure_ascii=False)
EXEC = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(5)], "continuation": None}, ensure_ascii=False)
RECAP = json.dumps({"bullets": ["הוסבר CAC"]}, ensure_ascii=False)
OK = {"verdict": "correct", "confidence": 0.9, "explanation": "נכון.", "sources": ["https://a"]}


def make(tmp_path, replies, n_chunks=2, gemini=None):
    fake = FakeOllama(replies)
    released = asyncio.Event()

    class SlowList(ListSource):
        async def __aiter__(self):
            async for c in super().__aiter__():
                yield c
            await released.wait()                       # the "mic" stays open until stop()

    def source_factory(lecture_id, kind, file=None, pace="fast"):
        chunks = [AudioChunk(lecture_id, i, tmp_path / f"chunk_{i:04d}.wav", (i - 1) * 45.0, i * 45.0)
                  for i in range(1, n_chunks + 1)]
        src = SlowList(chunks)
        src.stop = released.set
        return src
    session = Session(db=tmp_path / "copilot.sqlite", courses_root=tmp_path / "courses", runs_dir=tmp_path / "runs",
                      ollama_client_factory=fake.async_client, asr_factory=lambda: FakeASR(),
                      source_factory=source_factory, gemini_factory=(lambda: gemini) if gemini else None,
                      course_name="AI Developers — Python", backoff_s=0)
    app = create_app(session)
    return TestClient(app), session, fake


def wait_for(client, status, tries=200):
    for _ in range(tries):
        s = client.get("/api/state").json()
        if (s["current"] or {}).get("status") == status or (status == "idle" and s["current"] is None):
            return s
        import time
        time.sleep(0.02)
    raise AssertionError(f"never reached {status}: {client.get('/api/state').json()}")


def test_state_before_a_lecture(tmp_path):
    client, session, fake = make(tmp_path, [])
    with client:
        s = client.get("/api/state").json()
    assert s["current"] is None and s["courses"] == ["AI Developers — Python"] and s["interrupted"] == []
    assert s["ready"] is True


def test_record_then_stop_gives_a_digest_and_rows_arrive_meanwhile(tmp_path):
    client, session, fake = make(tmp_path, [REPLY, REPLY, SECTION, EXEC], gemini=FakeGemini([OK, OK]))
    with client:
        r = client.post("/api/record", json={"course": "AI Developers — Python", "title": "W01", "fact_check": True})
        assert r.status_code == 200 and r.json()["lecture_id"]
        s = wait_for(client, "recording")
        for _ in range(100):
            s = client.get("/api/state").json()
            if s["current"]["chunks"] == 2:
                break
            import time
            time.sleep(0.02)
        assert s["current"]["chunks"] == 2 and [c["term"] for c in s["dashboard"]["concepts"]] == ["CAC"]
        assert s["dashboard"]["claims"][0]["text"] == "CAC ירד" and s["dashboard"]["questions"] == ["למה?"]
        r = client.post("/api/stop")
        assert r.status_code == 200
        s = wait_for(client, "digested")
        run_row = json.loads(session.store.con.execute("select output_json from decisions where node = 'run' "
                                                       "order by ts desc limit 1").fetchone()[0])
    assert s["current"]["digest"]["sections"] == 9 and s["current"]["digest"]["folder"]
    assert run_row["via"] == "web" and run_row["source_kind"] == "mic"
    assert s["current"]["verifier"]["verified"] == 2 and s["current"]["cost_usd"] > 0   # the second from the cache
    assert (tmp_path / "courses" / "AI Developers — Python" / "course.html").is_file()


def test_mark_and_note_become_items_of_the_running_lecture(tmp_path):
    client, session, fake = make(tmp_path, [REPLY, REPLY, SECTION, EXEC])
    with client:
        client.post("/api/record", json={"course": "AI Developers — Python", "title": "W01", "fact_check": False})
        s = wait_for(client, "recording")
        for _ in range(100):
            if client.get("/api/state").json()["current"]["chunks"] == 2:
                break
            import time
            time.sleep(0.02)
        assert client.post("/api/mark").json()["ok"] is True
        assert client.post("/api/note", json={"text": "לשאול על LTV"}).json()["ok"] is True
        s = client.get("/api/state").json()
        assert s["dashboard"]["highlights"] == [{"text": "הוסבר CAC", "user": True}]
        assert s["dashboard"]["notes"][0]["text"] == "לשאול על LTV" and s["dashboard"]["notes"][0]["t"]
        client.post("/api/stop")
        wait_for(client, "digested")
        md = session.store.con.execute("select digest_md from lecture_summaries").fetchone()[0]
    assert "- לשאול על LTV" in md and "★ הוסבר CAC" in md


def test_recap_and_search(tmp_path):
    client, session, fake = make(tmp_path, [REPLY, REPLY, RECAP, SECTION, EXEC])
    with client:
        client.post("/api/record", json={"course": "AI Developers — Python", "title": "W01", "fact_check": False})
        wait_for(client, "recording")
        for _ in range(100):
            if client.get("/api/state").json()["current"]["chunks"] == 2:
                break
            import time
            time.sleep(0.02)
        r = client.get("/api/recap").json()
        assert r["bullets"] == ["הוסבר CAC"]
        hits = client.get("/api/search", params={"q": "CAC"}).json()
        assert any(h["kind"] == "concept" and h["text"] == "CAC" for h in hits)
        client.post("/api/stop")
        wait_for(client, "digested")


def test_an_interrupted_lecture_shows_a_banner_and_can_be_resumed(tmp_path):
    client, session, fake = make(tmp_path, [REPLY, REPLY, SECTION, EXEC])
    with client:
        client.post("/api/record", json={"course": "AI Developers — Python", "title": "W01", "fact_check": False})
        wait_for(client, "recording")
        for _ in range(100):
            if client.get("/api/state").json()["current"]["chunks"] == 2:
                break
            import time
            time.sleep(0.02)
    # the process died: a new session over the same database
    fake2 = FakeOllama([SECTION, EXEC])
    session2 = Session(db=tmp_path / "copilot.sqlite", courses_root=tmp_path / "courses", runs_dir=tmp_path / "runs",
                       ollama_client_factory=fake2.async_client, asr_factory=lambda: FakeASR(),
                       source_factory=lambda *a, **k: ListSource([]), gemini_factory=None,
                       course_name="AI Developers — Python", backoff_s=0)
    client2 = TestClient(create_app(session2))
    with client2:
        s = client2.get("/api/state").json()
        assert len(s["interrupted"]) == 1 and s["interrupted"][0]["chunks"] == 2 and s["current"] is None
        lid = s["interrupted"][0]["lecture_id"]
        assert client2.post("/api/resume", json={"lecture_id": lid}).status_code == 200
        s = wait_for(client2, "digested")
        assert s["current"]["digest"]["sections"] == 9 and s["interrupted"] == []
        row = session2.store.con.execute("select output_json from decisions where node = 'run' and input_ref = 'resume'"
                                         ).fetchone()
        assert json.loads(row[0]) == {"status": "resumed", "chunks": 2, "via": "web"}


def test_the_page_is_hebrew_rtl_and_never_pops_anything(tmp_path):
    client, session, fake = make(tmp_path, [])
    with client:
        html = client.get("/").text
    assert '<html lang="he" dir="rtl">' in html
    for forbidden in ("alert(", "Notification", "confirm(", "<audio", "play()"):
        assert forbidden not in html
