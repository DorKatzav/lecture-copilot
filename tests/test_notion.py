import asyncio
import json

import pytest

from lecture_copilot.output.digest import SECTIONS, ClaimRow, ConceptRow, DigestDoc, TaskRow, section_headings
from lecture_copilot.output.notion import (
    NOTION_HOST,
    NotionAPI,
    NotionIds,
    NotionSink,
    notion_init,
    page_id_from,
    render_notion_markdown,
    save_env_keys,
)
from lecture_copilot.store.db import Store
from lecture_copilot.store.net import Net, NetError, Offline
from tests.stubs import FakeNotion


def doc(**kw):
    base = dict(
        lecture_id="L1", course_id="K1", course_name="יזמות וחדשנות", title="מודלים עסקיים ב'", date="2026-11-04",
        week=5, minutes=88, language="he", exec_summary=[f"נקודה {i}" for i in range(1, 6)],
        full_summary=["פסקה ראשונה.", "פסקה שנייה על CAC."], highlights=["שלושת סוגי ה-Freemium"],
        concepts=[ConceptRow("CAC", "עלות רכישת לקוח", "cac"), ConceptRow("Freemium", "חינם עם תשלום", "freemium"),
                  ConceptRow("LTV", "ערך חיי לקוח", "ltv", first_seen="W04")],
        claims=[ClaimRow("OpenAI הוקמה למטרות רווח", 90, "verified", "לא נכון", "incorrect", "מלכ\"ר ב-2015",
                         ["https://example.org/openai"], id="C1"),
                ClaimRow("CAC ירד ב-2024", 80, "pending", "עדיין לא נבדק", id="C2")],
        questions=["למה?"], tasks=[TaskRow("ניתוח מקרה", "דור", "2026-11-11", id="T1"),
                                  TaskRow("לקרוא פרק 3", None, "בשבוע הבא", id="T2")],
        notes=["הערה שלי"], segments=[])
    base.update(kw)
    return DigestDoc(**base)


@pytest.fixture
def world(tmp_path):
    fake = FakeNotion()
    store = Store(tmp_path / "copilot.sqlite")
    net = Net(store, backoff_s=0.0)
    api = NotionAPI("secret-token", client=fake.async_client())
    ids = asyncio.run(notion_init(api, net, fake.root_page))
    yield fake, store, net, api, ids
    store.close()


# ---------- ids and .env ----------

@pytest.mark.parametrize("text", [
    "https://www.notion.so/dor/Studies-2f3a1b4c5d6e7f8091a2b3c4d5e6f708",
    "https://www.notion.so/2f3a1b4c5d6e7f8091a2b3c4d5e6f708?v=abc",
    "2f3a1b4c-5d6e-7f80-91a2-b3c4d5e6f708",
    "2f3a1b4c5d6e7f8091a2b3c4d5e6f708",
])
def test_page_id_from_a_url_or_uuid(text):
    assert page_id_from(text) == "2f3a1b4c-5d6e-7f80-91a2-b3c4d5e6f708"


def test_page_id_from_garbage_is_none():
    assert page_id_from("not a page") is None


def test_save_env_keys_updates_in_place_and_appends(tmp_path):
    env = tmp_path / ".env"
    env.write_text('GEMINI_API_KEY="k"\nNOTION_TOKEN=\n# comment\n', encoding="utf-8")
    save_env_keys(env, {"NOTION_TOKEN": "t", "NOTION_DS_COURSES": "ds1"})
    assert env.read_text(encoding="utf-8") == ('GEMINI_API_KEY="k"\nNOTION_TOKEN=t\n# comment\n'
                                               'NOTION_DS_COURSES=ds1\n')


def test_ids_round_trip_through_env():
    ids = NotionIds("c", "l", "g", "k", "t")
    assert NotionIds.from_env(ids.as_env()) == ids
    assert NotionIds.from_env({"NOTION_DS_COURSES": "c"}) is None


# ---------- notion-init ----------

def test_init_creates_the_five_databases_under_the_root_page(world):
    fake, store, net, api, ids = world
    assert set(fake.databases) == {ids.courses, ids.lectures, ids.glossary, ids.claims, ids.tasks}
    assert all(d["parent"] == {"type": "page_id", "page_id": fake.root_page} for d in fake.databases.values())
    assert [fake.databases[i]["title"][0] for i in (ids.courses, ids.lectures, ids.glossary, ids.claims, ids.tasks)
            ] == ["🎓", "📚", "📖", "🔍", "📌"]


def test_every_database_relates_to_courses_and_carries_a_hidden_key(world):
    fake, store, net, api, ids = world
    for ds in (ids.lectures, ids.glossary, ids.claims, ids.tasks):
        props = fake.databases[ds]["properties"]
        assert props["key"] == {"rich_text": {}}
        assert any(p.get("relation", {}).get("data_source_id") == ids.courses for p in props.values())


def test_init_goes_through_net_and_is_logged(world):
    fake, store, net, api, ids = world
    rows = [json.loads(r[0]) for r in store.con.execute("select output_json from decisions where node = 'net'")]
    assert len(rows) == 5 and all(r["host"] == NOTION_HOST and r["status"] == "ok" for r in rows)
    assert all(r["cost_usd"] == 0 for r in rows)


def test_init_twice_reuses_existing_databases(world):
    fake, store, net, api, ids = world
    again = asyncio.run(notion_init(api, net, fake.root_page, existing=ids))
    assert again == ids and len(fake.databases) == 5


def test_init_recreates_a_database_that_was_deleted(world):
    fake, store, net, api, ids = world
    del fake.databases[ids.tasks]
    again = asyncio.run(notion_init(api, net, fake.root_page, existing=ids))
    assert again.tasks != ids.tasks and again.courses == ids.courses and len(fake.databases) == 5


# ---------- the lecture page ----------

def test_notion_markdown_has_the_nine_sections():
    md = render_notion_markdown(doc())
    assert section_headings(md) == SECTIONS


def test_notion_markdown_uses_a_callout_for_the_exec_summary_and_a_toggle_for_the_full_one():
    md = render_notion_markdown(doc())
    assert "<callout" in md and "נקודה 1" in md.split("</callout>")[0]
    assert "<details>" in md and "פסקה ראשונה." in md.split("</details>")[0]


def test_notion_markdown_tasks_are_checkboxes():
    assert "- [ ] ניתוח מקרה" in render_notion_markdown(doc())


def test_write_lecture_creates_the_page_and_the_rows(world):
    fake, store, net, api, ids = world
    url = asyncio.run(NotionSink(api, net, ids).write_lecture(doc()))
    lectures = fake.rows(ids.lectures)
    assert len(lectures) == 1 and url == f"https://www.notion.so/{next(iter(fake.pages))}" or url.startswith("https://")
    page = lectures[0]
    assert page["properties"]["key"]["rich_text"][0]["plain_text"] == "L1"
    assert section_headings(page["markdown"]) == SECTIONS
    assert len(fake.rows(ids.courses)) == 1
    assert len(fake.rows(ids.glossary)) == 3 and len(fake.rows(ids.claims)) == 2 and len(fake.rows(ids.tasks)) == 2


def test_rows_relate_to_the_course_and_the_lecture(world):
    fake, store, net, api, ids = world
    asyncio.run(NotionSink(api, net, ids).write_lecture(doc()))
    course_id = next(pid for pid, p in fake.pages.items() if p["parent"] == ids.courses)
    lecture_id = next(pid for pid, p in fake.pages.items() if p["parent"] == ids.lectures)
    for ds in (ids.glossary, ids.claims, ids.tasks):
        for row in fake.rows(ds):
            rels = [r["id"] for p in row["properties"].values() if "relation" in p for r in p["relation"]]
            assert course_id in rels and lecture_id in rels


def test_an_iso_due_date_becomes_a_date_and_free_text_stays_in_the_title(world):
    fake, store, net, api, ids = world
    asyncio.run(NotionSink(api, net, ids).write_lecture(doc()))
    by_key = {r["properties"]["key"]["rich_text"][0]["plain_text"]: r["properties"] for r in fake.rows(ids.tasks)}
    assert by_key["L1:T1"]["תאריך יעד"] == {"date": {"start": "2026-11-11"}}
    assert by_key["L1:T2"]["תאריך יעד"] == {"date": None}
    assert "בשבוע הבא" in by_key["L1:T2"]["משימה"]["title"][0]["plain_text"]


def test_resync_twice_updates_in_place(world):
    fake, store, net, api, ids = world
    sink = NotionSink(api, net, ids)
    asyncio.run(sink.write_lecture(doc()))
    before = {ds: [pid for pid, p in fake.pages.items() if p["parent"] == ds] for ds in ids.as_env().values()}
    asyncio.run(sink.write_lecture(doc(exec_summary=["חדש"] * 5)))
    after = {ds: [pid for pid, p in fake.pages.items() if p["parent"] == ds] for ds in ids.as_env().values()}
    assert after == before
    page = fake.rows(ids.lectures)[0]
    assert "חדש" in page["markdown"] and "נקודה 1" not in page["markdown"]


def test_a_returning_concept_keeps_its_first_lecture(world):
    fake, store, net, api, ids = world
    sink = NotionSink(api, net, ids)
    asyncio.run(sink.write_lecture(doc()))
    asyncio.run(sink.write_lecture(doc(lecture_id="L2", title="הרצאה הבאה", week=6,
                                       concepts=[ConceptRow("CAC", "הסבר מעודכן", "cac", first_seen="W05")])))
    assert len(fake.rows(ids.glossary)) == 3
    cac = next(r for r in fake.rows(ids.glossary) if r["properties"]["key"]["rich_text"][0]["plain_text"] == "K1:cac")
    first = {p["properties"]["key"]["rich_text"][0]["plain_text"]: pid for pid, p in fake.pages.items()
             if p["parent"] == ids.lectures}
    assert cac["properties"]["נראה לראשונה"]["relation"] == [{"id": first["L1"]}]
    assert cac["properties"]["הסבר"]["rich_text"][0]["plain_text"] == "הסבר מעודכן"


def test_the_task_done_box_is_not_reset_on_resync(world):
    fake, store, net, api, ids = world
    sink = NotionSink(api, net, ids)
    asyncio.run(sink.write_lecture(doc()))
    row = next(pid for pid, p in fake.pages.items() if p["parent"] == ids.tasks)
    fake.pages[row]["properties"]["בוצע"] = {"checkbox": True}
    asyncio.run(sink.write_lecture(doc()))
    assert fake.pages[row]["properties"]["בוצע"] == {"checkbox": True}


# ---------- failures ----------

def test_one_failure_is_retried(world):
    fake, store, net, api, ids = world
    fake.fail_next = 1
    asyncio.run(NotionSink(api, net, ids).write_lecture(doc()))
    assert len(fake.rows(ids.lectures)) == 1


def test_two_failures_raise_and_are_logged(world):
    fake, store, net, api, ids = world
    fake.fail_next = 2
    with pytest.raises(NetError):
        asyncio.run(NotionSink(api, net, ids).write_lecture(doc()))
    row = store.con.execute("select output_json from decisions where node = 'net' and lecture_id = 'L1' "
                            "order by ts desc limit 1").fetchone()
    assert json.loads(row[0])["status"] == "failed"


def test_offline_raises_offline(tmp_path):
    fake = FakeNotion(offline=True)
    store = Store(tmp_path / "copilot.sqlite")
    api = NotionAPI("secret-token", client=fake.async_client())
    with pytest.raises(Offline):
        asyncio.run(NotionSink(api, Net(store, 0.0), NotionIds("c", "l", "g", "k", "t")).write_lecture(doc()))
    store.close()


def test_the_token_never_appears_in_a_log_row(world):
    fake, store, net, api, ids = world
    asyncio.run(NotionSink(api, net, ids).write_lecture(doc()))
    blob = " ".join(r[0] for r in store.con.execute("select output_json from decisions"))
    assert "secret-token" not in blob
