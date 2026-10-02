import asyncio
import json

import pytest

from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult, Item
from lecture_copilot.asr.base import Segment
from lecture_copilot.config import DIGEST_MODEL
from lecture_copilot.output.digest import (
    SECTIONS,
    digest,
    estimate_tokens,
    plan_blocks,
    render_markdown,
    section_headings,
    target_words,
)
from lecture_copilot.store.db import Store
from tests.stubs import FakeOllama

SECTION = json.dumps({"paragraphs": ["פסקה ראשונה על git.", "פסקה שנייה על GitHub."]}, ensure_ascii=False)
EXEC = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(1, 6)], "continuation": None}, ensure_ascii=False)


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    yield s
    s.close()


def lecture(store, chunks=3, week=5, **extra):
    course = store.upsert_course("יזמות וחדשנות", language="he")
    lid = store.upsert_lecture(course, audio_path="/x/a.m4a", source="file", title="מודלים עסקיים ב'",
                               date="2026-11-04", fact_check=True, week=week)
    for i in range(1, chunks + 1):
        res = ExtractResult(
            chunk_summary=f"תקציר של קטע {i} בהרצאה על מודלים עסקיים",
            concepts=[Concept(term="CAC", explanation="עלות רכישת לקוח", canonical_key="cac"),
                      Concept(term=f"מושג {i}", explanation=f"הסבר {i}", canonical_key=f"k{i}")],
            claims=[Claim(text=f"טענה {i}", normalized="n", importance=min(100, 50 + 10 * i))],
            items=extra.get("items", [Item(kind="question", text=f"שאלה {i}")]) if i == 1 else [])
        store.write_chunk(lid, i, [Segment(t0=(i - 1) * 45.0, t1=i * 45.0, text=f"טקסט {i}")], asr="mw", result=res)
    store.end_lecture(lid)
    return lid


def run(store, lid, fake):
    async def go():
        async with fake.async_client() as client:
            return await digest(lid, store=store, client=client, backoff_s=0)
    return asyncio.run(go())


# ---------- planning ----------

def test_token_estimate_is_three_per_word():
    assert estimate_tokens("אחת שתיים שלוש") == 9 and estimate_tokens("") == 0


def test_target_words_follow_the_material_inside_the_spec_range():
    assert target_words(100) == 120        # never less than a short paragraph
    assert target_words(300) == 270        # about 0.9 of what came in
    assert target_words(5000) == 550       # the spec's 400–600 for a full lecture


def test_a_short_lecture_is_one_block():
    lines = [(i, "מילה " * 12) for i in range(1, 15)]
    (block,) = plan_blocks(lines, {}, budget_tokens=2600)
    assert [i for i, _ in block.lines] == list(range(1, 15))


def test_a_long_lecture_is_split_into_even_blocks_that_fit():
    lines = [(i, "מילה " * 12) for i in range(1, 121)]             # 120 chunks ≈ 90 minutes
    blocks = plan_blocks(lines, {}, budget_tokens=2600)
    assert len(blocks) >= 2 and [i for b in blocks for i, _ in b.lines] == list(range(1, 121))
    assert max(len(b.lines) for b in blocks) - min(len(b.lines) for b in blocks) <= 1
    assert all(b.tokens <= 2600 for b in blocks)


def test_concepts_fill_what_is_left_of_the_budget():
    lines = [(1, "מילה " * 12), (2, "מילה " * 12)]
    concepts = {1: [("CAC", "הסבר " * 10)] * 3, 2: [("LTV", "הסבר " * 10)] * 3}
    (block,) = plan_blocks(lines, concepts, budget_tokens=150)
    assert block.tokens <= 150 and 0 < len(block.concepts) < 6


# ---------- the document ----------

def test_nine_sections_in_the_fixed_order(store):
    doc = run(store, lecture(store), FakeOllama([SECTION, EXEC]))
    assert section_headings(render_markdown(doc)) == SECTIONS and len(SECTIONS) == 9


def test_heading_has_course_week_title_and_date(store):
    md = render_markdown(run(store, lecture(store), FakeOllama([SECTION, EXEC])))
    assert md.splitlines()[0] == "# יזמות וחדשנות — W05 · מודלים עסקיים ב' · 4.11.2026"


def test_heading_without_a_week(store):
    md = render_markdown(run(store, lecture(store, week=None), FakeOllama([SECTION, EXEC])))
    assert md.splitlines()[0] == "# יזמות וחדשנות — מודלים עסקיים ב' · 4.11.2026"


def test_llm_text_lands_in_its_sections(store):
    doc = run(store, lecture(store), FakeOllama([SECTION, EXEC]))
    assert doc.exec_summary == [f"נקודה {i}" for i in range(1, 6)]
    assert doc.full_summary == ["פסקה ראשונה על git.", "פסקה שנייה על GitHub."] and doc.degraded == []
    md = render_markdown(doc)
    assert "- נקודה 1\n- נקודה 2" in md and "פסקה ראשונה על git.\n\nפסקה שנייה על GitHub." in md


def test_concepts_are_listed_once_per_key_with_a_count(store):
    doc = run(store, lecture(store), FakeOllama([SECTION, EXEC]))
    assert [c.term for c in doc.concepts] == ["CAC", "מושג 1", "מושג 2", "מושג 3"]
    assert "## מושגים (4)" in render_markdown(doc) and "- **CAC** — עלות רכישת לקוח" in render_markdown(doc)


def test_only_material_claims_are_flagged_most_important_first(store):
    doc = run(store, lecture(store), FakeOllama([SECTION, EXEC]))
    assert [(c.text, c.importance) for c in doc.claims] == [("טענה 3", 80), ("טענה 2", 70)]   # 60 is minor
    assert '- המרצה אמר: "טענה 3" · [עדיין לא נבדק]' in render_markdown(doc)
    assert len(doc.all_claims) == 3                                   # claims.json keeps every claim


def test_items_go_to_their_sections(store):
    items = [Item(kind="question", text="למה?"), Item(kind="highlight", text="זה במבחן"),
             Item(kind="action", text="לקרוא פרק 3", due="2026-11-11", owner="דור"),
             Item(kind="note", text="לבדוק בבית")]
    md = render_markdown(run(store, lecture(store, items=items), FakeOllama([SECTION, EXEC])))
    assert "## ★ למבחן / הודגש\n\n- זה במבחן" in md and "## שאלות פתוחות\n\n- למה?" in md
    assert "## משימות\n\n- לקרוא פרק 3 (עד 2026-11-11 · דור)" in md and "## ההערות שלי\n\n- לבדוק בבית" in md


def test_empty_sections_say_so_and_stay(store):
    md = render_markdown(run(store, lecture(store, items=[]), FakeOllama([SECTION, EXEC])))
    assert "## ★ למבחן / הודגש\n\nאין.\n" in md and "## ההערות שלי\n\nאין.\n" in md
    assert "## המשך מההרצאה הקודמת\n\nזו ההרצאה הראשונה בקורס.\n" in md
    assert section_headings(md) == SECTIONS


# ---------- the calls ----------

def test_map_then_reduce_with_the_digest_model(store):
    fake = FakeOllama([SECTION, EXEC])
    run(store, lecture(store), fake)
    first, second = fake.requests
    assert first["model"] == second["model"] == DIGEST_MODEL
    assert "תקציר של קטע 2" in first["messages"][1]["content"] and "CAC" in first["messages"][1]["content"]
    assert "פסקה ראשונה על git." in second["messages"][1]["content"]   # the reduce step reads the full summary


def test_a_long_lecture_makes_one_map_call_per_block(store):
    lid = lecture(store, chunks=150)
    fake = FakeOllama([SECTION] * 3 + [EXEC])                          # 150 lines ≈ 4,500 tokens → 3 blocks
    doc = run(store, lid, fake)
    assert len(fake.requests) == 4 and len(doc.full_summary) == 6 and doc.degraded == []
    assert "Part 1 of 3" in fake.requests[0]["messages"][1]["content"]
    assert "Part 3 of 3" in fake.requests[2]["messages"][1]["content"]


def test_every_call_and_the_whole_digest_are_logged(store):
    lid = lecture(store)
    run(store, lid, FakeOllama([SECTION, EXEC]))
    rows = [(r["input_ref"].split("#")[1], json.loads(r["output_json"])) for r in store.con.execute(
        "select input_ref, output_json from decisions where node = 'digest' order by ts")]
    assert [k for k, _ in rows] == ["map-1", "exec", "digest"]
    assert rows[2][1]["blocks"] == 1 and rows[2][1]["degraded"] == [] and "total_s" in rows[2][1]


def test_the_digest_is_saved_and_the_lecture_is_digested(store):
    lid = lecture(store)
    doc = run(store, lid, FakeOllama([SECTION, EXEC]))
    row = store.con.execute("select * from lecture_summaries").fetchone()
    assert row["digest_md"] == render_markdown(doc) and json.loads(row["bullets_json"]) == doc.exec_summary
    assert store.lecture(lid)["status"] == "digested"


# ---------- failures never cost the student the Digest ----------

def test_a_failed_block_falls_back_to_its_chunk_summaries(store):
    doc = run(store, lecture(store), FakeOllama(["not json", "{still", EXEC]))
    assert doc.degraded == ["map-1"] and "תקציר של קטע 1" in doc.full_summary[0]
    assert section_headings(render_markdown(doc)) == SECTIONS


def test_a_failed_executive_summary_is_said_in_the_digest(store):
    doc = run(store, lecture(store), FakeOllama([SECTION, 500, 500]))
    assert doc.degraded == ["exec"] and doc.exec_summary == []
    assert "## סיכום מנהלים\n\nלא נוצר" in render_markdown(doc)


def test_four_bullets_are_retried(store):
    four = json.dumps({"exec_summary": ["א", "ב", "ג", "ד"], "continuation": None}, ensure_ascii=False)
    fake = FakeOllama([SECTION, four, EXEC])
    doc = run(store, lecture(store), fake)
    assert len(doc.exec_summary) == 5 and len(fake.requests) == 3


def test_foreign_script_in_the_summary_is_retried(store):
    bad = json.dumps({"paragraphs": ["פסקה על бизнес"]}, ensure_ascii=False)
    fake = FakeOllama([bad, SECTION, EXEC])
    doc = run(store, lecture(store), fake)
    assert "Cyrillic" in fake.requests[1]["messages"][-1]["content"] and "бизнес" not in render_markdown(doc)


def test_a_lecture_without_any_summary_still_gets_a_digest(store):
    course = store.upsert_course("c", language="he")
    lid = store.upsert_lecture(course, audio_path="/x/e.m4a", source="file", title="t", date="2026-11-04",
                               fact_check=True)
    fake = FakeOllama([])
    doc = run(store, lid, fake)
    assert fake.requests == [] and doc.degraded == ["empty"]
    assert section_headings(render_markdown(doc)) == SECTIONS


@pytest.mark.parametrize("items", [[], [Item(kind="highlight", text="זה במבחן"), Item(kind="action", text="לקרוא")]])
def test_markdown_is_well_formed(store, items):
    md = render_markdown(run(store, lecture(store, items=items), FakeOllama([SECTION, EXEC])))
    lines = md.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("## "):
            assert lines[i - 1] == "" and lines[i + 1] == "", f"no blank line around {line!r}"
    assert "\n\n\n" not in md and md.endswith("\n") and not md.endswith("\n\n")


# ---------- M3: the previous lecture ----------

CONT = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(1, 6)],
                   "continuation": {"new": ["LTV"], "repeated": ["CAC"], "contradicts": []}}, ensure_ascii=False)


def with_previous(store):
    """W4 was digested; W5 is the lecture being digested now, and CAC was first seen in W4."""
    w5 = lecture(store)
    course = store.lecture(w5)["course_id"]
    w4 = store.upsert_lecture(course, audio_path="/x/w4.m4a", source="file", title="מודלים עסקיים א'",
                              date="2026-10-28", fact_check=True, week=4)
    store.end_lecture(w4)
    store.save_digest(w4, bullets=["CAC הוסבר", "LTV נדחה לשבוע הבא"], digest_md="# w4")
    cac = next(r for r in store.items(w5, kind="concept") if r["text"] == "CAC")
    store.set_first_seen(cac["id"], w4)
    return w4, w5


def test_previous_lecture_is_the_latest_digested_one_before_this(store):
    w4, w5 = with_previous(store)
    course = store.lecture(w5)["course_id"]
    assert store.previous_lecture(course, w5) == w4
    assert store.previous_lecture(course, w4) is None


def test_the_reduce_step_sees_the_previous_bullets_and_the_chapter_is_filled(store):
    w4, w5 = with_previous(store)
    fake = FakeOllama([SECTION, CONT])
    doc = run(store, w5, fake)
    user = fake.requests[1]["messages"][1]["content"]
    assert "W04 · מודלים עסקיים א'" in user and "LTV נדחה לשבוע הבא" in user
    assert doc.prev_title == "W04 · מודלים עסקיים א'" and doc.continuation.new == ["LTV"]
    md = render_markdown(doc)
    assert "## המשך מ-W04 · מודלים עסקיים א'" in md and "- **מה חדש:** LTV" in md
    assert store.lecture(w5)["continues_id"] == w4


def test_returned_concepts_are_counted_and_marked(store):
    w4, w5 = with_previous(store)
    md = render_markdown(run(store, w5, FakeOllama([SECTION, CONT])))
    assert "## מושגים (4, +1 שחזר מ-W04)" in md and "- **CAC** — עלות רכישת לקוח (נאמר ב-W04)" in md


def test_a_lecture_without_a_previous_one_keeps_the_placeholder(store):
    md = render_markdown(run(store, lecture(store), FakeOllama([SECTION, EXEC])))
    assert "זו ההרצאה הראשונה בקורס" in md and "## מושגים (4)\n" in md


def test_a_null_continuation_twice_keeps_the_bullets_and_says_so(store):
    w4, w5 = with_previous(store)
    doc = run(store, w5, FakeOllama([SECTION, EXEC, EXEC]))
    md = render_markdown(doc)
    assert doc.exec_summary == [f"נקודה {i}" for i in range(1, 6)] and doc.degraded == ["continuation"]
    assert "## המשך מ-W04 · מודלים עסקיים א'" in md and "המודל לא השווה" in md


def test_with_a_previous_lecture_a_null_continuation_is_retried(store):
    w4, w5 = with_previous(store)
    fake = FakeOllama([SECTION, EXEC, CONT])            # EXEC has continuation: null
    doc = run(store, w5, fake)
    assert doc.continuation is not None and doc.continuation.new == ["LTV"] and len(fake.requests) == 3
    assert "continuation" in fake.requests[2]["messages"][-1]["content"]
    assert fake.requests[1]["format"]["required"] == ["exec_summary", "continuation"]


def test_without_a_previous_lecture_null_is_fine(store):
    fake = FakeOllama([SECTION, EXEC])
    doc = run(store, lecture(store), fake)
    assert doc.continuation is None and len(fake.requests) == 2


def test_an_english_continuation_is_retried_and_none_placeholders_are_dropped(store):
    w4, w5 = with_previous(store)
    english = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(1, 6)],
                          "continuation": {"new": ["This lecture adds LTV"], "repeated": [], "contradicts": ["None"]}},
                         ensure_ascii=False)
    hebrew = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(1, 6)],
                         "continuation": {"new": ["LTV נוסף"], "repeated": ["none"], "contradicts": ["-"]}},
                        ensure_ascii=False)
    fake = FakeOllama([SECTION, english, hebrew])
    doc = run(store, w5, fake)
    assert "Hebrew" in fake.requests[2]["messages"][-1]["content"]
    c = doc.continuation
    assert c.new == ["LTV נוסף"] and c.repeated == [] and c.contradicts == []


def test_a_bare_term_in_the_continuation_passes_the_hebrew_check(store):
    w4, w5 = with_previous(store)
    doc = run(store, w5, FakeOllama([SECTION, CONT]))       # CONT lists "LTV" and "CAC"
    assert doc.continuation.new == ["LTV"] and doc.degraded == []


# ---------- M4: verdicts in the claims section ----------

def test_verified_claims_show_the_verdict_what_is_actually_the_case_and_the_source(store):
    lid = lecture(store)
    claims = {c["text"]: c for c in store.claims(lid)}
    store.set_verdict(claims["טענה 3"]["id"], status="verified", verdict="incorrect", confidence=0.9,
                      explanation="בפועל הפונקציה מחזירה None.", sources=["https://docs.python.org/3/"])
    store.set_verdict(claims["טענה 2"]["id"], status="unchecked")
    doc = run(store, lid, FakeOllama([SECTION, EXEC]))
    md = render_markdown(doc)
    assert ('- המרצה אמר: "טענה 3" · [לא נכון]\n  בפועל: בפועל הפונקציה מחזירה None. · מקור: https://docs.python.org/3/'
            in md)
    assert '- המרצה אמר: "טענה 2" · [לא נבדק — אין רשת]' in md
    assert doc.claims[0].verdict == "incorrect" and doc.claims[0].sources == ["https://docs.python.org/3/"]


# ---------- the saved Digest (M6: re-sync without the model) ----------

def test_a_saved_digest_is_rebuilt_without_the_model(store):
    from lecture_copilot.output.digest import saved_digest
    lid = lecture(store)
    fake = FakeOllama([SECTION, EXEC])
    doc = run(store, lid, fake)
    again = saved_digest(lid, store)
    assert render_markdown(again) == render_markdown(doc)
    assert again.full_summary == doc.full_summary and again.exec_summary == doc.exec_summary
    assert again.course_id == doc.course_id and [c.id for c in again.claims] == [c.id for c in doc.claims]
    assert len(fake.requests) == 2          # nothing new was asked of the model


def test_a_lecture_without_a_saved_digest_has_none(store):
    from lecture_copilot.output.digest import saved_digest
    assert saved_digest(lecture(store), store) is None


def test_the_saved_continuation_survives(store):
    from lecture_copilot.output.digest import saved_digest
    first = lecture(store, week=4)
    run(store, first, FakeOllama([SECTION, EXEC]))
    second = lecture(store, week=5)
    exec2 = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(1, 6)],
                        "continuation": {"new": ["LTV"], "repeated": ["CAC"], "contradicts": []}}, ensure_ascii=False)
    doc = run(store, second, FakeOllama([SECTION, exec2]))
    again = saved_digest(second, store)
    assert again.continuation == doc.continuation and again.prev_title == doc.prev_title
