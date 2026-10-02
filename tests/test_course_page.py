import re

from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult, Item
from lecture_copilot.asr.base import Segment
from lecture_copilot.output.course_page import course_page, render_course_html
from lecture_copilot.output.sinks import FolderSink
from lecture_copilot.store.db import Store


def course_with_two_lectures(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("יזמות וחדשנות", language="he")
    w4 = s.upsert_lecture(course, audio_path="/w4", source="file", title="מודלים א'", date="2026-10-28",
                          fact_check=True, week=4)
    w5 = s.upsert_lecture(course, audio_path="/w5", source="file", title="מודלים ב'", date="2026-11-04",
                          fact_check=True, week=5)
    s.write_chunk(w4, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=ExtractResult(
        chunk_summary="s", concepts=[Concept(term="CAC", explanation="עלות רכישת לקוח", canonical_key="cac")],
        claims=[Claim(text="Dropbox הגיעה ל-4%", normalized="n", importance=90)],
        items=[Item(kind="highlight", text="CAC במבחן"), Item(kind="question", text="מה עם LTV?")]))
    s.write_chunk(w5, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=ExtractResult(
        chunk_summary="s", concepts=[Concept(term="cac", explanation="הסבר חוזר", canonical_key="CAC"),
                                     Concept(term="LTV", explanation="ערך חיי לקוח", canonical_key="ltv")],
        claims=[],
        items=[Item(kind="action", text="לקרוא פרק 4", due="2026-11-11"), Item(kind="note", text="הערה שלי")]))
    for r in s.items(w5, kind="concept"):
        if r["canonical_key"] == "CAC":
            s.set_first_seen(r["id"], w4)
    claim = s.claims(w4)[0]
    s.set_verdict(claim["id"], status="verified", verdict="correct", confidence=0.9, explanation="נכון", sources=["https://a"])
    s.save_digest(w4, bullets=["א"], digest_md="# w4")
    s.save_digest(w5, bullets=["ב"], digest_md="# w5")
    return s, course, w4, w5


def test_glossary_is_cumulative_and_deduplicated_by_key(tmp_path):
    s, course, w4, w5 = course_with_two_lectures(tmp_path)
    page = course_page(course, s)
    assert [(g.term, g.first_seen) for g in page.glossary] == [("CAC", "W04"), ("LTV", "W05")]
    assert page.glossary[0].explanation == "עלות רכישת לקוח"          # the first explanation wins
    s.close()


def test_stars_claims_questions_and_tasks_across_lectures(tmp_path):
    s, course, w4, w5 = course_with_two_lectures(tmp_path)
    page = course_page(course, s)
    assert [(h.text, h.lecture) for h in page.highlights] == [("CAC במבחן", "W04")]
    assert page.claims[0].text == "Dropbox הגיעה ל-4%" and page.claims[0].verdict == "correct"
    assert [q.text for q in page.questions] == ["מה עם LTV?"] and page.tasks[0].due == "2026-11-11"
    assert [lec.label for lec in page.lectures] == ["W05 · מודלים ב'", "W04 · מודלים א'"]   # newest first
    s.close()


def test_course_html_is_rtl_with_a_search_box_and_the_data_inline(tmp_path):
    s, course, w4, w5 = course_with_two_lectures(tmp_path)
    html = render_course_html(course_page(course, s))
    assert '<html lang="he" dir="rtl">' in html and 'type="search"' in html
    assert "עלות רכישת לקוח" in html and "Dropbox הגיעה ל-4%" in html and "לקרוא פרק 4" in html
    assert re.search(r"<h2[^>]*>מילון", html) and "<script>" in html
    s.close()


def test_folder_sink_writes_course_html(tmp_path):
    s, course, w4, w5 = course_with_two_lectures(tmp_path)
    path = FolderSink(tmp_path / "courses").write_course(course_page(course, s))
    assert path == tmp_path / "courses" / "יזמות וחדשנות" / "course.html" and "CAC" in path.read_text(encoding="utf-8")
    s.close()
