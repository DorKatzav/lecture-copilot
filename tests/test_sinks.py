import json
import re

import pytest

from lecture_copilot.agents.schemas import Continuation
from lecture_copilot.output.digest import SECTIONS, ClaimRow, ConceptRow, DigestDoc, TaskRow, render_markdown
from lecture_copilot.output.sinks import FolderSink, folder_name, render_html, safe_name


def doc(**kw):
    base = dict(
        lecture_id="L1", course_name="יזמות וחדשנות", title="מודלים עסקיים ב'", date="2026-11-04", week=5,
        minutes=88, language="he", exec_summary=[f"נקודה {i}" for i in range(1, 6)],
        full_summary=["פסקה ראשונה.", "פסקה שנייה על CAC."], highlights=["זה במבחן"],
        concepts=[ConceptRow("CAC", "עלות רכישת לקוח", "cac")],
        claims=[ClaimRow("CAC ירד ב-2024", 90, "pending", "עדיין לא נבדק")],
        all_claims=[{"id": "C1", "text": "CAC ירד ב-2024", "normalized": "CAC fell in 2024", "importance": 90,
                     "status": "pending", "verdict": None, "confidence": None, "sources_json": None,
                     "embedding": b"\x00\x01", "lecture_id": "L1", "segment_id": "S1", "cache_key": None},
                    {"id": "C2", "text": "פרט קטן", "normalized": "minor", "importance": 40, "status": "pending",
                     "verdict": None, "confidence": None, "sources_json": '["https://example.org"]',
                     "embedding": None, "lecture_id": "L1", "segment_id": "S1", "cache_key": None}],
        questions=["למה?"], tasks=[TaskRow("לקרוא פרק 3", "דור", "2026-11-11")], notes=[],
        segments=[{"t0": 0.0, "t1": 9.0, "text": "שלום לכולם", "speaker": None},
                  {"t0": 3725.5, "t1": 3730.0, "text": "נדבר על CAC", "speaker": "דנה"}])
    base.update(kw)
    return DigestDoc(**base)


# ---------- names ----------

def test_folder_name_has_week_date_and_slug():
    assert folder_name(doc()) == "W05_2026-11-04_מודלים-עסקיים-ב'"


def test_folder_name_without_a_week():
    assert folder_name(doc(week=None)) == "2026-11-04_מודלים-עסקיים-ב'"


@pytest.mark.parametrize("raw, safe", [("AI Developers — Python", "AI Developers — Python"),
                                       ("7/6 Lecture: intro?", "7-6 Lecture- intro"), ("  ..hidden  ", "hidden"),
                                       ("", "untitled")])
def test_names_are_safe_for_a_synced_folder(raw, safe):
    assert safe_name(raw) == safe


# ---------- files ----------

def test_lecture_folder_has_the_four_files(tmp_path):
    folder = FolderSink(tmp_path).write_lecture(doc())
    assert folder == tmp_path / "יזמות וחדשנות" / "W05_2026-11-04_מודלים-עסקיים-ב'"
    assert sorted(p.name for p in folder.iterdir()) == ["claims.json", "digest.html", "digest.md", "transcript.txt"]


def test_digest_md_is_the_rendered_markdown(tmp_path):
    d = doc()
    assert (FolderSink(tmp_path).write_lecture(d) / "digest.md").read_text(encoding="utf-8") == render_markdown(d)


def test_transcript_has_timestamps_and_speakers(tmp_path):
    text = (FolderSink(tmp_path).write_lecture(doc()) / "transcript.txt").read_text(encoding="utf-8")
    assert text == "[00:00:00] שלום לכולם\n[01:02:05] דנה: נדבר על CAC\n"


def test_claims_json_keeps_every_claim_without_internals(tmp_path):
    rows = json.loads((FolderSink(tmp_path).write_lecture(doc()) / "claims.json").read_text(encoding="utf-8"))
    assert [r["text"] for r in rows] == ["CAC ירד ב-2024", "פרט קטן"]
    assert rows[1]["sources"] == ["https://example.org"] and rows[0]["sources"] == []
    assert set(rows[0]) == {"text", "normalized", "importance", "status", "verdict", "confidence", "sources"}


def test_files_use_unix_newlines(tmp_path):
    folder = FolderSink(tmp_path).write_lecture(doc())
    assert all(b"\r" not in p.read_bytes() for p in folder.iterdir())


def test_writing_again_replaces_the_files(tmp_path):
    sink = FolderSink(tmp_path)
    sink.write_lecture(doc())
    folder = sink.write_lecture(doc(exec_summary=[f"חדש {i}" for i in range(5)]))
    assert "חדש 0" in (folder / "digest.md").read_text(encoding="utf-8")
    assert len(list((tmp_path / "יזמות וחדשנות").iterdir())) == 2          # the lecture folder + index.md


def test_course_index_lists_lectures_newest_first(tmp_path):
    sink = FolderSink(tmp_path)
    sink.write_lecture(doc(lecture_id="L0", week=4, date="2026-10-28", title="מודלים עסקיים א'"))
    sink.write_lecture(doc())
    index = (tmp_path / "יזמות וחדשנות" / "index.md").read_text(encoding="utf-8")
    assert index.splitlines() == [
        "# יזמות וחדשנות", "",
        "- [W05 · מודלים עסקיים ב' · 4.11.2026](<W05_2026-11-04_מודלים-עסקיים-ב'/digest.md>)",
        "- [W04 · מודלים עסקיים א' · 28.10.2026](<W04_2026-10-28_מודלים-עסקיים-א'/digest.md>)"]


# ---------- html ----------

def test_html_is_hebrew_rtl_with_the_nine_sections_in_order():
    html = render_html(doc())
    assert '<html lang="he" dir="rtl">' in html
    heads = [re.sub(r"<[^>]+>", "", h).strip() for h in re.findall(r"<h2[^>]*>(.*?)</h2>", html, re.DOTALL)]
    assert [next(s for s in SECTIONS if h.startswith(s)) for h in heads] == SECTIONS


def test_html_escapes_what_the_model_wrote():
    html = render_html(doc(full_summary=["<script>alert(1)</script> & co"]))
    assert "<script>alert(1)" not in html and "&lt;script&gt;alert(1)&lt;/script&gt; &amp; co" in html


def test_html_isolates_terms_and_numbers():
    html = render_html(doc())
    assert "<bdi>CAC</bdi>" in html and '<span class="num">88</span>' in html


def test_html_shows_the_continuation_when_there_is_one():
    d = doc(prev_title="W04 · מודלים עסקיים א'", continuation=Continuation(new=["LTV"], repeated=["CAC"],
                                                                           contradicts=[]))
    html = render_html(d)
    assert "המשך מ-" in html and "מה סותר" in html and "זו ההרצאה הראשונה" not in html


def test_html_says_when_the_digest_is_degraded():
    assert "class=\"callout warn\"" in render_html(doc(degraded=["exec"], exec_summary=[]))
    assert "class=\"callout warn\"" not in render_html(doc())


def test_the_root_is_not_invented_when_its_parent_is_missing(tmp_path):
    # COURSES_ROOT defaults to a Google Drive path: never create a fake "Google Drive" folder
    sink = FolderSink(tmp_path / "Google Drive" / "My Drive" / "Lecture-Copilot")
    with pytest.raises(OSError, match="does not exist"):
        sink.write_lecture(doc())
    assert not (tmp_path / "Google Drive").exists()


# ---------- bidi: found in a browser screenshot, not by reading the markup ----------

@pytest.mark.parametrize("text, expected", [
    ("Dropbox הגיעה ל-4% משלמים במודל Freemium", "rtl"),      # a Hebrew sentence that opens with a Latin word
    ("W04 · מודלים עסקיים א'", "rtl"),
    ("random.shuffle returns None", "ltr"),
    ("", "rtl"), ("2026", "rtl"),
])
def test_direction_follows_the_dominant_script_not_the_first_letter(text, expected):
    from lecture_copilot.output.sinks import textdir
    assert textdir(text) == expected


@pytest.mark.parametrize("text, html", [
    ("רק 2%–5% מהמשתמשים", 'רק <span class="num">2%–5%</span> מהמשתמשים'),
    ("קטעים של 30-60 שניות", 'קטעים של <span class="num">30-60</span> שניות'),
    ("בין 4 – 5.10", 'בין <span class="num">4 – 5.10</span>'),
    ("עד 2026-11-11", 'עד <span class="num">2026-11-11</span>'),
    ("ל-4% משלמים, פי 3", "ל-4% משלמים, פי 3"),                 # single numbers are left to the browser
    ("יחס LTV/CAC של 3", "יחס LTV/CAC של 3"),
    ("<b>1-2</b> & co", '&lt;b&gt;<span class="num">1-2</span>&lt;/b&gt; &amp; co'),
])
def test_number_ranges_are_isolated_and_everything_is_escaped(text, html):
    from lecture_copilot.output.sinks import isolate_ranges
    assert str(isolate_ranges(text)) == html


def test_a_claim_that_opens_with_a_latin_word_keeps_its_order():
    html = render_html(doc(claims=[ClaimRow("Dropbox הגיעה ל-4% משלמים", 90, "pending", "עדיין לא נבדק")]))
    assert '<q dir="rtl">Dropbox הגיעה ל-4% משלמים</q>' in html and "<bdi>Dropbox" not in html


def test_ranges_in_generated_text_are_isolated_in_the_page():
    html = render_html(doc(exec_summary=["רק 2%–5% משלמים"] * 5, full_summary=["בין 30-60 שניות"]))
    assert html.count('<span class="num">2%–5%</span>') == 5 and '<span class="num">30-60</span>' in html
