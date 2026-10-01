from lecture_copilot.agents.ranker import LABELS, rank
from lecture_copilot.agents.schemas import Claim, ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.store.db import Store


def test_rank_orders_by_importance_and_labels_material(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("c", language="he")
    lid = s.upsert_lecture(course, audio_path="/x", source="file", title="t", date="d", fact_check=True)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[], claims=[
        Claim(text="minor", normalized="n", importance=40), Claim(text="major", normalized="n", importance=90),
        Claim(text="material", normalized="n", importance=70)])
    s.write_chunk(lid, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    s.set_verdict([c for c in s.claims(lid) if c["text"] == "major"][0]["id"], status="verified", verdict="incorrect",
                  confidence=0.9, explanation="לא נכון", sources=["https://a"])
    rows = rank(lid, s)
    assert [(r.text, r.label) for r in rows] == [("major", "material"), ("material", "minor"), ("minor", "minor")]
    assert rows[0].verdict == "incorrect" and rows[0].verdict_he == LABELS["incorrect"] and rows[0].sources == ["https://a"]
    assert rows[1].verdict_he == LABELS["pending"] and rows[2].status == "pending"
    s.close()


def test_unverifiable_and_first_person_claims_rank_below_facts(tmp_path):
    from lecture_copilot.agents.ranker import score
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("c", language="he")
    lid = s.upsert_lecture(course, audio_path="/x", source="file", title="t", date="d", fact_check=True)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[], claims=[
        Claim(text="המטרה שלי זה לאתגר אתכם", normalized="n", importance=95),       # first person: about the class
        Claim(text="הפונקציה מדפיסה את ערך X כ-50", normalized="n", importance=95),  # verified unverifiable
        Claim(text="print לא מחזירה כלום", normalized="n", importance=90),           # a fact about the world
        Claim(text="git שומר גרסאות", normalized="n", importance=85)])
    s.write_chunk(lid, 1, [Segment(t0=0, t1=40, text="x")], asr="mw", result=res)
    by = {c["text"]: c["id"] for c in s.claims(lid)}
    s.set_verdict(by["הפונקציה מדפיסה את ערך X כ-50"], status="verified", verdict="unverifiable", confidence=0.9,
                  explanation="דוגמה", sources=[])
    s.set_verdict(by["print לא מחזירה כלום"], status="verified", verdict="correct", confidence=1.0, explanation="כן",
                  sources=["https://a"])
    assert [r.text for r in rank(lid, s)] == ["print לא מחזירה כלום", "git שומר גרסאות",
                                             "המטרה שלי זה לאתגר אתכם", "הפונקציה מדפיסה את ערך X כ-50"]
    assert score(95, None, "המטרה שלי") < score(85, None, "git שומר")
    assert score(95, "unverifiable", "x") < score(70, None, "x")
    s.close()
