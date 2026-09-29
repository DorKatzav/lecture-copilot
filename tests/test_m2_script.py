from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult, Item
from scripts import m2


def res(concepts=0, claims=(), kinds=()):
    return ExtractResult(chunk_summary="s", concepts=[Concept(term="t", explanation="e", canonical_key="k")] * concepts,
                         claims=[Claim(text="c", normalized="n", importance=i) for i in claims],
                         items=[Item(kind=k, text="x") for k in kinds])


def test_counts_by_kind_over_all_chunks():
    out = m2.count_extraction([res(2, (90, 60), ("highlight", "question")), res(1, (), ("highlight",)), None])
    assert out == {"chunks": 3, "failed": 1, "concepts": 3, "claims": 2, "claims_flagged": 1, "highlights": 2,
                   "questions": 1, "actions": 0, "chunks_with_a_claim": 1}


def test_share_of_summaries_that_mention_the_lecturer():
    assert m2.lecturer_mentions(["המרצה פותח תרגיל", "git שומר גרסאות", "המרצה ממליצה", "הסטודנטים שאלו"]) == 2


def test_the_sample_lecture_fills_every_section():
    from lecture_copilot.output.digest import SECTIONS, render_markdown, section_headings
    md = render_markdown(m2.sample_doc())
    assert section_headings(md) == SECTIONS and "אין." not in md and "המשך מ-W04" in md


def test_digest_stats_count_what_each_section_holds():
    md = m2.render_markdown(m2.sample_doc())
    out = m2.digest_stats(md)
    assert out["sections"] == 9 and out["concepts"] == 5 and out["highlights"] == 2 and out["claims_flagged"] == 3
    assert out["questions"] == 2 and out["tasks"] == 2 and out["exec_bullets"] == 5
    assert out["full_summary_words"] > 50 and out["full_summary_sentences_naming_the_lecturer"] == 0
