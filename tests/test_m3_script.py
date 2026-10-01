import json

from lecture_copilot.agents.schemas import Concept, ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.store.db import Store
from scripts import m3
from tests.stubs import fake_embedding


def seed(tmp_path):
    s = Store(tmp_path / "c.sqlite")
    course = s.upsert_course("c", language="he")
    a = s.upsert_lecture(course, audio_path="/a.vtt", source="transcript", title="7-6", date="2026-06-07",
                         fact_check=True)
    b = s.upsert_lecture(course, audio_path="/b.vtt", source="transcript", title="9-6", date="2026-06-09",
                         fact_check=True)
    commit_explained = "שמירת השינויים בקוד למאגר המקומי של הגרסאות"
    for lid, concepts in ((a, [("git", "מערכת גרסאות", "git"), ("commit", commit_explained, "commit")]),
                          (b, [("Git", "ניהול גרסאות", "git"), ("branch", "ענף", "branch"),
                               ("קומיט", commit_explained, "save_changes")])):
        res = ExtractResult(chunk_summary="s", items=[], claims=[],
                            concepts=[Concept(term=t, explanation=e, canonical_key=k) for t, e, k in concepts])
        s.write_chunk(lid, 1, [Segment(t0=0, t1=40, text="x")], asr="transcript", result=res)
        for r in s.items(lid, kind="concept"):
            s.set_embedding("items", r["id"], fake_embedding(f"{r['text']} {r['explanation']}"))
    git_b = next(r for r in s.items(b, kind="concept") if r["text"] == "Git")
    s.set_first_seen(git_b["id"], a)
    return s, course, a, b


def test_candidates_pair_the_later_lectures_concepts_with_earlier_ones(tmp_path):
    s, course, a, b = seed(tmp_path)
    rows = m3.candidate_pairs(s, a, b)
    by_term = {r["term"]: r for r in rows}
    assert by_term["Git"]["match"] == "git" and by_term["Git"]["how"] == "key" and by_term["Git"]["flagged"]
    k = by_term["קומיט"]
    assert k["match"] == "commit" and k["how"] == "meaning" and not k["flagged"]
    assert "branch" not in by_term
    s.close()


def test_candidates_json_is_a_benchmark_template(tmp_path):
    s, course, a, b = seed(tmp_path)
    text = m3.candidates_json(m3.candidate_pairs(s, a, b), first="2026-06-07", again="2026-06-09")
    data = json.loads(text)
    assert [c["term"] for c in data["shared_concepts"]] == ["Git", "קומיט"]
    assert data["shared_concepts"][0] == {"term": "Git", "first": "2026-06-07", "again": "2026-06-09"}
    s.close()


def test_memory_stats_of_a_lecture(tmp_path):
    s, course, a, b = seed(tmp_path)
    s.log("run", lecture_id=b, input_ref="R", output={"chunks": 1, "status": {"ok": 1}, "counts": {},
                                                     "audio_s": 40, "already_said": 1, "contradictions": 0,
                                                     "timing": {"memory_s": {"p95": 0.8}}})
    s.close()
    out = m3.memory_stats(tmp_path / "c.sqlite", b)
    assert out["concepts"] == 3 and out["returned"] == 1 and out["memory_p95_s"] == 0.8 and out["already_said"] == 1
