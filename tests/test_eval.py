import asyncio
import json

from lecture_copilot.agents.schemas import Claim, ExtractResult
from lecture_copilot.asr.base import Segment
from lecture_copilot.eval import precision_at_k, run_eval, score_verdicts
from lecture_copilot.store.db import Store
from tests.stubs import FakeGemini


def test_verdict_scoring_counts_exact_matches_and_errors_caught():
    labels = [{"verdict": "correct", "injected": False}, {"verdict": "incorrect", "injected": True},
              {"verdict": "incorrect", "injected": True}, {"verdict": "unverifiable", "injected": False}]
    preds = ["correct", "incorrect", "imprecise", "correct"]
    s = score_verdicts(labels, preds)
    assert s["n"] == 4 and s["accuracy"] == 0.5 and s["injected"] == 2 and s["injected_caught"] == 2
    assert s["confusion"]["incorrect"] == {"incorrect": 1, "imprecise": 1}
    assert s["confusion"]["unverifiable"] == {"correct": 1}


def test_precision_at_k():
    labels = {"a": True, "b": False, "c": True, "d": True, "e": True}
    assert precision_at_k(["a", "b", "c", "d", "e", "f"], labels) == 0.8
    assert precision_at_k(["x"], {}) == 0.0


def test_run_eval_scores_the_benchmark_on_a_scratch_copy(tmp_path):
    s = Store(tmp_path / "copilot.sqlite")
    course = s.upsert_course("AI Developers — Python", language="he")
    lid = s.upsert_lecture(course, audio_path="/a.vtt", source="transcript", title="7-6", date="2026-06-07",
                           fact_check=True)
    res = ExtractResult(chunk_summary="s", items=[], concepts=[], claims=[
        Claim(text="print לא מחזירה כלום", normalized="print returns None", importance=95),
        Claim(text="המפגש הבא ב-14 ביוני", normalized="next meeting June 14", importance=90)])
    s.write_chunk(lid, 1, [Segment(t0=0, t1=40, text="x")], asr="transcript", result=res)
    s.close()
    bench = {"labels_by": "test", "claims": [
        {"text": "print לא מחזירה כלום", "normalized": "print returns None", "lecture": "2026-06-07",
         "verdict": "correct", "injected": False, "contradicts_7_6": False},
        {"text": "print מחזירה ערך", "normalized": "print returns a value", "lecture": "2026-06-07",
         "verdict": "incorrect", "injected": True, "contradicts_7_6": True}],
        "material": {"2026-06-07": {"print לא מחזירה כלום": True, "המפגש הבא ב-14 ביוני": False}}}
    (tmp_path / "benchmark.json").write_text(json.dumps(bench, ensure_ascii=False), encoding="utf-8")
    ok = {"verdict": "correct", "confidence": 0.9, "explanation": "נכון.", "sources": ["https://a"]}
    bad = {"verdict": "incorrect", "confidence": 0.9, "explanation": "לא.", "sources": ["https://b"]}
    gemini = FakeGemini([ok, bad])
    out = asyncio.run(run_eval(tmp_path / "copilot.sqlite", tmp_path / "benchmark.json", gemini,
                               results_path=tmp_path / "results.json", work=tmp_path / "work"))
    assert out["labels_by"] == "test" and out["verdicts"]["accuracy"] == 1.0 and out["verdicts"]["injected_caught"] == 1
    assert out["precision_at_5"]["2026-06-07"] == {"k": 1, "precision": 1.0, "material_labelled": 1}
    assert out["cache_hit"] is True and out["cost_usd"] > 0
    assert len(gemini.calls) == 2                         # the cache-hit probe makes no third call
    assert json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))["verdicts"]["n"] == 2
    assert Store(tmp_path / "copilot.sqlite").con.execute("select count(*) from claims").fetchone()[0] == 2  # untouched
