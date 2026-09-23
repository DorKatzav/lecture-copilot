import json

import pytest

from scripts import stage0


def test_percentile_nearest_rank():
    assert stage0.percentile([5, 1, 3, 2, 4], 50) == 3
    assert stage0.percentile(list(range(1, 11)), 95) == 10
    assert stage0.percentile([7.0], 95) == 7.0


def test_mw_verdict_hot_when_warm_runs_fit_budget():
    v = stage0.mw_verdict([14.0, 5.0, 5.2, 4.9, 5.1], budget_s=8)
    assert v["verdict"] == "hot" and v["decision"] == "mw"
    assert v["cold_s"] == 14.0 and v["p50_s"] == 5.1


def test_mw_verdict_cold_when_warm_runs_exceed_budget():
    v = stage0.mw_verdict([12.0, 9.5, 9.1, 10.0, 9.8], budget_s=8)
    assert v["verdict"] == "cold" and v["decision"] == "mlx"


def test_mw_verdict_hang_forces_mlx_even_if_fast():
    v = stage0.mw_verdict([3.0, 3.0, None, 3.0, 3.0], budget_s=8)
    assert v["hangs"] == 1 and v["decision"] == "mlx"


def test_mw_verdict_needs_five_runs():
    with pytest.raises(ValueError):
        stage0.mw_verdict([1.0, 1.0], budget_s=8)


def test_hebrew_share():
    assert stage0.hebrew_share("המרצה הסביר את CAC") > 0.7
    assert stage0.hebrew_share("The lecturer explained CAC") == 0.0
    assert stage0.hebrew_share("123 !!") == 0.0


def test_cut_offsets_spread_inside_margins_without_overlap():
    offs = stage0.cut_offsets(duration_s=5400, n=10, chunk_s=45, margin_s=300)
    assert len(offs) == 10 and offs == sorted(offs)
    assert offs[0] >= 300 and offs[-1] + 45 <= 5400 - 300
    assert all(b - a >= 45 for a, b in zip(offs, offs[1:], strict=False))


def test_cut_offsets_rejects_a_too_short_file():
    with pytest.raises(ValueError):
        stage0.cut_offsets(duration_s=900, n=10, chunk_s=45, margin_s=300)


def test_update_results_merges_and_keeps_other_keys(tmp_path):
    path = tmp_path / "stage0.json"
    stage0.update_results(path, "mw", {"p50_s": 4.1})
    stage0.update_results(path, "mic", {"readable": "yes"})
    data = json.loads(path.read_text())
    assert data["mw"] == {"p50_s": 4.1} and data["mic"] == {"readable": "yes"}


def test_summarize_extract_counts_first_attempt_validity():
    calls = [
        {"first_valid": True, "valid": True, "ms": 4000, "hebrew": 0.9, "concepts": 2, "claims": 1},
        {"first_valid": False, "valid": True, "ms": 9000, "hebrew": 0.8, "concepts": 0, "claims": 0},
        {"first_valid": False, "valid": False, "ms": 12000, "hebrew": 0.0, "concepts": 0, "claims": 0},
    ]
    s = stage0.summarize_extract(calls)
    assert (s["n"], s["valid_first"], s["valid_after_retry"]) == (3, 1, 2)
    assert s["p50_ms"] == 9000 and s["p95_ms"] == 12000
    assert s["hebrew_summaries"] == 2 and s["concepts"] == 2 and s["claims"] == 1


def test_fill_metrics_replaces_span_contents_in_place():
    results = {"mw": {"p50_s": 4.2}, "extract": {"qwen3:8b": {"n": 10}}}
    html = '<p><span data-metric="mw.p50_s">?</span> n=<span data-metric="extract.qwen3:8b.n">old</span></p>'
    out = stage0.fill_metrics(html, results)
    assert out == '<p><span data-metric="mw.p50_s">4.2</span> n=<span data-metric="extract.qwen3:8b.n">10</span></p>'
    assert stage0.fill_metrics(out, results) == out  # idempotent


def test_fill_metrics_unknown_key_raises():
    with pytest.raises(KeyError, match="mw.p99_s"):
        stage0.fill_metrics('<span data-metric="mw.p99_s"></span>', {"mw": {}})


def test_mw_cmd_is_explicit_and_writes_one_json_file(tmp_path):
    cmd = stage0.mw_cmd(tmp_path / "c.wav", tmp_path / "c.json", "he", model="whisperkit:some-model", binary="mw")
    assert cmd[:3] == ["mw", "transcribe", str(tmp_path / "c.wav")]
    assert cmd[cmd.index("--model") + 1] == "whisperkit:some-model"
    assert cmd[cmd.index("--language") + 1] == "he"
    assert cmd[cmd.index("--format") + 1] == "json"
    assert cmd[cmd.index("-o") + 1] == str(tmp_path / "c.json")
    assert "--no-speakers" in cmd and "--overwrite" in cmd


def test_mw_text_reads_the_mw_14_json_shape(tmp_path):
    out = tmp_path / "c.json"
    out.write_text(json.dumps({"text": " alpha beta gamma ", "segments": [
        {"id": "1", "start": 20, "end": 3880, "text": "alpha beta", "words": []},
        {"id": "2", "start": 3900, "end": 5000, "text": "gamma", "words": []}]}))
    assert stage0.mw_text(out) == "alpha beta gamma"


def test_mw_text_unknown_shape_raises(tmp_path):
    out = tmp_path / "c.json"
    out.write_text(json.dumps({"transcript": "x"}))
    with pytest.raises(RuntimeError, match="unknown mw json shape"):
        stage0.mw_text(out)


def test_mw_text_missing_file_raises(tmp_path):
    with pytest.raises(RuntimeError, match="no json"):
        stage0.mw_text(tmp_path / "nope.json")


def test_segment_coverage_sums_spans_in_seconds():
    segs = [{"start": 0, "end": 11400}, {"start": 11400, "end": 14500}, {"start": 30000, "end": 30600}]
    assert stage0.segment_coverage(segs) == 15.1


def test_level_stats_separates_floor_from_speech():
    import numpy as np
    sr = 16000
    quiet = np.full(sr * 2, 10 ** (-50 / 20), dtype=np.float32)  # 2 s at -50 dBFS
    loud = np.full(sr * 2, 10 ** (-30 / 20), dtype=np.float32)   # 2 s at -30 dBFS
    s = stage0.level_stats(np.concatenate([quiet, loud]), sr=sr, floor_db=-40)
    assert s["p10_db"] == -50.0 and s["p90_db"] == -30.0
    assert s["share_above_floor"] == 0.5


def test_fill_metrics_works_on_any_element_and_attribute_order():
    results = {"mic": {"words": 150, "by_model": {"whisper-cpp:ivrit-ai-largev3": {"asr_s": 16.35}}}}
    html = ('<td class="num" data-metric="mic.words">?</td>'
            '<span class="num" data-metric="mic.by_model.whisper-cpp:ivrit-ai-largev3.asr_s" title="t">?</span>')
    out = stage0.fill_metrics(html, results)
    assert out.startswith('<td class="num" data-metric="mic.words">150</td>')
    assert out.endswith('asr_s" title="t">16.35</span>')


@pytest.mark.parametrize(("language", "model"), [("he", "whisper-cpp:ivrit-ai-largev3"),
                                                 ("en", "whisperkit:openai_whisper-large-v3-v20240930")])
def test_mw_cmd_picks_the_model_from_the_course_language(tmp_path, language, model):
    cmd = stage0.mw_cmd(tmp_path / "c.wav", tmp_path / "c.json", language)
    assert cmd[cmd.index("--model") + 1] == model
