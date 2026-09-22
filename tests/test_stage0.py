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
