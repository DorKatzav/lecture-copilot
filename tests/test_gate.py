import json
import subprocess

import httpx
import pytest

from lecture_copilot.config import LIVE_MODEL
from scripts import gate
from tests.stubs import FakeOllama

FAKE_GEMINI = "AIza" + "Q" * 35  # built at runtime so this file never matches the scan itself
FAKE_GEMINI_AQ = "AQ." + "Ab8RN6-x_Z." + "k" * 39  # the newer AI Studio key format (53 chars)


def git_repo(tmp_path, files: dict[str, str]):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args],
                   check=True, capture_output=True)


# ---- secret scan ----

def test_secret_scan_clean_repo_passes(tmp_path):
    root = git_repo(tmp_path, {"a.py": "KEY = os.environ['GEMINI_API_KEY']\n"})
    assert gate.check_secret_scan(root).status == "PASS"


def test_secret_scan_catches_untracked_key_and_never_prints_it(tmp_path):
    root = git_repo(tmp_path, {"a.py": f"KEY = '{FAKE_GEMINI}'\n"})
    r = gate.check_secret_scan(root)
    assert r.status == "FAIL" and "a.py" in r.detail
    assert FAKE_GEMINI not in r.detail


def test_secret_scan_catches_the_newer_aq_key_format(tmp_path):
    root = git_repo(tmp_path, {".env.example": f"GEMINI_API_KEY={FAKE_GEMINI_AQ}\n"})
    r = gate.check_secret_scan(root)
    assert r.status == "FAIL" and ".env.example" in r.detail
    assert FAKE_GEMINI_AQ not in r.detail


def test_secret_scan_ignores_gitignored_env(tmp_path):
    root = git_repo(tmp_path, {".gitignore": ".env\n", ".env": f"GEMINI_API_KEY={FAKE_GEMINI}\n"})
    assert gate.check_secret_scan(root).status == "PASS"


def test_secret_scan_does_not_match_its_own_pattern():
    assert gate.check_secret_scan(gate.ROOT).status == "PASS"


# ---- mw ----

def test_mw_missing_binary_fails():
    r = gate.check_mw_version(cmd=("definitely-not-mw-xyz", "version"))
    assert r.status == "FAIL" and "PATH" in r.detail


def test_mw_nonzero_exit_fails():
    assert gate.check_mw_version(cmd=("false",)).status == "FAIL"


def test_mw_ok():
    r = gate.check_mw_version(cmd=("echo", "mw 13.1"))
    assert r.status == "PASS" and "13.1" in r.detail


# ---- ollama ----

def test_ollama_all_models_present():
    assert gate.check_ollama_models(client=FakeOllama().client()).status == "PASS"


def test_ollama_missing_model_is_named():
    r = gate.check_ollama_models(client=FakeOllama(models=["qwen3:8b", "bge-m3:latest"]).client())
    assert r.status == "FAIL" and "gemma3:12b" in r.detail


def test_ollama_down_fails():
    def refuse(_):
        raise httpx.ConnectError("refused")
    client = httpx.Client(transport=httpx.MockTransport(refuse), base_url="http://127.0.0.1:11434")
    r = gate.check_ollama_models(client=client)
    assert r.status == "FAIL" and "not reachable" in r.detail


# ---- sqlite-vec ----

def test_sqlite_vec_loads_for_real():
    assert gate.check_sqlite_vec(results={}).status == "PASS"


def test_sqlite_vec_failure_without_flagged_fallback_fails():
    def broken(_):
        raise AttributeError("enable_load_extension")
    assert gate.check_sqlite_vec(results={}, loader=broken).status == "FAIL"


def test_sqlite_vec_failure_with_flagged_numpy_fallback_passes():
    def broken(_):
        raise AttributeError("enable_load_extension")
    r = gate.check_sqlite_vec(results={"sqlite_vec": {"fallback": "numpy"}}, loader=broken)
    assert r.status == "PASS" and "numpy" in r.detail


# ---- fixture ----

def test_fixture_missing_fails(tmp_path):
    assert gate.check_fixture(tmp_path / "fixture_10min.m4a").status == "FAIL"


def test_fixture_wrong_length_fails(tmp_path):
    f = tmp_path / "fixture_10min.m4a"
    f.write_bytes(b"x")
    assert gate.check_fixture(f, duration=lambda _: 180.0).status == "FAIL"


def test_fixture_ten_minutes_passes(tmp_path):
    f = tmp_path / "fixture_10min.m4a"
    f.write_bytes(b"x")
    assert gate.check_fixture(f, duration=lambda _: 600.02).status == "PASS"


# ---- stage-0 results ----

MW_OK = {"runs_s": [9.1, 4.0, 4.1, 3.9, 4.2], "p50_s": 4.1, "verdict": "hot", "decision": "mw"}


def test_mw_bench_passes_with_five_runs_and_verdict():
    assert gate.check_mw_bench({"mw": MW_OK}).status == "PASS"


def test_mw_bench_fails_without_verdict():
    r = gate.check_mw_bench({"mw": {**MW_OK, "verdict": None}})
    assert r.status == "FAIL"


def test_mw_bench_fails_with_four_runs():
    assert gate.check_mw_bench({"mw": {**MW_OK, "runs_s": [1, 1, 1, 1]}}).status == "FAIL"


def test_extract_bench_nine_of_ten_passes():
    r = gate.check_extract_bench({"extract": {LIVE_MODEL: {"n": 10, "valid_first": 9}}})
    assert r.status == "PASS"


def test_extract_bench_eight_of_ten_fails():
    r = gate.check_extract_bench({"extract": {LIVE_MODEL: {"n": 10, "valid_first": 8}}})
    assert r.status == "FAIL" and "8/10" in r.detail


def test_extract_bench_too_few_chunks_fails():
    assert gate.check_extract_bench({"extract": {LIVE_MODEL: {"n": 5, "valid_first": 5}}}).status == "FAIL"


def test_mic_verdict_recorded_and_file_present(tmp_path):
    (tmp_path / "mic_seat.m4a").write_bytes(b"x")
    r = gate.check_mic({"mic": {"file": "mic_seat.m4a", "readable": "no"}}, root=tmp_path)
    assert r.status == "PASS" and "no" in r.detail


def test_mic_without_verdict_fails(tmp_path):
    (tmp_path / "mic_seat.m4a").write_bytes(b"x")
    assert gate.check_mic({"mic": {"file": "mic_seat.m4a", "readable": None}}, root=tmp_path).status == "FAIL"


def test_mic_recording_missing_fails(tmp_path):
    assert gate.check_mic({"mic": {"file": "gone.m4a", "readable": "yes"}}, root=tmp_path).status == "FAIL"


# ---- summary ----

def test_summary_all_pass():
    rs = [gate.Result("a", "PASS", ""), gate.Result("b", "PASS", "")]
    assert gate.summary(0, rs) == "GATE M0: PASS 2/2"


def test_summary_any_fail():
    rs = [gate.Result("a", "PASS", ""), gate.Result("b", "FAIL", "x"), gate.Result("c", "SKIP", "y")]
    assert gate.summary(0, rs) == "GATE M0: FAIL 1/3"


def test_summary_skip_without_fail():
    rs = [gate.Result("a", "PASS", ""), gate.Result("b", "SKIP", "y")]
    assert gate.summary(0, rs) == "GATE M0: SKIP 1/2"


def test_unknown_milestone_exits_nonzero(capsys):
    assert gate.main(["--m", "9"]) == 2


def test_crashing_check_is_a_named_fail_not_a_crash(monkeypatch, capsys):
    def boom():
        raise RuntimeError("disk on fire")
    monkeypatch.setitem(gate.CHECKS, 0, [("fixture", boom)])
    [r] = gate.run(0)
    assert (r.name, r.status) == ("fixture", "FAIL") and "disk on fire" in r.detail


def test_ollama_required_models_are_deduplicated():
    r = gate.check_ollama_models(client=FakeOllama(models=["gemma3:12b", "bge-m3:latest"]).client())
    assert r.status == "PASS" and r.detail == "gemma3:12b, bge-m3"


def test_secret_scan_catches_a_key_that_only_lives_in_history(tmp_path):
    root = git_repo(tmp_path, {".env.example": f"GEMINI_API_KEY={FAKE_GEMINI_AQ}\n"})
    git(root, "add", ".env.example")
    git(root, "commit", "-q", "-m", "oops")
    (root / ".env.example").write_text("GEMINI_API_KEY=\n")
    git(root, "commit", "-q", "-am", "remove key")
    r = gate.check_secret_scan(root)
    assert r.status == "FAIL" and "history" in r.detail and ".env.example" in r.detail
    assert FAKE_GEMINI_AQ not in r.detail


def test_secret_scan_catches_a_staged_key_deleted_from_the_working_file(tmp_path):
    root = git_repo(tmp_path, {"a.py": f"K = '{FAKE_GEMINI}'\n"})
    git(root, "add", "a.py")
    (root / "a.py").write_text("K = ''\n")
    r = gate.check_secret_scan(root)
    assert r.status == "FAIL" and "index" in r.detail and "a.py" in r.detail


def test_secret_scan_catches_notion_tokens(tmp_path):
    ntn, legacy = "ntn_" + "A" * 30, "secret_" + "B" * 30
    root = git_repo(tmp_path, {"n1.txt": ntn + "\n", "n2.txt": legacy + "\n"})
    r = gate.check_secret_scan(root)
    assert r.status == "FAIL" and "n1.txt" in r.detail and "n2.txt" in r.detail


def test_secret_scan_keeps_file_names_with_spaces_whole(tmp_path):
    root = git_repo(tmp_path, {"my notes.txt": FAKE_GEMINI + "\n"})
    assert "my notes.txt" in gate.check_secret_scan(root).detail


# ---------- M1 ----------

M1_FIXTURE = "/x/eval/fixture_10min.m4a"
GOOD_MEMORY = {"samples": 200, "peak_used_gb": 19.5, "total_gb": 24.0, "baseline_used_gb": 9.0, "max_pressure": 1,
               "swap_growth_mb": 0.0, "models_at_peak_gb": {"gemma3:12b": 8.3, "bge-m3": 0.6},
               "procs_peak_mb": {"MacWhisper": 4100, "llama-server": 1800}}


def m1_store(tmp_path, runs=2, counts=None, totals=(14.6, 17.8), memory=None, lectures=1, source_error=None,
             stage_p95=(5.2, 12.9)):
    from lecture_copilot.store.db import Store
    s = Store(tmp_path / "copilot.sqlite")
    course = s.upsert_course("AI Developers — Python", language="he")
    lid = s.upsert_lecture(course, audio_path=M1_FIXTURE, source="file", title="f", date="2026-06-19", fact_check=True)
    for i in range(1, lectures):
        s.upsert_lecture(course, audio_path=f"/other/{i}.m4a", source="file", title="o", date="d", fact_check=True)
    counts = counts or [{"segments": 58, "items": 28, "claims": 23}] * runs
    from lecture_copilot.store.db import new_id
    for r in range(runs):
        run_id = new_id(now_ms=1_000 + r)
        for i, total in enumerate(totals, 1):
            s.log("chunk", lecture_id=lid, input_ref=f"{run_id}#{i:04d}", ms=total * 1000,
                  output={"idx": i, "status": "ok", "total_s": total})
        out = {"chunks": len(totals), "status": {"ok": len(totals)}, "counts": counts[r],
               "timing": {"asr_s": {"p95": stage_p95[0]}, "extract_s": {"p95": stage_p95[1]},
                          "total_s": {"p95": max(totals), "max": max(totals)}},
               "memory": memory if memory is not None else GOOD_MEMORY}
        if source_error and r == runs - 1:
            out["source_error"] = source_error
        s.log("run", lecture_id=lid, input_ref=run_id, output=out)
    # the rows the last run left behind
    c = counts[-1] if counts else {"segments": 0, "items": 0, "claims": 0}
    from lecture_copilot.agents.schemas import Claim, Concept, ExtractResult
    from lecture_copilot.asr.base import Segment
    for i in range(c["segments"]):
        res = None
        if i == 0:
            res = ExtractResult(chunk_summary="s", items=[],
                                concepts=[Concept(term="t", explanation="e", canonical_key="k")] * c["items"],
                                claims=[Claim(text="c", normalized="n", importance=50)] * c["claims"])
        s.write_chunk(lid, i + 1, [Segment(t0=i, t1=i + 1, text="x")], asr="mw", result=res)
    s.close()
    return tmp_path / "copilot.sqlite"


def m1(check, tmp_path, **kw):
    from pathlib import Path
    return check(m1_store(tmp_path, **kw), fixture=Path(M1_FIXTURE))


def test_m1_fixture_replay_passes_with_enough_rows(tmp_path):
    r = m1(gate.check_fixture_replay, tmp_path)
    assert r.status == "PASS" and "58 segments" in r.detail


@pytest.mark.parametrize("counts", [{"segments": 9, "items": 28, "claims": 23},
                                    {"segments": 58, "items": 4, "claims": 23},
                                    {"segments": 58, "items": 28, "claims": 0}])
def test_m1_fixture_replay_fails_below_a_threshold(tmp_path, counts):
    assert m1(gate.check_fixture_replay, tmp_path, runs=1, counts=[counts]).status == "FAIL"


def test_m1_fixture_replay_fails_without_a_run(tmp_path):
    r = m1(gate.check_fixture_replay, tmp_path, runs=0, counts=[])
    assert r.status == "FAIL" and "replay" in r.detail


def test_m1_fixture_replay_ignores_a_broken_last_run(tmp_path):
    r = m1(gate.check_fixture_replay, tmp_path, runs=1, source_error="RuntimeError: ffmpeg")
    assert r.status == "FAIL"


def test_m1_rerun_identical_counts_passes(tmp_path):
    assert m1(gate.check_rerun_upsert, tmp_path).status == "PASS"


def test_m1_rerun_passes_when_counts_differ_between_runs_and_says_so(tmp_path):
    # D-M1-5: the ASR provider is not deterministic, so two replays may extract different rows
    counts = [{"segments": 58, "items": 28, "claims": 23}, {"segments": 51, "items": 27, "claims": 18}]
    r = m1(gate.check_rerun_upsert, tmp_path, counts=counts)
    assert r.status == "PASS" and "segments 58→51" in r.detail


def test_m1_rerun_fails_when_a_row_survives_from_an_older_run(tmp_path):
    import sqlite3
    from pathlib import Path

    from lecture_copilot.store.db import new_id
    path = m1_store(tmp_path)
    con = sqlite3.connect(path)
    con.execute("update claims set id = ? where id = (select min(id) from claims)", (new_id(now_ms=500),))
    con.commit()
    r = gate.check_rerun_upsert(path, fixture=Path(M1_FIXTURE))
    assert r.status == "FAIL" and "older" in r.detail and "claims" in r.detail


def test_m1_rerun_needs_two_runs(tmp_path):
    assert m1(gate.check_rerun_upsert, tmp_path, runs=1).status == "FAIL"


def test_m1_rerun_fails_when_the_table_disagrees_with_the_run(tmp_path):
    counts = [{"segments": 58, "items": 28, "claims": 23}] * 2
    path = m1_store(tmp_path, counts=counts)
    import sqlite3
    con = sqlite3.connect(path)
    con.execute("insert into segments (id, lecture_id, chunk_id, t0, t1, text) "
                "select 'DUP', lecture_id, 99, 0, 1, 'dup' from segments limit 1")
    con.commit()
    from pathlib import Path
    assert gate.check_rerun_upsert(path, fixture=Path(M1_FIXTURE)).status == "FAIL"


def test_m1_budget_passes_under_30_s(tmp_path):
    r = m1(gate.check_chunk_budget, tmp_path)
    assert r.status == "PASS" and "p95" in r.detail


def test_m1_budget_fails_when_one_chunk_is_over(tmp_path):
    r = m1(gate.check_chunk_budget, tmp_path, totals=(14.6, 31.0))
    assert r.status == "FAIL" and "0002" in r.detail


@pytest.mark.parametrize("stage_p95", [(8.5, 12.9), (5.2, 15.5)])
def test_m1_budget_fails_when_a_stage_p95_is_over_its_budget(tmp_path, stage_p95):
    assert m1(gate.check_chunk_budget, tmp_path, stage_p95=stage_p95).status == "FAIL"


def test_m1_memory_passes_with_normal_pressure_and_all_three_resident(tmp_path):
    assert m1(gate.check_peak_memory, tmp_path).status == "PASS"


def test_m1_memory_passes_on_pressure_warning_and_says_so(tmp_path):
    # D-M1-6: "warning" is the OS compressing memory; it fits as long as swap does not grow and the budget holds
    r = m1(gate.check_peak_memory, tmp_path, memory=GOOD_MEMORY | {"max_pressure": 2, "swap_growth_mb": 692.0})
    assert r.status == "PASS" and "warning" in r.detail


def test_m1_memory_fails_on_pressure_critical(tmp_path):
    r = m1(gate.check_peak_memory, tmp_path, memory=GOOD_MEMORY | {"max_pressure": 4})
    assert r.status == "FAIL" and "critical" in r.detail


def test_m1_memory_fails_when_swap_grows(tmp_path):
    r = m1(gate.check_peak_memory, tmp_path, memory=GOOD_MEMORY | {"max_pressure": 2, "swap_growth_mb": 7743.0})
    assert r.status == "FAIL" and "7743" in r.detail


def test_m1_memory_fails_when_a_model_was_not_resident(tmp_path):
    r = m1(gate.check_peak_memory, tmp_path, memory=GOOD_MEMORY | {"models_at_peak_gb": {"gemma3:12b": 8.3}})
    assert r.status == "FAIL" and "bge-m3" in r.detail


def test_m1_memory_fails_when_not_measured(tmp_path):
    assert m1(gate.check_peak_memory, tmp_path, memory={"samples": 0, "errors": 3}).status == "FAIL"


def test_m1_tests_check_runs_the_commands():
    assert gate.check_tests(cmds=(("true",), ("true",))).status == "PASS"
    r = gate.check_tests(cmds=(("true",), ("false",)))
    assert r.status == "FAIL" and "false" in r.detail


def asr_error_check(tmp_path, asr, replies):
    import json

    from lecture_copilot.audio.sources import AudioChunk
    fake = FakeOllama([json.dumps({"chunk_summary": "s", "concepts": [], "claims": [], "items": []})] * replies)
    chunks = [AudioChunk("", 1, tmp_path / "chunk_0001.wav", 0.0, 1.0),
              AudioChunk("", 2, tmp_path / "chunk_0002.wav", 1.0, 31.0)]
    return gate.check_asr_error(work=tmp_path, asr=asr, client=fake.async_client(), chunks=chunks)


def test_m1_asr_error_passes_when_the_corrupt_chunk_fails_and_the_next_runs(tmp_path):
    from lecture_copilot.asr.base import ASRError
    from tests.stubs import FakeASR
    r = asr_error_check(tmp_path, FakeASR(script={1: ASRError("mw exit 1: bad wav")}), replies=1)
    assert r.status == "PASS" and "bad wav" in r.detail


def test_m1_asr_error_fails_when_the_corrupt_chunk_is_not_marked(tmp_path):
    from tests.stubs import FakeASR
    assert asr_error_check(tmp_path, FakeASR(), replies=2).status == "FAIL"


def test_m1_asr_error_fails_when_the_run_stops(tmp_path):
    from lecture_copilot.asr.base import ASRError
    from tests.stubs import FakeASR
    asr = FakeASR(script={1: ASRError("bad wav"), 2: ASRError("also bad")})
    assert asr_error_check(tmp_path, asr, replies=0).status == "FAIL"


def test_m1_checks():
    assert [name for name, _ in gate.CHECKS[1]] == ["fixture_replay", "rerun_upsert", "chunk_budget", "asr_error",
                                                    "peak_memory", "prompt_cache", "tests", "secret_scan"]


def test_m1_prompt_cache_off_passes():
    r = gate.check_prompt_cache(server_env=lambda: "ollama serve TERM=xterm LLAMA_ARG_CACHE_RAM=0 HOME=/Users/x")
    assert r.status == "PASS"


@pytest.mark.parametrize("env", ["ollama serve TERM=xterm HOME=/Users/x", "ollama serve LLAMA_ARG_CACHE_RAM=8192"])
def test_m1_prompt_cache_on_fails_with_the_fix(env):
    r = gate.check_prompt_cache(server_env=lambda: env)
    assert r.status == "FAIL" and "scripts/ollama_serve.sh" in r.detail


def test_m1_prompt_cache_without_a_server_fails():
    r = gate.check_prompt_cache(server_env=lambda: None)
    assert r.status == "FAIL" and "not running" in r.detail


def test_m1_asr_error_detail_is_terminal_safe(tmp_path):
    from lecture_copilot.asr.base import ASRError
    from tests.stubs import FakeASR
    err = ASRError("mw exit 1: Transcribing chunk_0001.wav...\nError: לא ניתן היה להשלים את הפעולה.")
    r = asr_error_check(tmp_path, FakeASR(script={1: err}), replies=1)
    assert "\n" not in r.detail and not any("֐" <= ch <= "׿" for ch in r.detail)


# ---------- M2 ----------

def m2_store(tmp_path, *, source="file", minutes=83, total_s=71.0, degraded=(), md=None, sink_ok=True, root=None,
             files=("digest.md", "digest.html", "transcript.txt", "claims.json"), html=None):
    from lecture_copilot.output.digest import SECTIONS
    from lecture_copilot.store.db import Store
    root = root or tmp_path / "courses"
    s = Store(tmp_path / "copilot.sqlite")
    course = s.upsert_course("AI Developers — Python", language="he")
    lid = s.upsert_lecture(course, audio_path=f"/x/{source}", source=source, title="t", date="2026-06-19",
                           fact_check=True)
    s.end_lecture(lid)
    good = "# h\n\n" + "\n\n".join(f"## {name}\n\nאין." for name in SECTIONS) + "\n"
    s.save_digest(lid, bullets=[], digest_md=good if md is None else md)
    s.log("digest", lecture_id=lid, input_ref="R#digest", ms=total_s * 1000, output={
        "status": "degraded" if degraded else "ok", "degraded": list(degraded), "minutes": minutes,
        "total_s": total_s, "blocks": 2})
    folder = root / "AI Developers — Python" / "2026-06-19_t"
    folder.mkdir(parents=True)
    for name in files:
        (folder / name).write_text(good if name.endswith(".md") else "x", encoding="utf-8")
    if "digest.html" in files:
        heads = "".join(f"<h2>{name}</h2>" for name in SECTIONS)
        (folder / "digest.html").write_text(html or f'<html lang="he" dir="rtl"><body>{heads}</body></html>',
                                            encoding="utf-8")
    (root / "AI Developers — Python" / "index.md").write_text("# c\n", encoding="utf-8")
    s.log("sink", lecture_id=lid, input_ref=lid, output={"sink": "FolderSink", "status": "ok", "files": len(files),
                                                        "folder": str(folder)} if sink_ok else
          {"sink": "FolderSink", "status": "failed", "error": "OSError: disk full"})
    s.close()
    return tmp_path / "copilot.sqlite", root


def test_m2_sections_pass_when_every_digest_has_the_nine_in_order(tmp_path):
    db, _ = m2_store(tmp_path)
    r = gate.check_digest_sections(db)
    assert r.status == "PASS" and "9 sections" in r.detail


def test_m2_sections_fail_on_a_missing_or_moved_section(tmp_path):
    from lecture_copilot.output.digest import SECTIONS
    swapped = [SECTIONS[1], SECTIONS[0], *SECTIONS[2:]]
    db, _ = m2_store(tmp_path, md="# h\n\n" + "\n\n".join(f"## {n}\n\nאין." for n in swapped) + "\n")
    assert gate.check_digest_sections(db).status == "FAIL"
    db2, _ = m2_store(tmp_path / "b", md="# h\n\n" + "\n\n".join(f"## {n}\n\nx" for n in SECTIONS[:8]) + "\n")
    assert gate.check_digest_sections(db2).status == "FAIL"


def test_m2_sections_fail_without_a_digest(tmp_path):
    from lecture_copilot.store.db import Store
    Store(tmp_path / "e.sqlite").close()
    assert gate.check_digest_sections(tmp_path / "e.sqlite").status == "FAIL"


def test_m2_digest_time_passes_for_an_hour_under_two_minutes(tmp_path):
    db, _ = m2_store(tmp_path, minutes=83, total_s=71.0)
    r = gate.check_digest_time(db)
    assert r.status == "PASS" and "83 min" in r.detail and "71.0 s" in r.detail


@pytest.mark.parametrize("kw", [dict(total_s=131.0), dict(minutes=10), dict(degraded=("map-2",))])
def test_m2_digest_time_fails_when_slow_short_or_degraded(tmp_path, kw):
    db, _ = m2_store(tmp_path, **kw)
    assert gate.check_digest_time(db).status == "FAIL"


def test_m2_digest_time_judges_lectures_of_one_to_two_hours_and_reports_longer_ones(tmp_path):
    # the spec promises two minutes for a lecture of up to two hours; a four-hour Zoom day is reported, not judged
    from lecture_copilot.store.db import Store
    db, _ = m2_store(tmp_path, minutes=78, total_s=94.3)
    s = Store(db)
    s.log("digest", lecture_id="OTHER", input_ref="R2#digest", output={
        "status": "ok", "degraded": [], "minutes": 251, "total_s": 140.0, "blocks": 6})
    s.close()
    r = gate.check_digest_time(db)
    assert r.status == "PASS" and "78 min" in r.detail and "251 min" in r.detail and "140.0 s" in r.detail


def test_m2_digest_time_fails_when_any_lecture_in_range_is_slow(tmp_path):
    from lecture_copilot.store.db import Store
    db, _ = m2_store(tmp_path, minutes=78, total_s=94.3)
    s = Store(db)
    s.log("digest", lecture_id="OTHER", input_ref="R2#digest", output={
        "status": "ok", "degraded": [], "minutes": 110, "total_s": 125.0, "blocks": 4})
    s.close()
    assert gate.check_digest_time(db).status == "FAIL"


def test_m2_digest_time_uses_the_latest_digest_of_each_lecture(tmp_path):
    from lecture_copilot.store.db import Store
    db, _ = m2_store(tmp_path, minutes=78, total_s=130.0)
    s = Store(db)
    lid = s.con.execute("select id from lectures").fetchone()[0]
    s.log("digest", lecture_id=lid, input_ref="R2#digest", output={
        "status": "ok", "degraded": [], "minutes": 78, "total_s": 94.3, "blocks": 3})
    s.close()
    assert gate.check_digest_time(db).status == "PASS"


def test_m2_vtt_replay_passes_for_a_transcript_lecture_with_nine_sections(tmp_path):
    db, _ = m2_store(tmp_path, source="transcript")
    assert gate.check_vtt_replay(db).status == "PASS"


def test_m2_vtt_replay_fails_when_only_audio_was_replayed(tmp_path):
    db, _ = m2_store(tmp_path, source="file")
    r = gate.check_vtt_replay(db)
    assert r.status == "FAIL" and ".vtt" in r.detail


def test_m2_course_folder_passes_and_says_where_it_is(tmp_path):
    db, root = m2_store(tmp_path)
    r = gate.check_course_folder(db, courses_root=root)
    assert r.status == "PASS" and "4 files" in r.detail and "local folder" in r.detail


def test_m2_course_folder_inside_google_drive_is_named_as_such(tmp_path):
    db, root = m2_store(tmp_path, root=tmp_path / "CloudStorage" / "GoogleDrive-x" / "My Drive" / "LC")
    assert "Google Drive" in gate.check_course_folder(db, courses_root=root).detail


@pytest.mark.parametrize("kw", [dict(sink_ok=False), dict(files=("digest.md", "digest.html"))])
def test_m2_course_folder_fails_on_a_failed_sink_or_missing_files(tmp_path, kw):
    db, root = m2_store(tmp_path, **kw)
    assert gate.check_course_folder(db, courses_root=root).status == "FAIL"


def test_m2_course_folder_fails_when_the_folder_is_outside_the_root(tmp_path):
    db, _ = m2_store(tmp_path)
    assert gate.check_course_folder(db, courses_root=tmp_path / "elsewhere").status == "FAIL"


def test_m2_html_passes_with_rtl_markup_and_both_screenshots(tmp_path):
    db, _ = m2_store(tmp_path)
    img = tmp_path / "img"
    img.mkdir()
    for name in ("m2_digest_desktop.jpg", "m2_digest_phone.jpg"):
        (img / name).write_bytes(b"\xff\xd8" + b"0" * 5000)
    assert gate.check_html_rtl(db, img_dir=img).status == "PASS"


def test_m2_html_fails_without_a_screenshot(tmp_path):
    db, _ = m2_store(tmp_path)
    (tmp_path / "img").mkdir()
    r = gate.check_html_rtl(db, img_dir=tmp_path / "img")
    assert r.status == "FAIL" and "screenshot" in r.detail


def test_m2_html_fails_when_the_page_is_not_rtl(tmp_path):
    db, _ = m2_store(tmp_path, html='<html lang="en"><body><h2>x</h2></body></html>')
    img = tmp_path / "img"
    img.mkdir()
    for name in ("m2_digest_desktop.jpg", "m2_digest_phone.jpg"):
        (img / name).write_bytes(b"\xff\xd8" + b"0" * 5000)
    assert gate.check_html_rtl(db, img_dir=img).status == "FAIL"


def test_m2_checks():
    assert [name for name, _ in gate.CHECKS[2]] == ["digest_sections", "digest_time", "vtt_replay", "course_folder",
                                                    "html_rtl", "prompt_cache", "tests", "secret_scan"]


# ---------- M3 ----------

def m3_store(tmp_path, *, with_digest=True, continuation=True, first_seen_share=1.0, memory_p95=1.2):
    """7/6 and 9/6 as transcript lectures; 9/6's concepts partly first seen in 7/6."""
    from lecture_copilot.agents.schemas import Concept, ExtractResult
    from lecture_copilot.asr.base import Segment
    from lecture_copilot.store.db import Store
    from tests.stubs import fake_embedding
    s = Store(tmp_path / "copilot.sqlite")
    course = s.upsert_course("AI Developers — Python", language="he")
    a = s.upsert_lecture(course, audio_path="/x/2026-06-07.vtt", source="transcript", title="7-6", date="2026-06-07",
                         fact_check=True)
    b = s.upsert_lecture(course, audio_path="/x/2026-06-09.vtt", source="transcript", title="9-6", date="2026-06-09",
                         fact_check=True)
    terms = [f"term{i}" for i in range(10)]
    for lid, idx in ((a, 1), (b, 1)):
        res = ExtractResult(chunk_summary="s", items=[], claims=[], concepts=[
            Concept(term=t, explanation=f"explanation of {t} number {i}", canonical_key=t)
            for i, t in enumerate(terms)])
        s.write_chunk(lid, idx, [Segment(t0=0, t1=40, text="x")], asr="transcript", result=res)
        for r in s.items(lid, kind="concept"):
            s.set_embedding("items", r["id"], fake_embedding(f"{r['text']} {r['explanation']}"))
    shared = int(round(first_seen_share * len(terms)))
    for r in s.items(b, kind="concept")[:shared]:
        s.set_first_seen(r["id"], a)
    s.end_lecture(a)
    s.end_lecture(b)
    s.save_digest(a, bullets=["x"], digest_md="# a")
    if with_digest:
        md = "# b\n\n## המשך מ-7-6\n\n- **מה חדש:** y\n" if continuation else \
            "# b\n\n## המשך מ-7-6\n\nהמודל לא השווה להרצאה הקודמת הפעם.\n"
        s.save_digest(b, bullets=["x"], digest_md=md)
        s.set_continues(b, a)
    s.log("run", lecture_id=b, input_ref="R", output={"chunks": 1, "status": {"ok": 1}, "counts": {},
                                                     "audio_s": 40, "timing": {"memory_s": {"p95": memory_p95}}})
    s.close()
    return tmp_path / "copilot.sqlite"


def benchmark(tmp_path, terms):
    import json
    p = tmp_path / "benchmark.json"
    p.write_text(json.dumps({"shared_concepts": [{"term": t, "first": "2026-06-07", "again": "2026-06-09"}
                                                 for t in terms]}), encoding="utf-8")
    return p


def test_m3_shared_concepts_pass_at_80_percent(tmp_path):
    db = m3_store(tmp_path, first_seen_share=0.8)
    r = gate.check_shared_concepts(db, benchmark_path=benchmark(tmp_path, [f"term{i}" for i in range(10)]))
    assert r.status == "PASS" and "8/10" in r.detail


def test_m3_shared_concepts_fail_below_80_percent(tmp_path):
    db = m3_store(tmp_path, first_seen_share=0.7)
    bench = benchmark(tmp_path, [f"term{i}" for i in range(10)])
    assert gate.check_shared_concepts(db, benchmark_path=bench).status == "FAIL"


def test_m3_shared_concepts_skip_without_dors_labels(tmp_path):
    db = m3_store(tmp_path)
    r = gate.check_shared_concepts(db, benchmark_path=tmp_path / "missing.json")
    assert r.status == "SKIP" and "benchmark.json" in r.detail


def test_m3_search_top1_over_sampled_terms(tmp_path):
    db = m3_store(tmp_path)
    r = gate.check_search_top1(db)
    assert r.status == "PASS" and "10/10" in r.detail


def test_m3_search_top1_fails_when_the_index_is_empty(tmp_path):
    import sqlite3
    db = m3_store(tmp_path)
    con = sqlite3.connect(db)
    con.execute("delete from items_fts")
    con.execute("update items set embedding = null")
    con.commit()
    con.close()
    assert gate.check_search_top1(db).status == "FAIL"


def test_m3_continuation_present(tmp_path):
    assert gate.check_continuation(m3_store(tmp_path)).status == "PASS"


@pytest.mark.parametrize("kw", [dict(continuation=False), dict(with_digest=False)])
def test_m3_continuation_fails_when_empty_or_missing(tmp_path, kw):
    assert gate.check_continuation(m3_store(tmp_path, **kw)).status == "FAIL"


def test_m3_memory_budget(tmp_path):
    assert gate.check_memory_budget(m3_store(tmp_path, memory_p95=2.9)).status == "PASS"
    assert gate.check_memory_budget(m3_store(tmp_path / "b", memory_p95=3.5)).status == "FAIL"


def test_m3_numpy_fallback_agrees_with_sqlite_vec(tmp_path):
    r = gate.check_numpy_fallback(m3_store(tmp_path), tests=(("true",),))
    assert r.status == "PASS" and "numpy" in r.detail


def test_m3_contradiction_check_reads_the_run(tmp_path):
    from tests.stubs import FakeOllama
    a = json.dumps({"chunk_summary": "s", "concepts": [], "items": [],
                    "claims": [{"text": "תקופת ההחזר המקובלת היא 18 חודשים", "normalized": "payback 18 months",
                                "importance": 70}]}, ensure_ascii=False)
    b = json.dumps({"chunk_summary": "s", "concepts": [], "items": [],
                    "claims": [{"text": "תקופת ההחזר המקובלת היא 12 חודשים", "normalized": "payback 12 months",
                                "importance": 70, "contradicts": "תקופת ההחזר המקובלת היא 18 חודשים"}]},
                   ensure_ascii=False)
    r = gate.check_contradiction(work=tmp_path, client=FakeOllama([a, b]).async_client())
    assert r.status == "PASS" and "70 → 90" in r.detail


def test_m3_contradiction_check_fails_when_the_model_misses_it(tmp_path):
    from tests.stubs import FakeOllama
    a = json.dumps({"chunk_summary": "s", "concepts": [], "items": [],
                    "claims": [{"text": "18 חודשים", "normalized": "18", "importance": 70}]}, ensure_ascii=False)
    b = json.dumps({"chunk_summary": "s", "concepts": [], "items": [],
                    "claims": [{"text": "12 חודשים", "normalized": "12", "importance": 70}]}, ensure_ascii=False)
    assert gate.check_contradiction(work=tmp_path, client=FakeOllama([a, b]).async_client()).status == "FAIL"


def test_m3_checks():
    assert [name for name, _ in gate.CHECKS[3]] == ["shared_concepts", "contradiction", "search_top1", "continuation",
                                                    "memory_budget", "numpy_fallback", "prompt_cache", "tests",
                                                    "secret_scan"]


# ---------- M4 ----------

def results(tmp_path, accuracy=0.83, p=None, cache=True):
    tmp_path.mkdir(parents=True, exist_ok=True)
    r = {"labels_by": "Claude, draft",
         "verdicts": {"n": 42, "accuracy": accuracy, "injected": 10, "injected_caught": 8},
         "precision_at_5": p or {"2026-06-07": {"k": 5, "precision": 1.0, "material_labelled": 9},
                                 "2026-06-09": {"k": 2, "precision": 0.5, "material_labelled": 2}},
         "cache_hit": cache, "cost_usd": 0.0136, "unchecked": 0}
    path = tmp_path / "results.json"
    path.write_text(json.dumps(r), encoding="utf-8")
    return path


def test_m4_accuracy_passes_at_80_percent_and_names_the_labels(tmp_path):
    r = gate.check_verdict_accuracy(results(tmp_path))
    assert r.status == "PASS" and "83%" in r.detail and "Claude, draft" in r.detail and "8/10" in r.detail


def test_m4_accuracy_fails_below_80_percent_or_without_results(tmp_path):
    assert gate.check_verdict_accuracy(results(tmp_path, accuracy=0.79)).status == "FAIL"
    assert gate.check_verdict_accuracy(tmp_path / "missing.json").status == "FAIL"


def test_m4_precision_judges_lectures_with_five_material_claims_and_reports_the_rest(tmp_path):
    r = gate.check_precision(results(tmp_path))
    assert r.status == "PASS" and "100%" in r.detail and "k=2" in r.detail


def test_m4_precision_fails_when_a_judged_lecture_is_low_or_none_is_judged(tmp_path):
    low = {"2026-06-07": {"k": 5, "precision": 0.6, "material_labelled": 9}}
    assert gate.check_precision(results(tmp_path, p=low)).status == "FAIL"
    none = {"2026-06-09": {"k": 2, "precision": 1.0, "material_labelled": 2}}
    assert gate.check_precision(results(tmp_path / "b", p=none)).status == "FAIL"


def test_m4_cache_hit(tmp_path):
    assert gate.check_cache_hit(results(tmp_path)).status == "PASS"
    assert gate.check_cache_hit(results(tmp_path / "b", cache=False)).status == "FAIL"


def test_m4_cost_per_lecture_from_the_net_rows(tmp_path):
    from lecture_copilot.store.db import Store
    s = Store(tmp_path / "copilot.sqlite")
    course = s.upsert_course("c", language="he")
    lid = s.upsert_lecture(course, audio_path="/a", source="transcript", title="t", date="d", fact_check=True)
    for cost in (0.01, 0.0122):
        s.log("net", lecture_id=lid, input_ref="x", output={"host": "h", "status": "ok"}, cost_usd=cost)
    s.close()
    r = gate.check_cost(tmp_path / "copilot.sqlite")
    assert r.status == "PASS" and "0.0222" in r.detail
    s = Store(tmp_path / "copilot.sqlite")
    s.log("net", lecture_id=lid, input_ref="x", output={"host": "h", "status": "ok"}, cost_usd=0.3)
    s.close()
    assert gate.check_cost(tmp_path / "copilot.sqlite").status == "FAIL"


def test_m4_ci_never_calls_gemini():
    assert gate.check_ci_offline().status == "PASS"


def test_m4_outage_check_passes_when_the_claim_goes_unchecked_then_verified(tmp_path):
    from tests.stubs import FakeGemini
    ok = {"verdict": "correct", "confidence": 0.9, "explanation": "נכון.", "sources": ["https://a"]}
    extraction = json.dumps({"chunk_summary": "s", "concepts": [], "items": [],
                             "claims": [{"text": "18 חודשים", "normalized": "18 months", "importance": 80}]},
                            ensure_ascii=False)
    r = gate.check_outage(work=tmp_path, client=FakeOllama([extraction]).async_client(), gemini=FakeGemini([ok, ok]))
    assert r.status == "PASS" and "unchecked" in r.detail and "verified" in r.detail


def test_m4_outage_check_fails_when_nothing_was_retried(tmp_path):
    from tests.stubs import FakeGemini
    extraction = json.dumps({"chunk_summary": "s", "concepts": [], "items": [],
                             "claims": [{"text": "x", "normalized": "x", "importance": 40}]}, ensure_ascii=False)
    r = gate.check_outage(work=tmp_path, client=FakeOllama([extraction]).async_client(), gemini=FakeGemini())
    assert r.status == "FAIL"


def test_m4_checks():
    assert [name for name, _ in gate.CHECKS[4]] == ["verdict_accuracy", "precision_at_5", "outage", "cost_per_lecture",
                                                    "cache_hit", "ci_offline", "prompt_cache", "tests", "secret_scan"]


# ---------- M5 ----------

def m5_store(tmp_path, *, minutes=21, via="web", pace="realtime", digest_s=40.0, failed=0, resumed=True):
    from lecture_copilot.store.db import Store
    s = Store(tmp_path / "copilot.sqlite")
    course = s.upsert_course("AI Developers — Python", language="he")
    lid = s.upsert_lecture(course, audio_path="/x/live.m4a", source="file", title="live", date="2026-10-02",
                           fact_check=True)
    s.end_lecture(lid)
    s.save_digest(lid, bullets=["x"], digest_md="# d")
    status = {"ok": 28 - failed, "empty": 0}
    if failed:
        status["failed"] = failed
    s.log("run", lecture_id=lid, input_ref="R", output={"chunks": 28, "status": status, "counts": {},
                                                       "audio_s": minutes * 60, "via": via, "source_kind": "file",
                                                       "pace": pace, "timing": {}})
    s.log("digest", lecture_id=lid, input_ref="R#digest", output={"status": "ok", "degraded": [], "minutes": minutes,
                                                                 "total_s": digest_s, "blocks": 1})
    if resumed:
        other = s.upsert_lecture(course, audio_path="/x/crash.m4a", source="file", title="crash", date="2026-10-02",
                                 fact_check=True)
        s.end_lecture(other)
        s.log("run", lecture_id=other, input_ref="resume", output={"status": "resumed", "chunks": 3, "via": "web"})
        s.save_digest(other, bullets=["x"], digest_md="# d")
    s.close()
    return tmp_path / "copilot.sqlite"


def test_m5_live_simulation_passes_for_a_real_time_run_of_20_minutes(tmp_path):
    r = gate.check_live_simulation(m5_store(tmp_path))
    assert r.status == "PASS" and "21 min" in r.detail and "40.0 s" in r.detail


@pytest.mark.parametrize("kw", [dict(minutes=10), dict(pace="fast"), dict(via="cli"), dict(digest_s=130.0),
                                dict(failed=1)])
def test_m5_live_simulation_fails_when_short_fast_not_via_the_page_slow_or_failed(tmp_path, kw):
    assert gate.check_live_simulation(m5_store(tmp_path, **kw)).status == "FAIL"


def test_m5_crash_resume(tmp_path):
    assert gate.check_crash_resume(m5_store(tmp_path)).status == "PASS"
    assert gate.check_crash_resume(m5_store(tmp_path / "b", resumed=False)).status == "FAIL"


def test_m5_mic_level_reads_dors_check_or_skips(tmp_path):
    p = tmp_path / "mic_check.json"
    assert gate.check_mic_level(p).status == "SKIP"
    p.write_text(json.dumps({"peak_db": -22.5, "seconds_to_minus40": 1.2, "seconds": 10}), encoding="utf-8")
    assert gate.check_mic_level(p).status == "PASS"
    p.write_text(json.dumps({"peak_db": -55.0, "seconds_to_minus40": None, "seconds": 10}), encoding="utf-8")
    assert gate.check_mic_level(p).status == "FAIL"


def test_m5_course_html_lists_the_glossary_across_two_lectures(tmp_path):
    course = tmp_path / "courses" / "AI Developers — Python"
    course.mkdir(parents=True)
    (course / "course.html").write_text(
        '<html lang="he" dir="rtl"><section id="lectures" class="lectures"><a>W01</a><a>W02</a></section>'
        '<div data-list="glossary"><div class="row">a</div><div class="row">b</div></div></html>', encoding="utf-8")
    r = gate.check_course_html(courses_root=tmp_path / "courses")
    assert r.status == "PASS" and "2 lectures" in r.detail
    (course / "course.html").write_text('<html lang="he" dir="rtl"><section id="lectures"><a>W01</a></section>'
                                        '<div data-list="glossary"></div></html>', encoding="utf-8")
    assert gate.check_course_html(courses_root=tmp_path / "courses").status == "FAIL"


def test_m5_page_rtl_needs_the_markup_and_three_screenshots(tmp_path):
    img = tmp_path / "img"
    img.mkdir()
    assert gate.check_page_rtl(img_dir=img).status == "FAIL"
    for name in ("m5_before.jpg", "m5_during.jpg", "m5_after.jpg"):
        (img / name).write_bytes(b"\xff\xd8" + b"0" * 5000)
    assert gate.check_page_rtl(img_dir=img).status == "PASS"


def test_m5_checks():
    assert [name for name, _ in gate.CHECKS[5]] == ["live_simulation", "crash_resume", "mic_level", "course_html",
                                                    "page_rtl", "prompt_cache", "tests", "secret_scan"]


def test_mic_report_from_levels():
    from lecture_copilot.web.launcher import mic_report
    r = mic_report([-70.0, -65.0, -38.0, -30.0, -42.0], block_s=0.25)
    assert r == {"peak_db": -30.0, "seconds_to_minus40": 0.75, "seconds": 1.25, "readable": True}
    assert mic_report([-70.0, -66.0], block_s=0.25)["seconds_to_minus40"] is None
