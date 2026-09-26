import subprocess

import httpx

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
