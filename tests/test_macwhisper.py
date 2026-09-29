import asyncio
import json
import os

import pytest

from lecture_copilot.asr.base import ASRError
from lecture_copilot.asr.macwhisper import MacWhisperASR
from lecture_copilot.config import MW_MODELS
from tests.stubs import fake_mw


def transcribe(tmp_path, mode, language="he", **kw):
    wav = tmp_path / "chunk_0001.wav"
    wav.write_bytes(b"RIFF")
    asr = MacWhisperASR(binary=str(fake_mw(tmp_path, mode)), **kw)
    return asyncio.run(asr.transcribe(wav, language))


def calls(tmp_path):
    return [json.loads(line) for line in (tmp_path / "mw_calls.jsonl").read_text().splitlines()]


def test_segments_with_chunk_relative_seconds(tmp_path):
    segs = transcribe(tmp_path, "ok")
    assert [(s.t0, s.t1, s.text) for s in segs] == [(0.0, 4.2, "שלום לכולם"), (4.2, 9.0, "היום נדבר על CAC")]


def test_command_is_explicit_about_model_language_and_output(tmp_path):
    transcribe(tmp_path, "ok", language="en")
    (argv,) = calls(tmp_path)
    assert argv[:2] == ["transcribe", str(tmp_path / "chunk_0001.wav")]
    assert argv[argv.index("--model") + 1] == MW_MODELS["en"]
    assert argv[argv.index("--language") + 1] == "en"
    assert argv[argv.index("-o") + 1] == str(tmp_path / "chunk_0001.json")
    assert {"--format", "json", "--no-speakers", "--overwrite"} <= set(argv)
    assert "--persist" not in argv  # never MacWhisper's own history


def test_silence_is_no_segments_not_an_error(tmp_path):
    assert transcribe(tmp_path, "silent") == []


def test_non_zero_exit_is_an_asr_error_with_stderr(tmp_path):
    with pytest.raises(ASRError, match="could not decode audio"):
        transcribe(tmp_path, "exit1")


def test_no_output_file_is_an_asr_error(tmp_path):
    with pytest.raises(ASRError, match="no output"):
        transcribe(tmp_path, "no_output")


def test_unparseable_output_is_an_asr_error(tmp_path):
    with pytest.raises(ASRError, match="output"):
        transcribe(tmp_path, "garbage")


def test_a_hang_is_killed_and_is_an_asr_error(tmp_path):
    with pytest.raises(ASRError, match="timed out"):
        transcribe(tmp_path, "hang", timeout_s=1.0)
    pid = int((tmp_path / "mw_calls.jsonl.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_missing_binary_is_an_asr_error(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(ASRError, match="not found"):
        asyncio.run(MacWhisperASR(binary=str(tmp_path / "nope")).transcribe(wav, "he"))


def test_default_timeout_is_three_asr_budgets():
    assert MacWhisperASR().timeout_s == 24
