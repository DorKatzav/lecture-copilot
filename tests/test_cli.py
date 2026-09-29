import asyncio
import json

import pytest

from lecture_copilot import cli
from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.store.db import Store
from tests.stubs import FakeASR, FakeOllama, ListSource

REPLY = json.dumps({"chunk_summary": "סיכום", "concepts": [], "items": [],
                    "claims": [{"text": "t", "normalized": "n", "importance": 50}]})


def list_source(n):
    def factory(lecture_id, file, pace):
        return ListSource([AudioChunk(lecture_id, i, file.parent / f"chunk_{i:04d}.wav", (i - 1) * 45.0, i * 45.0)
                           for i in range(1, n + 1)])
    return factory


def replay(tmp_path, n=2, replies=None, **kw):
    fake = FakeOllama(replies or [REPLY] * n)
    lines = []
    args = dict(file=tmp_path / "tirgul.m4a", course="AI Developers — Python", language="he", title=None,
                date="2026-06-19", pace="fast", db=tmp_path / "copilot.sqlite", fact_check=True)
    args.update(kw)
    summary = asyncio.run(cli.replay(**args, asr=FakeASR(), client=fake.async_client(),
                                     source_factory=list_source(n), runs_dir=tmp_path / "runs", probe=None,
                                     echo=lines.append))
    return summary, lines


def test_replay_writes_the_lecture_and_ends_it(tmp_path):
    summary, _ = replay(tmp_path)
    s = Store(tmp_path / "copilot.sqlite")
    (lec,) = s.con.execute("select * from lectures").fetchall()
    assert lec["status"] == "ended" and lec["source"] == "file" and lec["title"] == "tirgul"
    assert lec["audio_path"] == str((tmp_path / "tirgul.m4a").resolve())
    assert summary["counts"] == {"segments": 2, "items": 0, "claims": 2}
    s.close()


def test_replay_twice_is_one_lecture(tmp_path):
    a, _ = replay(tmp_path)
    b, _ = replay(tmp_path)
    s = Store(tmp_path / "copilot.sqlite")
    assert s.con.execute("select count(*) from lectures").fetchone()[0] == 1 and a["counts"] == b["counts"]
    s.close()


def test_the_run_row_says_what_was_replayed(tmp_path):
    replay(tmp_path)
    s = Store(tmp_path / "copilot.sqlite")
    out = json.loads(s.con.execute("select output_json from decisions where node = 'run'").fetchone()[0])
    assert out["source"] == "tirgul.m4a" and out["pace"] == "fast" and out["language"] == "he"
    s.close()


def test_existing_course_keeps_its_language(tmp_path):
    replay(tmp_path)
    summary, lines = replay(tmp_path, language="en")
    assert any("course language is he" in line for line in lines)


def test_terminal_output_is_english_only(tmp_path):
    _, lines = replay(tmp_path)
    text = "\n".join(lines)
    assert "chunk 0001" in text and "claims 2" in text
    assert not any("֐" <= ch <= "׿" for ch in text)   # CLAUDE.md: Hebrew never goes to the terminal


@pytest.mark.parametrize("name", ["lecture.vtt", "lecture.json", "lecture.srt"])
def test_transcripts_are_refused_until_m2(tmp_path, name, capsys):
    assert cli.main(["replay", str(tmp_path / name), "--course", "X"]) == 2
    assert "M2" in capsys.readouterr().err


def test_missing_file_is_refused(tmp_path, capsys):
    assert cli.main(["replay", str(tmp_path / "nope.m4a"), "--course", "X"]) == 2
    assert "not found" in capsys.readouterr().err


MW_HE_ERROR = "mw exit 1: Transcribing chunk_0001.wav...\nError: לא ניתן היה להשלים את הפעולה."


def test_terminal_text_is_one_line_without_hebrew():
    out = cli.terminal_text(MW_HE_ERROR)
    assert "\n" not in out and not any("֐" <= ch <= "׿" for ch in out)
    assert out.startswith("mw exit 1: Transcribing chunk_0001.wav... Error:") and "[he]" in out


def test_chunk_line_never_prints_a_localized_error():
    line = cli.fmt_chunk({"idx": 2, "t0": 0.0, "t1": 30.0, "status": "asr_failed", "asr_s": 1.0, "extract_s": None,
                          "total_s": 1.0, "segments": 0, "error": MW_HE_ERROR})
    assert not any("֐" <= ch <= "׿" for ch in line) and "\n" not in line
