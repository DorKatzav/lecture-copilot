import asyncio
import json

import pytest

from lecture_copilot import cli
from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.output.sinks import FolderSink
from lecture_copilot.store.db import Store
from tests.stubs import FakeASR, FakeOllama, ListSource

REPLY = json.dumps({"chunk_summary": "סיכום", "concepts": [], "items": [],
                    "claims": [{"text": "t", "normalized": "n", "importance": 50}]})


def list_source(n):
    def factory(lecture_id, file, pace):
        return ListSource([AudioChunk(lecture_id, i, file.parent / f"chunk_{i:04d}.wav", (i - 1) * 45.0, i * 45.0)
                           for i in range(1, n + 1)])
    return factory


SECTION = json.dumps({"paragraphs": ["פסקה על git."]}, ensure_ascii=False)
EXEC = json.dumps({"exec_summary": [f"נקודה {i}" for i in range(5)], "continuation": None}, ensure_ascii=False)


def replay(tmp_path, n=2, replies=None, sink="folder", gemini="fake", **kw):
    from tests.stubs import FakeGemini
    fake = FakeOllama(replies or [REPLY] * n + [SECTION, EXEC])
    lines = []
    args = dict(file=tmp_path / "tirgul.m4a", course="AI Developers — Python", language="he", title=None,
                date="2026-06-19", pace="fast", db=tmp_path / "copilot.sqlite", fact_check=True)
    args.update(kw)
    sink = FolderSink(tmp_path / "courses") if sink == "folder" else sink
    gemini = FakeGemini() if gemini == "fake" else gemini
    summary = asyncio.run(cli.replay(**args, asr=FakeASR(), client=fake.async_client(),
                                     source_factory=list_source(n), runs_dir=tmp_path / "runs", probe=None,
                                     sink=sink, gemini=gemini, echo=lines.append))
    return summary, lines


def test_replay_writes_the_lecture_and_ends_it(tmp_path):
    summary, _ = replay(tmp_path)
    s = Store(tmp_path / "copilot.sqlite")
    (lec,) = s.con.execute("select * from lectures").fetchall()
    assert lec["status"] == "digested" and lec["source"] == "file" and lec["title"] == "tirgul"
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


@pytest.mark.parametrize("name", ["lecture.srt", "lecture.txt"])
def test_unsupported_transcript_formats_are_refused(tmp_path, name, capsys):
    (tmp_path / name).write_text("x")
    assert cli.main(["replay", str(tmp_path / name), "--course", "X"]) == 2
    assert ".vtt" in capsys.readouterr().err


# ---------- M2: the Digest at the end of a replay ----------

def test_replay_ends_with_a_digest_in_the_course_folder(tmp_path):
    summary, lines = replay(tmp_path, week=3)
    folder = tmp_path / "courses" / "AI Developers — Python" / "W03_2026-06-19_tirgul"
    assert sorted(p.name for p in folder.iterdir()) == ["claims.json", "digest.html", "digest.md", "meta.json",
                                                        "transcript.txt"]
    assert summary["digest"]["folder"] == str(folder) and summary["digest"]["degraded"] == []
    assert any(line.startswith("digest:") and "9 sections" in line for line in lines)


def test_the_sink_is_logged(tmp_path):
    replay(tmp_path)
    s = Store(tmp_path / "copilot.sqlite")
    out = json.loads(s.con.execute("select output_json from decisions where node = 'sink'").fetchone()[0])
    assert out["status"] == "ok" and out["sink"] == "FolderSink" and out["files"] == 5
    s.close()


def test_a_failing_sink_keeps_the_digest_in_the_database(tmp_path):
    class Broken:
        def write_lecture(self, doc):
            raise OSError("disk full")

    summary, lines = replay(tmp_path, sink=Broken())
    s = Store(tmp_path / "copilot.sqlite")
    assert s.con.execute("select digest_md from lecture_summaries").fetchone()[0].startswith("# AI Developers")
    out = json.loads(s.con.execute("select output_json from decisions where node = 'sink'").fetchone()[0])
    assert out["status"] == "failed" and "disk full" in out["error"] and summary["digest"]["folder"] is None
    s.close()


def test_transcript_replay_is_a_transcript_lecture(tmp_path):
    replay(tmp_path, file=tmp_path / "lecture.cc.vtt", source="transcript")
    s = Store(tmp_path / "copilot.sqlite")
    assert s.con.execute("select source from lectures").fetchone()[0] == "transcript"
    s.close()


def test_digest_lines_in_the_terminal_have_no_hebrew(tmp_path):
    _, lines = replay(tmp_path, course="יזמות וחדשנות", title="מודלים עסקיים")
    assert not any("\u0590" <= ch <= "\u05ff" for ch in "\n".join(lines))


def test_digest_command_rebuilds_the_last_lecture(tmp_path):
    replay(tmp_path)
    fake = FakeOllama([SECTION, EXEC])
    lines = []
    out = asyncio.run(cli.rebuild_digest(None, db=tmp_path / "copilot.sqlite", client=fake.async_client(),
                                         sink=FolderSink(tmp_path / "again"), echo=lines.append))
    assert (tmp_path / "again" / "AI Developers — Python").is_dir() and out["sections"] == 9
    assert len(fake.requests) == 2


def test_digest_command_without_any_lecture(tmp_path):
    with pytest.raises(RuntimeError, match="no lecture"):
        asyncio.run(cli.rebuild_digest(None, db=tmp_path / "empty.sqlite", client=FakeOllama().async_client(),
                                       sink=FolderSink(tmp_path / "x"), echo=print))


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


# ---------- M4: fact-checking from the command line ----------

def test_replay_with_fact_checking_verifies_and_reports_the_cost(tmp_path):
    from tests.stubs import FakeGemini
    ok = {"verdict": "correct", "confidence": 0.9, "explanation": "נכון.", "sources": ["https://a"]}
    material = json.dumps({"chunk_summary": "סיכום", "concepts": [], "items": [],
                           "claims": [{"text": "t", "normalized": "n", "importance": 90}]})
    summary, lines = replay(tmp_path, replies=[material, material, SECTION, EXEC], gemini=FakeGemini([ok, ok]))
    assert summary["verifier"]["verified"] == 2 and summary["cost_usd"] > 0
    assert any(line.startswith("verifier:") and "2 verified" in line and "$" in line for line in lines)


def test_replay_without_fact_checking_makes_no_network_call(tmp_path):
    summary, lines = replay(tmp_path, fact_check=False, gemini=None)
    assert summary["verifier"] is None and summary["net_calls"] == 0
    assert any("fact-checking off" in line for line in lines)


def test_main_refuses_fact_checking_without_a_key(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(cli, "ENV_FILE", tmp_path / "missing.env")
    f = tmp_path / "x.m4a"
    f.write_bytes(b"x")
    assert cli.main(["replay", str(f), "--course", "X"]) == 2
    assert "GEMINI_API_KEY" in capsys.readouterr().err


def test_verify_command_checks_pending_material_claims_and_rebuilds_the_digest(tmp_path):
    from tests.stubs import FakeGemini
    material = json.dumps({"chunk_summary": "סיכום", "concepts": [], "items": [],
                           "claims": [{"text": "t1", "normalized": "n1", "importance": 90},
                                      {"text": "t2", "normalized": "n2", "importance": 50}]})
    replay(tmp_path, n=1, replies=[material, SECTION, EXEC], fact_check=False, gemini=None)   # no verdicts yet
    ok = {"verdict": "correct", "confidence": 0.9, "explanation": "נכון.", "sources": ["https://a"]}
    gemini, lines = FakeGemini([ok]), []
    out = asyncio.run(cli.verify_lecture(None, db=tmp_path / "copilot.sqlite", gemini=gemini,
                                         client=FakeOllama([SECTION, EXEC]).async_client(),
                                         sink=FolderSink(tmp_path / "courses"), echo=lines.append))
    assert out["verifier"] == {"verified": 1, "unchecked": 0, "skipped": 1, "retried": 0} and len(gemini.calls) == 1
    s = Store(tmp_path / "copilot.sqlite")
    rows = {r["text"]: r["status"] for r in s.con.execute("select text, status from claims")}
    assert rows == {"t1": "verified", "t2": "skipped"}
    assert "נכון" in s.con.execute("select digest_md from lecture_summaries").fetchone()[0]
    s.close()
    assert any(line.startswith("verifier:") for line in lines)
