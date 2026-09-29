import asyncio
import json

import pytest

from lecture_copilot.asr.base import ASRError, Segment
from lecture_copilot.asr.transcript import TranscriptASR
from lecture_copilot.audio.transcript import TranscriptSource, group_segments, parse_mw_json, parse_vtt

VTT = """WEBVTT

1
00:03:15.000 --> 00:03:24.000
שלום לכולם

00:03:24.000 --> 00:03:25.500
<v דנה>היום נדבר על CAC</v>

NOTE a comment block
that spans lines

00:03:25.500 --> 00:03:31.000 align:start position:0%
שורה ראשונה
שורה שנייה

01:00:00.000 --> 01:00:02.000

"""


def seg(t0, t1, text="x"):
    return Segment(t0=t0, t1=t1, text=text)


def collect(source):
    async def go():
        return [c async for c in source]
    return asyncio.run(go())


# ---------- parsing ----------

def test_vtt_cues_become_segments_in_seconds():
    segs = parse_vtt(VTT)
    assert [(s.t0, s.t1, s.text) for s in segs] == [
        (195.0, 204.0, "שלום לכולם"), (204.0, 205.5, "היום נדבר על CAC"), (205.5, 211.0, "שורה ראשונה שורה שנייה")]


def test_vtt_voice_tag_is_the_speaker():
    assert [s.speaker for s in parse_vtt(VTT)] == [None, "דנה", None]


def test_vtt_without_header_is_refused():
    with pytest.raises(ValueError, match="WEBVTT"):
        parse_vtt("00:00:01.000 --> 00:00:02.000\nhi\n")


def test_vtt_short_timestamps_without_hours():
    assert parse_vtt("WEBVTT\n\n01:05.250 --> 01:07.000\nhi\n")[0].t0 == 65.25


def test_mw_json_export_with_speakers():
    data = {"segments": [{"start": 0, "end": 4200, "text": " שלום ", "speaker": "מרצה"},
                         {"start": 4200, "end": 9000, "text": "", "speaker": None},
                         {"start": 9000, "end": 12000, "text": "CAC"}]}
    segs = parse_mw_json(json.dumps(data))
    assert [(s.t0, s.t1, s.text, s.speaker) for s in segs] == [(0.0, 4.2, "שלום", "מרצה"), (9.0, 12.0, "CAC", None)]


def test_mw_json_of_another_shape_is_refused():
    with pytest.raises(ValueError, match="segments"):
        parse_mw_json('{"text": "only text"}')


# ---------- grouping ----------

def test_groups_close_at_the_target_length():
    segs = [seg(i * 10, i * 10 + 10) for i in range(12)]          # 120 s, contiguous
    groups = group_segments(segs)
    assert [(g[0].t0, g[-1].t1) for g in groups] == [(0, 50), (50, 100), (100, 120)]


def test_a_group_never_exceeds_the_maximum():
    segs = [seg(0, 40), seg(40, 70), seg(70, 75)]
    assert [(g[0].t0, g[-1].t1) for g in group_segments(segs)] == [(0, 40), (40, 75)]


def test_a_long_silence_closes_the_group():
    segs = [seg(0, 10), seg(10, 20), seg(900, 910)]               # a break in the lecture
    assert [(g[0].t0, g[-1].t1) for g in group_segments(segs)] == [(0, 20), (900, 910)]


def test_one_segment_longer_than_the_maximum_is_its_own_group():
    assert [len(g) for g in group_segments([seg(0, 5), seg(5, 200), seg(200, 210)])] == [1, 1, 1]


# ---------- source + provider ----------

def test_source_writes_pseudo_chunks_to_disk(tmp_path):
    f = tmp_path / "lecture.cc.vtt"
    f.write_text(VTT, encoding="utf-8")
    chunks = collect(TranscriptSource("LEC", f, runs_dir=tmp_path / "runs"))
    assert [c.idx for c in chunks] == [1] and chunks[0].path == tmp_path / "runs" / "LEC" / "chunk_0001.json"
    assert (chunks[0].t0, chunks[0].t1) == (195.0, 211.0)
    data = json.loads(chunks[0].path.read_text(encoding="utf-8"))
    assert data["segments"][0] == {"start": 0, "end": 9000, "text": "שלום לכולם", "speaker": None}


def test_provider_returns_chunk_relative_segments_with_speakers(tmp_path):
    f = tmp_path / "lecture.vtt"
    f.write_text(VTT, encoding="utf-8")
    (chunk,) = collect(TranscriptSource("LEC", f, runs_dir=tmp_path))
    segs = asyncio.run(TranscriptASR().transcribe(chunk.path, "he"))
    assert [(s.t0, s.t1, s.speaker) for s in segs] == [(0.0, 9.0, None), (9.0, 10.5, "דנה"), (10.5, 16.0, None)]


def test_json_export_is_detected_by_suffix(tmp_path):
    f = tmp_path / "export.json"
    f.write_text(json.dumps({"segments": [{"start": 0, "end": 1000, "text": "a"}]}), encoding="utf-8")
    assert len(collect(TranscriptSource("L", f, runs_dir=tmp_path))) == 1


def test_provider_fails_on_anything_that_is_not_a_pseudo_chunk(tmp_path):
    wav = tmp_path / "chunk_0001.wav"
    wav.write_bytes(b"RIFF")
    with pytest.raises(ASRError):
        asyncio.run(TranscriptASR().transcribe(wav, "he"))


def test_the_source_file_is_never_modified(tmp_path):
    f = tmp_path / "lecture.vtt"
    f.write_text(VTT, encoding="utf-8")
    before = f.read_bytes()
    collect(TranscriptSource("L", f, runs_dir=tmp_path / "runs"))
    assert f.read_bytes() == before
