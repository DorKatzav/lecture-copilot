import asyncio
import shutil
import wave

import numpy as np
import pytest

from lecture_copilot.audio.sources import AudioChunk, FileSource

SR = 16000


def speech_with_pauses():
    t = np.arange(40 * SR) / SR
    tone = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    gap = np.zeros(SR, dtype=np.float32)
    return np.concatenate([tone, gap, tone, gap, tone[: 30 * SR]])   # 40 + 1 + 40 + 1 + 30 = 112 s


def fake_decoder(audio, block=SR * 7):
    async def pcm_blocks(_path):
        for i in range(0, len(audio), block):
            yield audio[i:i + block]
    return pcm_blocks


def collect(source):
    async def go():
        return [c async for c in source]
    return asyncio.run(go())


def read_wav(path):
    with wave.open(str(path)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, SR)
        return w.getnframes()


def test_chunks_are_wavs_on_disk_with_lecture_times(tmp_path):
    src = FileSource("LEC1", tmp_path / "lecture.m4a", runs_dir=tmp_path / "runs",
                     decoder=fake_decoder(speech_with_pauses()))
    chunks = collect(src)
    assert [c.idx for c in chunks] == [1, 2, 3]
    assert [c.path.name for c in chunks] == ["chunk_0001.wav", "chunk_0002.wav", "chunk_0003.wav"]
    assert all(c.path.parent == tmp_path / "runs" / "LEC1" and c.lecture_id == "LEC1" for c in chunks)
    assert chunks[0].t0 == 0 and chunks[1].t0 == pytest.approx(40.5) and chunks[-1].t1 == pytest.approx(112.0)
    assert all(a.t1 == b.t0 for a, b in zip(chunks, chunks[1:], strict=False))
    assert [read_wav(c.path) for c in chunks] == [round((c.t1 - c.t0) * SR) for c in chunks]


def test_fast_pace_never_sleeps(tmp_path):
    sleeps = []

    async def fake_sleep(s):
        sleeps.append(s)

    src = FileSource("L", tmp_path / "x.m4a", runs_dir=tmp_path, decoder=fake_decoder(speech_with_pauses()),
                     sleep=fake_sleep)
    collect(src)
    assert sleeps == []


def test_realtime_pace_releases_each_chunk_when_its_audio_has_played(tmp_path):
    clock = [100.0]
    released = []

    async def fake_sleep(s):
        clock[0] += s

    src = FileSource("L", tmp_path / "x.m4a", pace="realtime", runs_dir=tmp_path,
                     decoder=fake_decoder(speech_with_pauses()), sleep=fake_sleep, clock=lambda: clock[0])
    for c in collect(src):
        released.append((clock[0] - 100.0, c.t1))
    assert all(at >= t1 - 1e-6 for at, t1 in released)


def test_audio_chunk_is_frozen(tmp_path):
    c = AudioChunk("L", 1, tmp_path / "a.wav", 0.0, 1.0)
    with pytest.raises(AttributeError):
        c.idx = 2


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_real_ffmpeg_decode(tmp_path):
    audio = speech_with_pauses()
    src_wav = tmp_path / "in.wav"
    with wave.open(str(src_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    chunks = collect(FileSource("L", src_wav, runs_dir=tmp_path / "runs"))
    assert len(chunks) == 3 and chunks[-1].t1 == pytest.approx(112.0, abs=0.05)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_unreadable_file_raises(tmp_path):
    bad = tmp_path / "bad.m4a"
    bad.write_bytes(b"not audio")
    with pytest.raises(RuntimeError, match="ffmpeg"):
        collect(FileSource("L", bad, runs_dir=tmp_path / "runs"))
