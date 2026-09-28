import numpy as np
import pytest

from lecture_copilot.audio.vad import Splitter, choose_cut, frame_db

SR = 16000


def tone(seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def silence(seconds):
    return np.zeros(int(seconds * SR), dtype=np.float32)


def cut_all(audio, block=None, **kw):
    sp = Splitter(sr=SR, **kw)
    cuts = []
    if block is None:
        cuts += sp.feed(audio)
    else:
        for i in range(0, len(audio), block):
            cuts += sp.feed(audio[i:i + block])
    return cuts + sp.flush()


# ---------- choose_cut ----------

def test_cut_is_the_middle_of_the_longest_silence_in_the_window():
    db = np.full(100, -10.0)
    db[40:43] = -60   # short pause
    db[70:80] = -60   # long pause
    assert choose_cut(db, 30, 90, -40) == 75


def test_equal_silences_prefer_the_earliest():
    db = np.full(100, -10.0)
    db[40:44] = -60
    db[70:74] = -60
    assert choose_cut(db, 30, 90, -40) == 42


def test_silence_outside_the_window_is_ignored():
    db = np.full(100, -10.0)
    db[5:25] = -60
    db[95:100] = -60
    db[50] = -30      # quietest in window, not silent
    assert choose_cut(db, 30, 90, -40) == 50


def test_no_silence_cuts_at_the_quietest_frame():
    db = np.full(100, -10.0)
    db[64] = -20
    assert choose_cut(db, 30, 90, -40) == 64


def test_frame_db_of_silence_and_full_scale():
    x = np.concatenate([np.zeros(1600, np.float32), np.ones(1600, np.float32)])
    db = frame_db(x, frame=1600)
    assert db[0] < -100 and db[1] == pytest.approx(0.0, abs=1e-6)


# ---------- Splitter ----------

def test_cuts_land_in_pauses_between_min_and_max():
    audio = np.concatenate([tone(40), silence(1), tone(50), silence(1), tone(30)])   # 122 s
    cuts = cut_all(audio)
    starts = [c.start / SR for c in cuts]
    assert starts[0] == 0 and starts[1] == pytest.approx(40.5, abs=0.1) and starts[2] == pytest.approx(91.5, abs=0.1)
    assert all(30 <= len(c.samples) / SR <= 60 for c in cuts[:-1])


def test_chunks_cover_the_audio_exactly_once():
    audio = np.concatenate([tone(40), silence(1), tone(50), silence(1), tone(30)])
    cuts = cut_all(audio)
    assert np.array_equal(np.concatenate([c.samples for c in cuts]), audio)
    assert [c.start for c in cuts] == list(np.cumsum([0] + [len(c.samples) for c in cuts[:-1]]))


def test_streaming_in_odd_blocks_gives_the_same_cuts():
    audio = np.concatenate([tone(35), silence(2), tone(45), silence(0.5), tone(70)])
    one = cut_all(audio)
    streamed = cut_all(audio, block=12_345)
    assert [(c.start, len(c.samples)) for c in one] == [(c.start, len(c.samples)) for c in streamed]


def test_speech_without_pauses_is_cut_at_max():
    cuts = cut_all(tone(130) * np.linspace(1, 0.5, int(130 * SR), dtype=np.float32))  # fading: quietest at the end
    assert all(len(c.samples) / SR <= 60 for c in cuts)


def test_a_remainder_shorter_than_half_a_second_is_dropped():
    assert cut_all(tone(0.3)) == []


def test_short_recording_is_one_chunk():
    cuts = cut_all(tone(12))
    assert len(cuts) == 1 and len(cuts[0].samples) == 12 * SR
