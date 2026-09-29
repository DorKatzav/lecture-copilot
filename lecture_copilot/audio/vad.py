"""Energy VAD: cut a mono stream into 30–60 s chunks at pauses, no overlap (DESIGN_HE §engineering).

Once CHUNK_MAX_S of audio is buffered, the cut goes to the middle of the longest run of frames below SILENCE_DB
inside [CHUNK_MIN_S, CHUNK_MAX_S); with no pause, to the quietest frame. Only the first CHUNK_MAX_S of the buffer
is looked at, so feeding in any block size gives the same cuts (LiveSource feeds blocks, FileSource a stream).
"""

from dataclasses import dataclass

import numpy as np

from lecture_copilot.config import CHUNK_MAX_S, CHUNK_MIN_S, SILENCE_DB

FRAME_S = 0.1
MIN_TAIL_S = 0.5


def frame_db(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    rms = np.sqrt(np.mean(np.square(x[: n * frame].reshape(n, frame), dtype=np.float64), axis=1))
    return 20 * np.log10(rms + 1e-12)


def choose_cut(db: np.ndarray, min_f: int, max_f: int, silence_db: float) -> int:
    """Frame index to cut at, in [min_f, max_f): middle of the longest silent run (earliest wins a tie)."""
    window = db[min_f:max_f]
    best_start, best_len, run_start = -1, 0, None
    for i, quiet in enumerate(np.append(window < silence_db, False)):
        if quiet and run_start is None:
            run_start = i
        elif not quiet and run_start is not None:
            if i - run_start > best_len:
                best_start, best_len = run_start, i - run_start
            run_start = None
    if best_len:
        return min_f + best_start + best_len // 2
    return min_f + int(np.argmin(window))


@dataclass(frozen=True)
class Cut:
    start: int              # first sample, counted from the start of the stream
    samples: np.ndarray     # float32 mono


class Splitter:
    def __init__(self, sr: int = 16000, min_s: float = CHUNK_MIN_S, max_s: float = CHUNK_MAX_S,
                 silence_db: float = SILENCE_DB):
        self.sr, self.silence_db = sr, silence_db
        self.frame = int(sr * FRAME_S)
        self.min_f, self.max_f = int(min_s / FRAME_S), int(max_s / FRAME_S)
        self.max_samples = self.max_f * self.frame
        self.buf = np.zeros(0, dtype=np.float32)
        self.start = 0

    def _emit(self, n: int) -> Cut:
        cut = Cut(self.start, self.buf[:n].copy())
        self.buf, self.start = self.buf[n:], self.start + n
        return cut

    def feed(self, samples: np.ndarray) -> list[Cut]:
        self.buf = np.concatenate([self.buf, samples.astype(np.float32, copy=False)])
        cuts = []
        while len(self.buf) >= self.max_samples:
            db = frame_db(self.buf[: self.max_samples], self.frame)
            cuts.append(self._emit(choose_cut(db, self.min_f, self.max_f, self.silence_db) * self.frame))
        return cuts

    def flush(self) -> list[Cut]:
        if len(self.buf) < MIN_TAIL_S * self.sr:
            self.buf = np.zeros(0, dtype=np.float32)
            return []
        return [self._emit(len(self.buf))]
