"""Chunk sources (PLAN.md §3.2). The pipeline iterates a ChunkSource and never knows where a chunk came from.

The disk is the interface: every chunk is a 16 kHz mono wav at runs/<lecture_id>/chunk_NNNN.wav before it is
yielded, so a crash loses at most the chunk in flight. LiveSource (M5) and TranscriptSource (M2) come later.
"""

import asyncio
import time
import wave
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

from lecture_copilot.audio.vad import Cut, Splitter
from lecture_copilot.config import RUNS_DIR

SR = 16000
BLOCK_BYTES = SR * 2 * 5  # 5 s of s16le


@dataclass(frozen=True)
class AudioChunk:
    lecture_id: str
    idx: int
    path: Path
    t0: float
    t1: float


class ChunkSource(Protocol):
    def __aiter__(self) -> AsyncIterator[AudioChunk]: ...


async def ffmpeg_pcm(path: Path) -> AsyncIterator[np.ndarray]:
    """Decode any audio/video file to float32 mono 16 kHz blocks. The source file is only read."""
    proc = await asyncio.create_subprocess_exec(
        "ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-vn", "-ac", "1", "-ar", str(SR), "-f", "s16le", "-",
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    pending = b""
    while chunk := await proc.stdout.read(BLOCK_BYTES):
        pending += chunk
        usable = len(pending) - len(pending) % 2
        yield np.frombuffer(pending[:usable], dtype=np.int16).astype(np.float32) / 32768
        pending = pending[usable:]
    err = (await proc.stderr.read()).decode(errors="replace").strip()
    if await proc.wait() != 0:
        raise RuntimeError(f"ffmpeg could not decode {path.name}: {err[:200]}")


def write_wav(path: Path, samples: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1, 1) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


class LiveSource:
    """The microphone (M5). sounddevice calls back on its own thread; blocks cross to the event loop through
    `call_soon_threadsafe`, the VAD cuts them, every chunk is a wav on disk before it is yielded (the disk is the
    interface — a crash loses at most the chunk in flight). `level` is the RMS of the last 250 ms in dB, for the
    meter. `stop()` ends the stream and flushes the tail."""

    def __init__(self, lecture_id: str, device: int | None = None, *, runs_dir: Path = RUNS_DIR, stream_cls=None):
        self.lecture_id, self.device = lecture_id, device
        self.dir = Path(runs_dir) / lecture_id
        self.stream_cls = stream_cls
        self.level = -120.0
        self._queue: asyncio.Queue[np.ndarray | None] | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False

    def _callback(self, indata, frames, time_info, status) -> None:   # audio thread
        block = np.ascontiguousarray(indata[:, 0], dtype=np.float32).copy()
        tail = block[-SR // 4:]
        rms = float(np.sqrt(np.mean(np.square(tail)))) if len(tail) else 0.0
        self.level = 20 * np.log10(rms + 1e-12)
        if self._loop is not None and self._queue is not None and not self._stopped:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, block)

    def stop(self) -> None:
        self._stopped = True
        if self._loop is not None and self._queue is not None:
            self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        import sounddevice as sd
        self._loop, self._queue = asyncio.get_running_loop(), asyncio.Queue()
        stream_cls = self.stream_cls or sd.InputStream
        stream = stream_cls(samplerate=SR, channels=1, blocksize=SR // 4, callback=self._callback, device=self.device)
        stream.start()
        splitter, idx = Splitter(sr=SR), 0

        def release(cut: Cut) -> AudioChunk:
            nonlocal idx
            idx += 1
            path = self.dir / f"chunk_{idx:04d}.wav"
            write_wav(path, cut.samples)
            return AudioChunk(self.lecture_id, idx, path, cut.start / SR, (cut.start + len(cut.samples)) / SR)
        try:
            while (block := await self._queue.get()) is not None:
                for cut in splitter.feed(block):
                    yield release(cut)
            for cut in splitter.flush():
                yield release(cut)
        finally:
            stream.stop()
            stream.close()


class FileSource:
    """A recording replayed through the same VAD as live audio. pace="realtime" releases each chunk only when
    its audio would have finished playing; "fast" does not wait."""

    def __init__(self, lecture_id: str, file: Path, pace: Literal["realtime", "fast"] = "fast", *,
                 runs_dir: Path = RUNS_DIR,
                 decoder: Callable[[Path], AsyncIterator[np.ndarray]] = ffmpeg_pcm,
                 sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
                 clock: Callable[[], float] = time.monotonic):
        self.lecture_id, self.file, self.pace = lecture_id, Path(file), pace
        self.dir = Path(runs_dir) / lecture_id
        self.decoder, self.sleep, self.clock = decoder, sleep, clock

    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        splitter, idx, started = Splitter(sr=SR), 0, self.clock()

        async def release(cut: Cut) -> AudioChunk:
            nonlocal idx
            idx += 1
            path = self.dir / f"chunk_{idx:04d}.wav"
            write_wav(path, cut.samples)
            chunk = AudioChunk(self.lecture_id, idx, path, cut.start / SR, (cut.start + len(cut.samples)) / SR)
            if self.pace == "realtime" and (wait := started + chunk.t1 - self.clock()) > 0:
                await self.sleep(wait)
            return chunk

        async for block in self.decoder(self.file):
            for cut in splitter.feed(block):
                yield await release(cut)
        for cut in splitter.flush():
            yield await release(cut)
