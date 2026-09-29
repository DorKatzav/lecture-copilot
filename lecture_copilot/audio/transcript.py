"""TranscriptSource (PLAN.md §3.2): an exported transcript — Zoom .vtt or MacWhisper JSON — replayed through the
same pipeline. Segments are grouped into pseudo-chunks of about 45 s and written to disk as chunk_NNNN.json in
mw's own JSON shape; `asr.transcript.TranscriptASR` reads them back, so the pipeline never learns that no audio
was transcribed (D-M2-1). The source file is only read.
"""

import json
import re
from collections.abc import AsyncIterator
from pathlib import Path

from lecture_copilot.asr.base import Segment
from lecture_copilot.audio.sources import AudioChunk
from lecture_copilot.config import CHUNK_MAX_S, CHUNK_MIN_S, RUNS_DIR

TARGET_S = (CHUNK_MIN_S + CHUNK_MAX_S) / 2
_CUE = re.compile(r"^((?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})\s+-->\s+((?:\d+:)?\d{1,2}:\d{2}[.,]\d{3})")
_VOICE = re.compile(r"<v(?:\.[^\s>]+)*\s+([^>]+)>")
_TAG = re.compile(r"<[^>]+>")


def _seconds(stamp: str) -> float:
    parts = stamp.replace(",", ".").split(":")
    return sum(float(p) * 60 ** i for i, p in enumerate(reversed(parts)))


def parse_vtt(text: str) -> list[Segment]:
    lines = text.lstrip("﻿").splitlines()
    if not lines or not lines[0].startswith("WEBVTT"):
        raise ValueError("not a WEBVTT file")
    segs, i = [], 1
    while i < len(lines):
        m = _CUE.match(lines[i].strip())
        i += 1
        if not m:
            continue
        body = []
        while i < len(lines) and lines[i].strip():
            body.append(lines[i].strip())
            i += 1
        raw = " ".join(body)
        voice = _VOICE.search(raw)
        clean = " ".join(_TAG.sub("", raw).split())
        if clean:
            segs.append(Segment(t0=_seconds(m.group(1)), t1=_seconds(m.group(2)), text=clean,
                                speaker=voice.group(1).strip() if voice else None))
    return segs


def parse_mw_json(text: str) -> list[Segment]:
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError("no segments list — expected a MacWhisper JSON export")
    segs = [Segment(t0=s["start"] / 1000, t1=s["end"] / 1000, text=" ".join(s.get("text", "").split()),
                    speaker=s.get("speaker")) for s in data["segments"]]
    return [s for s in segs if s.text]


def group_segments(segments: list[Segment], target_s: float = TARGET_S,
                   max_s: float = CHUNK_MAX_S) -> list[list[Segment]]:
    """Close a group when it reached the target, when the next segment would push it over the maximum, or when
    the silence before the next segment is longer than a whole chunk (a break)."""
    groups: list[list[Segment]] = []
    for s in segments:
        g = groups[-1] if groups else None
        if g is None or g[-1].t1 - g[0].t0 >= target_s or s.t1 - g[0].t0 > max_s or s.t0 - g[-1].t1 > max_s:
            groups.append([s])
        else:
            g.append(s)
    return groups


class TranscriptSource:
    def __init__(self, lecture_id: str, file: Path, *, runs_dir: Path = RUNS_DIR):
        self.lecture_id, self.file = lecture_id, Path(file)
        self.dir = Path(runs_dir) / lecture_id

    async def __aiter__(self) -> AsyncIterator[AudioChunk]:
        text = self.file.read_text(encoding="utf-8")
        segments = parse_mw_json(text) if self.file.suffix.lower() == ".json" else parse_vtt(text)
        self.dir.mkdir(parents=True, exist_ok=True)
        for idx, group in enumerate(group_segments(segments), 1):
            t0 = group[0].t0
            path = self.dir / f"chunk_{idx:04d}.json"
            rows = [{"start": round((s.t0 - t0) * 1000), "end": round((s.t1 - t0) * 1000), "text": s.text,
                     "speaker": s.speaker} for s in group]
            path.write_text(json.dumps({"segments": rows}, ensure_ascii=False), encoding="utf-8")
            yield AudioChunk(self.lecture_id, idx, path, t0, group[-1].t1)
