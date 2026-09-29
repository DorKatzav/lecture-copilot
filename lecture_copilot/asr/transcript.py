"""The "ASR" of an exported transcript: reads the pseudo-chunk TranscriptSource wrote (D-M2-1)."""

import json
from pathlib import Path

from lecture_copilot.asr.base import ASRError, Segment


class TranscriptASR:
    name = "transcript"

    async def transcribe(self, wav: Path, language: str) -> list[Segment]:
        try:
            rows = json.loads(Path(wav).read_text(encoding="utf-8"))["segments"]
            return [Segment(t0=r["start"] / 1000, t1=r["end"] / 1000, text=r["text"], speaker=r.get("speaker"))
                    for r in rows]
        except (OSError, ValueError, KeyError, TypeError) as e:
            raise ASRError(f"{Path(wav).name} is not a transcript chunk: {type(e).__name__}") from e
