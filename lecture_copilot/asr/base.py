"""ASR protocol (PLAN.md §3.3). A provider turns one chunk wav into segments with chunk-relative times."""

from pathlib import Path
from typing import Protocol

from pydantic import BaseModel


class Segment(BaseModel):
    t0: float
    t1: float
    text: str
    speaker: str | None = None


class ASRError(RuntimeError):
    """The provider failed on this chunk (non-zero exit, hang, no output). The caller marks the chunk failed."""


class ASR(Protocol):
    name: str

    async def transcribe(self, wav: Path, language: str) -> list[Segment]: ...
