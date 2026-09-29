"""MacWhisper as an ASR provider through its CLI (PLAN.md §3.3, D-M0-7/8). We call `mw transcribe` on a file and
read the JSON it writes next to the chunk; never `--persist`, never MacWhisper's internal database."""

import asyncio
import json
from pathlib import Path

from lecture_copilot.asr.base import ASRError, Segment
from lecture_copilot.config import BUDGET_S, MW_BIN, MW_MODELS


class MacWhisperASR:
    name = "mw"

    def __init__(self, binary: str = MW_BIN, models: dict[str, str] = MW_MODELS,
                 timeout_s: float = BUDGET_S["asr"] * 3):
        self.binary, self.models, self.timeout_s = binary, models, timeout_s

    def command(self, wav: Path, out_json: Path, language: str) -> list[str]:
        return [self.binary, "transcribe", str(wav), "--model", self.models[language], "--language", language,
                "--format", "json", "--no-speakers", "-o", str(out_json), "--overwrite"]

    async def transcribe(self, wav: Path, language: str) -> list[Segment]:
        out = Path(wav).with_suffix(".json")
        out.unlink(missing_ok=True)
        try:
            proc = await asyncio.create_subprocess_exec(*self.command(wav, out, language),
                                                        stdout=asyncio.subprocess.PIPE,
                                                        stderr=asyncio.subprocess.PIPE)
        except FileNotFoundError as e:
            raise ASRError(f"{self.binary} not found — MacWhisper → Settings → Advanced → Install CLI") from e
        try:
            _, err = await asyncio.wait_for(proc.communicate(), self.timeout_s)
        except TimeoutError:
            proc.kill()
            await proc.wait()
            raise ASRError(f"mw timed out after {self.timeout_s:.0f} s (killed)") from None
        if proc.returncode != 0:
            raise ASRError(f"mw exit {proc.returncode}: {err.decode(errors='replace').strip()[-200:]}")
        return parse_output(out)


def parse_output(out: Path) -> list[Segment]:
    """mw JSON: {"text", "segments": [{"id", "start", "end" (int ms), "text", "words"}]}. Times are chunk-relative."""
    if not out.exists() or out.stat().st_size == 0:
        raise ASRError(f"mw wrote no output at {out.name}")
    try:
        data = json.loads(out.read_text(encoding="utf-8"))
        raw = data["segments"]
        segs = [Segment(t0=s["start"] / 1000, t1=s["end"] / 1000, text=s["text"].strip()) for s in raw]
    except (ValueError, KeyError, TypeError) as e:
        raise ASRError(f"unreadable mw output {out.name}: {type(e).__name__}") from e
    return [s for s in segs if s.text]
