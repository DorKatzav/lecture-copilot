"""Paths, models, thresholds and the one mode flag (PLAN.md §3.1). Names are explicit, never inferred."""

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "copilot.sqlite"
RUNS_DIR = ROOT / "runs"
PROMPTS_DIR = ROOT / "prompts"
COURSES_ROOT = Path(os.getenv("COURSES_ROOT", "~/Google Drive/My Drive/Lecture-Copilot")).expanduser()

CHUNK_MIN_S, CHUNK_MAX_S, SILENCE_DB = 30, 60, -40
# one local LLM for live extraction and the Digest (D-M0-10: qwen3:8b leaked foreign scripts into Hebrew)
LIVE_MODEL, DIGEST_MODEL, EMBED_MODEL = "gemma3:12b", "gemma3:12b", "bge-m3"
VERIFIER_MODEL = "gemini-3.7-flash"
VERIFY_MIN_IMPORTANCE, MATERIAL_MIN_IMPORTANCE = 70, 85
BUDGET_S = {"asr": 8, "extract": 15, "embed": 2}

MW_BIN = os.getenv("MW_BIN", "mw")                             # MacWhisper CLI (Settings → Advanced → Install CLI)
# ASR model per course language (D-M0-8), explicit: never the app's current selection
MW_MODELS = {"he": "whisper-cpp:ivrit-ai-largev3", "en": "whisperkit:openai_whisper-large-v3-v20240930"}

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_NUM_CTX = 4096


class Profile(BaseModel):
    fact_check: bool = True
    language: Literal["he", "en"] = "he"
