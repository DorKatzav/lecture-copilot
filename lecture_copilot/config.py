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
CHUNK_BUDGET_S = 30                                            # all processing of one ~45 s chunk
EXTRACT_TIMEOUT_S = BUDGET_S["extract"] * 2                    # per attempt; a longer call is a hang
EXTRACT_OPTIONS = {"temperature": 0, "seed": 42}               # D-M1-2: reproducible replays

# prompt versions in use (prompts/<name>.md); a change is a new file + a PROJECT_LOG line
EXTRACT_PROMPT = "extract_v3"                                  # D-M2-4: v0 + the highlight rule
DIGEST_MAP_PROMPT, DIGEST_EXEC_PROMPT = "digest_sections_v1", "digest_exec_v1"   # D-M2-5

# Digest (D-M2-2): map-reduce sized for the 4,096-token context
TOKENS_PER_WORD = 3                                            # measured 2.86 for Hebrew on gemma3:12b
DIGEST_MAP_INPUT_TOKENS = 2600                                 # chunk summaries + concepts of one block
DIGEST_FULL_WORDS = (120, 550)                                 # full summary: short replay … full lecture
DIGEST_OPTIONS = {"temperature": 0, "seed": 42}
DIGEST_TIMEOUT_S = 90                                          # per attempt
DIGEST_BUDGET_S = 120                                          # spec: Digest within 2 minutes of "סיום"

MW_BIN = os.getenv("MW_BIN", "mw")                             # MacWhisper CLI (Settings → Advanced → Install CLI)
# ASR model per course language (D-M0-8), explicit: never the app's current selection
MW_MODELS = {"he": "whisper-cpp:ivrit-ai-largev3", "en": "whisperkit:openai_whisper-large-v3-v20240930"}

OLLAMA_URL = "http://127.0.0.1:11434"
OLLAMA_NUM_CTX = 4096
OLLAMA_KEEP_ALIVE = "30m"                                      # stays loaded through a lecture break
# every request that can load a model sends the same load options, so Ollama never reloads mid-lecture.
# use_mmap (D-M1-3): Ollama 0.34 otherwise starts its runner with --load-mode none and gemma3:12b holds
# ~18 GB of dirty memory instead of ~1.8 GB plus reclaimable file-backed weights.
OLLAMA_LOAD_OPTIONS = {"num_ctx": OLLAMA_NUM_CTX, "use_mmap": True}


class Profile(BaseModel):
    fact_check: bool = True
    language: Literal["he", "en"] = "he"
