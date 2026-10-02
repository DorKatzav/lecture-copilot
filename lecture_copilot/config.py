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
VERIFIER_MODEL = "gemini-3.7-flash"                           # listed for Dor's key on 2026-10-01
GEMINI_HOST = "generativelanguage.googleapis.com"
GEMINI_PRICE_PER_M = {"in": 0.75, "out": 3.75}                 # USD per 1M tokens, prices of 2026-09-22 (PROJECT_LOG)
VERIFY_TIMEOUT_S = 30
VERIFY_CONCURRENCY = 2                                         # spec: semaphore 2, never in the chunk loop
COST_BUDGET_USD = 0.20                                         # spec: per lecture
VERIFY_MIN_IMPORTANCE, MATERIAL_MIN_IMPORTANCE = 70, 85
BUDGET_S = {"asr": 8, "extract": 15, "embed": 2}
CHUNK_BUDGET_S = 30                                            # all processing of one ~45 s chunk
EXTRACT_TIMEOUT_S = BUDGET_S["extract"] * 2                    # per attempt; a longer call is a hang
EXTRACT_OPTIONS = {"temperature": 0, "seed": 42}               # D-M1-2: reproducible replays

# prompt versions in use (prompts/<name>.md); a change is a new file + a PROJECT_LOG line
EXTRACT_PROMPT = "extract_v4"                                  # v3 + the course memory (M3); v5 rejected (D-M4-2)
# ★ only on a strong signal in the transcript (Dor, 1.10): a highlight the model extracts is kept only when the chunk
# contains one of these; a plain "חשוב" is not enough (D-M4-2, enforced in code: a prompt rule alone killed every ★)
STRONG_SIGNALS = ("זה במבחן", "יהיה במבחן", "במבחן", "תזכרו", "תזכור", "אל תשכחו", "אל תשכח", "חשוב מאוד", "הכי חשוב",
                  "חשוב ביותר", "חשוב שתזכרו", "לזכור")
DIGEST_MAP_PROMPT, DIGEST_EXEC_PROMPT = "digest_sections_v1", "digest_exec_v2"   # D-M2-5, D-M3-5
RECAP_PROMPT = "recap_v1"

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

# memory (M3): bge-m3 vectors; sqlite-vec when it loads, numpy otherwise (set VEC_BACKEND to force one)
EMBED_DIMS = 1024
VEC_BACKEND = os.getenv("VEC_BACKEND")                        # None = auto, "sqlite-vec", "numpy"
MEMORY_K = 5                                                  # hits recalled per chunk (spec: top-5)
ALREADY_SAID_COSINE = 0.85                                    # a concept this close to an earlier one was already said
CONTRADICTION_BONUS = 20                                      # spec: a contradiction of an earlier lecture, +20


class Profile(BaseModel):
    fact_check: bool = True
    language: Literal["he", "en"] = "he"


def load_env(path: Path = ROOT / ".env", fact_check: bool = True) -> set[str]:
    """`.env` → os.environ, never overriding what is already set. Returns the names it set.
    Raises only when fact-checking is on and no GEMINI_API_KEY is available."""
    loaded = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip().strip('"').strip("'")
            if name and name not in os.environ:
                os.environ[name] = value
                loaded.add(name)
    if fact_check and not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not set (.env) — needed for fact-checking; use --no-fact-check to skip")
    return loaded
