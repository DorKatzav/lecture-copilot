# Project Log — Lecture Copilot

Decisions, experiments, metrics, failures, lessons. Newest entries last.
Decision ids: `D-M<milestone>-<n>`. Spec-level decisions 1–8 live in `DESIGN_HE.html`.

## 2026-09-22 — Planning (spec v1 → v9)
- Started from the v3 pitch (`lecture-copilot-pitch.html`, `Lecture-Copilot-OnePager.pdf`). Spec rewritten as an RTL HTML page and taken through 5 review passes (product/UX, engineering, academic+schedule, consistency, "a week as Dor"): ~60 findings, most applied.
- Research: MacWhisper CLI (`mw transcribe`, files only, `--stream`, json) cannot record or stream from the mic; meeting recording is in-app and post-hoc. Decision: our code records, `mw` transcribes chunks; no `--persist`, no reads of MacWhisper's SQLite. Speaker names arrive via exported transcript → `TranscriptSource`.
- Research: verifier API cost (published prices 2026-09-22). Gemini Flash: $0.75/$3.75 per 1M tokens + Google Search grounding 5,000 free requests/month → ~$0.05–0.10 per lecture. Claude Haiku 4.5: $1/$5 + $10/1k searches → ~$0.30. OpenAI gpt-5.6-luna: search billed as 8k tokens/call → ~$0.22. Decision 6: Gemini Flash; Claude as an alternative behind the same `Verifier` protocol.
- Research: MacBook mic from the audience — every source says placement beats models; sit in rows 1–3; no external mic (Dor's choice). Stage-0 check added.
- Locked with Dor: no alerts of any kind (Gatekeeper → Ranker); web page not TUI; sqlite-vec not Chroma; Zoom = record then Replay until BlackHole is tried on 17.10; meetings = course "פגישות" + decision/owner kinds; no submission — product-first priorities; live mic is core; Eval minimal; Notion is a sink, designed in M6; language per course (he/en); benchmark = 7/6 + 9/6 lectures (Zoom `.vtt` available).
- Deliverables written: `DESIGN_HE.html` (spec v9 + Notion section), `CLAUDE.md`, `PLAN.md` (contracts §3, M0–M7 gates §4, manual steps §7), `PLAN_HE.html`, `.cursor/rules/lecture-copilot.mdc`, `prompts/{extract,digest,verifier,recap}_v0.md`.
- Next: M0 (23–25.9). Dor's manual steps for M0: download 7/6 + 9/6 lectures, Gemini key, MacWhisper CLI install, 1-minute mic test from the seat.

## 2026-09-22 — M0 skeleton + stage-0 (in progress, branch `feat/m0-skeleton`)
- Built: git repo (planning docs on `main`), conda env `copilot` (Python 3.11.16, SQLite 3.53.4), package skeleton, `config.py`, `agents/schemas.py` (ExtractResult family), `prompts.py`, `llm.py`, `scripts/{setup_models.sh, make_fixture.sh, stage0.py, gate.py}`, CI. 66 tests green, ruff clean.
- Stage-0 check 3 (sqlite-vec): loads in `copilot` (sqlite-vec v0.1.9, `enable_load_extension` available) → no numpy fallback needed. From `eval/stage0.json`.
- Gate first run: `GATE M0: FAIL 2/8` — mw CLI, models, fixture, mw bench, extract bench, mic all missing, each failure names its fix. Fixture check seen passing on a synthetic 600 s file and failing without it (file deleted after).
- D-M0-1 — Ollama and ffmpeg were not installed (the plan assumed them). Installed via Homebrew formulae (ollama 0.34.2, ffmpeg 9.0.2), not the Ollama menu-bar app: no update popups, and the launcher starts `ollama serve` explicitly. Alternative: `Ollama.app`. Cost: none. Model pull runs at ~1 MB/s (~3 h for the three models).
- D-M0-2 — two modules not in PLAN §2's tree: `lecture_copilot/prompts.py` (strict loader: a missing or unknown `{{var}}` raises) and `lecture_copilot/llm.py` (the single Ollama JSON path). Why: stage-0 check 2 must measure the exact call path M1's `extract()` will use. Cost if wrong: move two files.
- D-M0-3 — Ollama structured outputs: `format` = the ExtractResult JSON schema, plus independent pydantic validation and one retry with the error appended. "Valid JSON" in the gate = the first attempt passes pydantic (stricter than parseable). Provider errors return as `JsonCall.error`, never raise. Alternative: `format="json"` only. Cost if wrong: one flag.
- D-M0-4 — mw verdict rule: 5 sequential runs on one 45 s clip; `hot` = p50 of the 5 ≤ 8 s (BUDGET asr); a run > 24 s (3 × budget, PLAN §3.3) is a hang; decision = mw only if hot and 0 hangs, else mlx-whisper moves up the ladder. Also logged: cold (1st) run, 2-in-parallel wall time, one run while the app is busy.
- D-M0-5 — stage-0 numbers live in `eval/stage0.json` (aggregates only, committed); raw transcripts and model outputs stay in `runs/stage0/` (gitignored, course material). `STAGE0_HE.html` gets numbers through `<span data-metric>` placeholders filled by `stage0.py report` — never typed.
- D-M0-6 — planning history (`lecture-copilot-pitch.html`, `Lecture-Copilot-OnePager.pdf`, `lecture-copilot-spec-v4…v9.html`) is not in PLAN §2 and the repo will be public → not committed; excluded locally in `.git/info/exclude` until Dor decides.
- `requirements.txt` holds M0 dependencies only and grows per milestone (build only the current milestone).
- Unverified until the CLI is installed: `mw transcribe` flags (`--format json --language he -o <dir>`) and its JSON shape; `stage0.mw_text` raises on an unknown shape instead of guessing.
- Model pull stopped at Dor's request (not on wifi) after ~1.0 GB of `qwen3:8b`; partial blobs kept in `~/.ollama/models` — `scripts/setup_models.sh` resumes them when Dor says go. No Ollama process left running.
- Folder tidied: the 8 planning-history files moved to `docs/history/` (still local-only per D-M0-6; `lecture-copilot-spec-v9.html` is byte-identical to `DESIGN_HE.html`), `lecture_copilot.egg-info` and empty placeholder dirs removed. Root now matches PLAN §2. Tests 66/66, gate unchanged (FAIL 2/8).
