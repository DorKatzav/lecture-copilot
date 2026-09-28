# CLAUDE.md — Lecture Copilot working agreement

Read this first in every session. Then read the tail of `PROJECT_LOG.md` (last entry) to know where we are.
The spec is `DESIGN_HE.html` (v9, approved 2026-09-22, decisions 1–8 inside). This plan argues from that spec.

## What this is (one paragraph)

A local-first "course memory" for Dor's B.A. lectures (Hebrew + English, in class and on Zoom). One button records,
MacWhisper transcribes (via its CLI), a local LLM extracts concepts / claims / questions / tasks / highlights, a
hybrid-search memory over the whole course flags what was already said and what contradicts, Gemini quietly fact-checks
material claims, and a Digest + a course page land in a Drive-synced folder and in Notion. **Nothing ever pops up
during a lecture.** No submission, no grade: this is a tool Dor uses from the first lecture on 2026-10-18.

## Language and reporting (Dor's preferences — same as Northwind, FunnelIQ, HarborVale)

- **Terminal conversation: English only**, short and professional. Hebrew never goes to the terminal.
- **Hebrew (masculine forms, לשון זכר) lives in:** HTML report pages, the Obsidian vault, and everything the product
  writes for Dor (Digest, course page, Notion pages, UI copy). Concept names stay in their original language (CAC is CAC).
- **After every milestone:** a Hebrew summary as an HTML page `docs/reports/M<N>_HE.html`, opened in the browser.
  Anything long or with options → a Hebrew HTML page (`docs/notes/*_HE.html`); the terminal gets a short pointer + the question.
- Hebrew HTML pages: `lang="he" dir="rtl"`, the Lecture Copilot design system (Heebo + Assistant + IBM Plex Mono;
  paper / local-teal / cloud-purple / accent-red / ok / warn tokens — copy the `<style>` block from `DESIGN_HE.html`).
  `<code>` is LTR. Verify RTL rendering with a real browser screenshot before calling it done.
- Never markdown tables in Hebrew; in Obsidian use labeled bullet lists.
- All code, UI ids, file names, commits, PR titles and bodies: English.
- Report findings as they are, even when unflattering (e.g. "the local model missed 40% of English terms").
- Start each milestone with a plain paragraph of what it does and why (Dor learns through the project).

## How Dor likes to work (the philosophy)

- **Design first, code second.** The spec is approved; do not re-open closed decisions. A surprise becomes a
  `D-M<N>-<n>` decision in `PROJECT_LOG.md` with alternatives and cost — never a silent patch.
- **Milestones with gates.** M0…M7. Every milestone has `python scripts/gate.py --m N` that prints
  `GATE M<N>: PASS k/k` (or FAIL / SKIP with a reason). Fix, don't skip. A check that was never seen failing is untested
  code — break things on purpose (renamed column, empty transcript, wifi off) and show the gate catching it.
- **Build only the current milestone.** No running ahead, no unrequested features. If something extra is forced, do it and
  say plainly it was not in the plan. Optional things are offered, not added.
- **The cut ladder is law** (spec §scope): if a milestone runs a day over, the next item on the ladder is dropped
  without discussion — Zoom/BlackHole → LangGraph wrapper → re-rank → flashcards → mlx backend → digest.html.
- **Tests next to the code, no notebooks.** `pytest -q && ruff check .` green before any push. Tests never call
  MacWhisper, Ollama or Gemini — they use `tests/stubs.py`.
- **Numbers are generated, never typed.** Eval numbers come from `eval/results.json`; cost and latency from the
  `decisions` table.
- **Provided files are sacred.** Course files in `data/lectures/<date>_<lecture|tirgul>/` (7/6 + 9/6 `.vtt`, 19/6 Tirgul
  recording for audio — D-M0-9) are never modified or renamed. They are also never committed (`.gitignore`) — course material, not ours to publish.
- **Dor owns manual steps** (keys, MacWhisper settings, BlackHole/Audio MIDI Setup, downloads). Collected in PLAN.md §7.
  Never type his passwords into a browser; never commit secrets.
- Two review moments per milestone: Dor reads the Hebrew report, then approves the merge.

## Sources of truth (do not re-derive, do not contradict)

- `DESIGN_HE.html` — the product spec v9: user flow, screens, Digest layout, memory mechanism, scope + cut ladder,
  engineering decisions, data model, minimal eval, schedule, "done" criteria, stage-0 checks, decisions 1–8.
- `PLAN.md` — contracts (§3: every protocol, dataclass and file format), milestones M0–M7 with gates (§4), protocol (§5),
  standing rules (§6), Dor's manual steps (§7).
- `PROJECT_LOG.md` — decisions `D-M<N>-<n>`, experiments, metrics, incidents, lessons. Append; never rewrite history.
- `prompts/*.md` — every LLM prompt is a versioned file (`extract_v0.md` …). Prompts never live inline in code.
- Obsidian vault `~/Documents/Obsidian Vault/Projects/Lecture Copilot/` — Hebrew notebook: home page with a status
  table, `שלבים/` one note per milestone, `Log/<date> — M<N>.md`, `החלטות.md`. Write the `.md` files directly.

## Per-milestone protocol (every M, no shortcuts)

1. `git checkout -b feat/m<N>-<slug>` from an up-to-date `main`; one feature branch + PR per milestone.
2. Open with a short plain-English paragraph: what this milestone does and why.
3. Build with tests next to the code; small commits with `feat:` / `test:` / `docs:` / `fix:` prefixes.
4. `python scripts/gate.py --m <N>` must print `GATE M<N>: PASS k/k`.
5. Hebrew report `docs/reports/M<N>_HE.html` (+ screenshots in `docs/reports/img/m<N>_*.jpg` when there is UI).
6. Short English terminal summary: what was built, gate result, what's next.
7. Obsidian: home status table, milestone note, `Log/<date> — M<N>.md`, `החלטות.md` for any `D-M<N>-x`.
8. `PROJECT_LOG.md` entry; push; `gh pr create`; CI green; **Dor reviews the Hebrew report and approves the merge.**
9. Docs of a milestone go on a separate `docs/m<N>-report` branch + PR after the feature PR.

## Standing rules (project-specific)

- **Silence is a feature.** No notifications, no popups, no sounds, no Mute — there is nothing to mute. Importance
  only *orders* the dashboard and the Digest. If a change would make something appear on its own, it is wrong.
- **Audio never leaves the machine.** The only outbound calls are Gemini (one claim's text + course context) and
  Notion (Digest text). Every outbound call goes through `store/net.py` and is logged in `decisions` (host, bytes, cost).
  Local audio, transcripts and the SQLite DB are gitignored.
- **One pipeline, many sources.** `ChunkSource` is an async iterator (`LiveSource`, `FileSource(pace)`,
  `TranscriptSource`). The pipeline never knows where a chunk came from. No `if live:` anywhere.
- **The disk is the interface** between recorder and worker (`chunk_0001.wav` + `call_soon_threadsafe`). A crash
  loses at most one chunk; the UI hydrates from SQLite, never from memory.
- **Budget per chunk ≤ 30 s** for 45 s of audio: ASR ≤ 8, one Ollama generation ≤ 15, one batch of embeddings ≤ 2.
  Gemini runs in a separate worker (semaphore 2) and never in the chunk loop. `queue_depth ≥ 2` → skip extraction on
  low-speech chunks, catch up in the Digest.
- **Language is per course** (`courses.language` he|en), never auto-detected per chunk. Digest prose is Hebrew.
- **One local LLM for live and Digest:** Gemma 3 12B (`num_ctx=4096`) for extraction and `digest()` (D-M0-10 — qwen3:8b
  leaked Cyrillic/Arabic into Hebrew in stage 0). Only one LLM loaded at a time; MacWhisper + Gemma + bge-m3 must fit
  24 GB together — measured in M1: peak 22.2 GB with Dor's apps open, pressure "warning", no swap growth (D-M1-6). Every extraction output passes a foreign-script check that lists
  forbidden scripts explicitly (Cyrillic, Arabic, CJK, Latin Extended Additional); accented Latin (é, ü) is allowed.
- **IDs are ULIDs minted in Python before any write.** Replaying the same lecture is an upsert, never a duplicate.
- **MacWhisper is a provider, not a dependency.** We call `mw transcribe` on files; we never read its internal SQLite,
  never use `--persist`. Speaker names come in through an exported transcript → `TranscriptSource`.
- **Prompts are files with versions.** Changing a prompt = new file + a line in `PROJECT_LOG.md` with before/after on the fixture.
- Secrets only in `.env` (gitignored): `GEMINI_API_KEY`, `NOTION_TOKEN`, `NOTION_*_DB`. `git grep --untracked -iE "AIza[0-9A-Za-z_-]{30,}|AQ\.[0-9A-Za-z_.-]{40,}|ntn_[A-Za-z0-9]{20,}|secret_[A-Za-z0-9]{20,}"` must stay empty before every push (same pattern as `scripts/gate.py`; AI Studio keys come as `AIza…` or `AQ.…`).

## Environment

- macOS, MacBook Pro M5 Pro, 24 GB. conda env **`copilot`** (Python 3.11, uv/Homebrew Python if conda's sqlite lacks
  `enable_load_extension` — checked in M0). Never `AI_dev` or `base`. Use `python`, not `python3`.
- Ollama is started with `scripts/ollama_serve.sh` (prompt cache off, D-M1-4) — never a plain `ollama serve`.
- Ollama: `gemma3:12b`, `bge-m3` (`qwen3:8b` kept only from the stage-0 comparison) — stored in `~/Projects/_shared/models/ollama` (shared across projects;
  `~/.ollama/models` is a symlink to it). MacWhisper Pro with the CLI installed (`mw version` works).
- Gemini key from aistudio.google.com in `.env` as `GEMINI_API_KEY`. Notion internal integration token as `NOTION_TOKEN`.
- Run: `python -m lecture_copilot.cli copilot` (launcher) · replay: `python -m lecture_copilot.cli replay <file|json> --pace fast`
  · tests: `pytest -q && ruff check .` · gate: `python scripts/gate.py --m N` · eval: `python -m lecture_copilot.cli eval`.
- `gh` is logged in as DorKatzav. Own git repo in this folder; GitHub `DorKatzav/lecture-copilot` (public, recordings excluded).
- Report viewer: `python -m http.server 8765` from the project folder, then open in Chrome.

## Lessons already paid for (from Northwind, FunnelIQ, HarborVale — don't repeat)

- Never `git add -A` from a parent folder; check `git rev-parse --show-toplevel` first.
- Provider calls fail: one retry with backoff, then mark the item failed/unchecked and continue — never abort the run.
- The gate's secret scan must match key material only (it once matched itself).
- Verify UI and RTL with a real browser screenshot, not by reading the code.
- Structured outputs with an independent Python validator + one retry: zero failures on real runs. Same here for
  every Ollama JSON call (`pydantic` model + one retry with the error message appended).
- Keep start commands and model names explicit in config, never inferred.
- `Path.write_text` without `newline=""` rewrites line endings; artifacts that must be byte-identical set it explicitly.
- Whisper on short chunks with auto language detection flips languages mid-lecture (spec review): fixed language per course.
