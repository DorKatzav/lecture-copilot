# Lecture Copilot — Implementation Plan

> Working language: terminal in English; Hebrew (masculine) in HTML reports, Obsidian and product output. Code / files / commits in English.
> Spec: `DESIGN_HE.html` (v9, decisions 1–8 approved 2026-09-22). This plan argues from that spec.
> Environment: conda `copilot` (Python 3.11), created in M0. Ollama, MacWhisper Pro + CLI, Gemini key, Notion token.
> Keep `PROJECT_LOG.md` · commit after every passed gate · one feature branch + PR per milestone · Hebrew HTML report + Obsidian after every milestone.

**Goal:** From the first lecture of the year (2026-10-18) Dor presses one button, listens, and gets — without a single
popup — a live dashboard he can glance at, a Digest (executive summary → full summary → ★ → continuation from the
previous lecture → concepts → flagged claims → questions → tasks → notes) in a Drive-synced course folder and in Notion,
and a course page (cumulative glossary, all ★, "lecturer said / actually", search) that he studies from before exams.

**Architecture:** One Python process, three asyncio tasks. `audio/` yields `AudioChunk`s from a mic, a file or an exported
transcript. `asr/` turns a chunk into `Segment`s through MacWhisper's CLI (mlx-whisper optional). `agents/` runs four plain
async functions on each chunk — Extractor (one Ollama call), Memory (embeddings + FTS5 + sqlite-vec + RRF), Verifier
(Gemini + Google Search grounding + `get_course_context`, in a separate worker), Ranker (importance orders, never alerts).
`store/` is one SQLite file (tables + FTS5 + vectors) and is the only state. `output/` renders Digest, course page and
`Sink`s (folder, Notion). `web/` is FastAPI + one HTML page over WebSocket that hydrates from SQLite.

**Tech stack:** Python 3.11 · sounddevice · ffmpeg · MacWhisper CLI (`mw`) · Ollama (qwen3:8b, gemma3:12b, bge-m3) ·
Gemini 3.7 Flash (`google-genai`) · SQLite + FTS5 + sqlite-vec · pydantic 2 · FastAPI + uvicorn · notion-client ·
pytest · ruff · GitHub Actions (stubs only, no providers).

## Global constraints (from the spec)

- **Silence.** Nothing appears on its own. No notifications, sounds, popups, Mute. `importance` sorts; it never triggers.
- **Privacy.** Audio and transcripts never leave the machine. Outbound = Gemini (one claim + context) and Notion (Digest). All through `store/net.py`, logged in `decisions`.
- **One pipeline.** `ChunkSource` async iterator; `process_chunk(chunk, ctx)` does not know the source. `Profile(fact_check: bool)` is the only mode flag.
- **Disk as interface.** Recorder writes `runs/<lecture_id>/chunk_NNNN.wav` then signals the loop. Worker consumes paths. UI reads SQLite.
- **Budget.** ≤ 30 s processing per 45 s chunk (ASR ≤ 8, Ollama ≤ 15, embeddings ≤ 2). Gemini outside the loop.
- **Models.** qwen3:8b for live extraction, gemma3:12b only for `digest()`, bge-m3 embeddings, Gemini 3.7 Flash verifier. Language per course.
- **IDs.** ULID minted in Python; upsert on replay. `lecture_id` is the partition key everywhere.
- **MacWhisper is a provider.** `mw transcribe <file> --model <MW_MODELS[lang]> --language <he|en> --format json --no-speakers -o <file.json>`; no `--persist`, no internal DB reads. ASR model per course language: `he` → ivrit.ai large-v3, `en` → large-v3 Turbo (D-M0-8).
- **Recordings never modified, never committed.** `data/lectures/**` and `runs/**` gitignored.
- **Secrets** only in `.env`; secret scan in the gate.
- Own git repo; GitHub `DorKatzav/lecture-copilot` (public).

---

## 1. Data flow

```text
LiveSource (sounddevice, mic) ─┐
FileSource (ffmpeg, --pace)  ─┼─ AudioChunk(path, t0, t1) ──► asr.transcribe ──► Segment[]
TranscriptSource (mw JSON)   ─┘        (skips asr)                                   │
                                                                                     ▼
                              ┌───────── agents.extractor (qwen3:8b, one call) ── ExtractResult
                              │           chunk_summary, concepts, claims, questions, actions, highlights, importance
                              ▼
                       agents.memory: embed batch → FTS5 + vec → RRF top-5 → already_said / contradicts → adjust importance
                              │
                              ├──► store: items, claims (status=pending), segments, decisions
                              │
                              └──► agents.verifier worker (Gemini, semaphore 2): claims with importance ≥ threshold
                                       verdict, confidence, sources → claims.status=verified | unchecked (offline)
   web/ (FastAPI + WS) ◄── store ──► agents.ranker: importance order for dashboard + Digest
   "סיום" ──► output.digest (gemma3:12b over chunk_summaries + items) ──► Sinks: FolderSink (md+html) · NotionSink
           ──► output.course_page (cumulative glossary, ★, claims, tasks, search) ──► same Sinks
```

## 2. Repository structure

```text
lecture-copilot/                      git root; GitHub: DorKatzav/lecture-copilot
├── lecture_copilot/
│   ├── config.py                     paths, models, thresholds, Profile, load_env()
│   ├── cli.py                        copilot | record | replay | digest | eval | course-page
│   ├── audio/  sources.py (ChunkSource, LiveSource, FileSource, TranscriptSource)  vad.py  meter.py
│   ├── asr/    base.py (ASR protocol, Segment)  macwhisper.py  mlx.py (optional, M5+ ladder)
│   ├── agents/ schemas.py (ExtractResult…)  extractor.py  memory.py  verifier.py  ranker.py  recap.py  graph.py (ladder)
│   ├── store/  db.py (schema, ULID, upserts)  search.py (fts + vec + rrf)  embed.py  net.py
│   ├── output/ digest.py  course_page.py  sinks.py (Sink, FolderSink, NotionSink)  templates/*.md.j2
│   └── web/    app.py  index.html  (single page: before / during / after)
├── prompts/  extract_v0.md  digest_v0.md  verifier_v0.md  recap_v0.md
├── eval/     fixture_10min.m4a (gitignored)  benchmark.json  eval.py  results.json
├── tests/    stubs.py (fake mw / ollama / gemini / notion) + one test file per module
├── scripts/  gate.py  setup_models.sh  make_fixture.sh
├── data/lectures/ (gitignored)  runs/ (gitignored)  db/copilot.sqlite (gitignored)
├── docs/     reports/M<N>_HE.html  reports/img/  notes/
├── .cursor/rules/lecture-copilot.mdc  .github/workflows/ci.yml  requirements.txt  ruff.toml  pyproject.toml  .env.example  .gitignore
└── README.md  CLAUDE.md  PLAN.md  PLAN_HE.html  PROJECT_LOG.md  DESIGN_HE.html
```

## 3. Shared contracts (single source of truth — every milestone builds on these)

### 3.1 `lecture_copilot/config.py`
```python
ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "db" / "copilot.sqlite"; RUNS_DIR = ROOT / "runs"
COURSES_ROOT = Path(os.getenv("COURSES_ROOT", "~/Google Drive/My Drive/Lecture-Copilot")).expanduser()  # Drive desktop syncs it
CHUNK_MIN_S, CHUNK_MAX_S, SILENCE_DB = 30, 60, -40          # VAD split window
LIVE_MODEL, DIGEST_MODEL, EMBED_MODEL = "qwen3:8b", "gemma3:12b", "bge-m3"
VERIFIER_MODEL = "gemini-3.7-flash"                          # google-genai, grounding on
VERIFY_MIN_IMPORTANCE = 70; MATERIAL_MIN_IMPORTANCE = 85     # ranker labels, never alerts
BUDGET_S = {"asr": 8, "extract": 15, "embed": 2}
MW_BIN = os.getenv("MW_BIN", "mw")                           # MacWhisper CLI
MW_MODELS = {"he": "whisper-cpp:ivrit-ai-largev3", "en": "whisperkit:openai_whisper-large-v3-v20240930"}  # D-M0-8
class Profile(BaseModel): fact_check: bool = True; language: Literal["he", "en"] = "he"
def load_env() -> None: ...                                  # .env → os.environ; raises on missing GEMINI_API_KEY only when fact_check
```

### 3.2 `audio/sources.py`
```python
@dataclass(frozen=True)
class AudioChunk: lecture_id: str; idx: int; path: Path; t0: float; t1: float
class ChunkSource(Protocol):
    def __aiter__(self) -> AsyncIterator[AudioChunk]: ...
class LiveSource:      # sounddevice InputStream → vad.split → writes runs/<id>/chunk_NNNN.wav → loop.call_soon_threadsafe
    def __init__(self, lecture_id: str, device: int | None = None): ...
    level: float       # RMS of the last 250 ms, read by web/ for the meter
class FileSource:      # ffmpeg -i file -ac 1 -ar 16000 → vad.split; pace="realtime" sleeps t1-t0, "fast" does not
    def __init__(self, lecture_id: str, file: Path, pace: Literal["realtime", "fast"] = "fast"): ...
class TranscriptSource:  # mw JSON export (or Zoom .vtt) → yields Segment groups as pseudo-chunks; pipeline skips ASR
    def __init__(self, lecture_id: str, file: Path): ...
```

### 3.3 `asr/base.py`
```python
class Segment(BaseModel): t0: float; t1: float; text: str; speaker: str | None = None
class ASR(Protocol):
    name: str
    async def transcribe(self, wav: Path, language: str) -> list[Segment]: ...
# macwhisper.py: asyncio.create_subprocess_exec(MW_BIN, "transcribe", wav, "--model", MW_MODELS[language], "--language",
#   language, "--format", "json", "--no-speakers", "-o", out_json, "--overwrite")   # verified on mw 14.7.1 (D-M0-7): -o is a FILE
#   output: {text, segments[{id, start, end (int ms), text, words[]}]}
#   timeout = BUDGET_S["asr"] * 3; non-zero exit or empty output → ASRError (caller marks chunk failed, continues)
```

### 3.4 `agents/schemas.py`
```python
class Concept(BaseModel):  term: str; explanation: str; canonical_key: str
class Claim(BaseModel):    text: str; normalized: str; importance: int = Field(ge=0, le=100)
class Item(BaseModel):     kind: Literal["question","action","highlight","note","decision"]; text: str; owner: str | None = None; due: str | None = None
class ExtractResult(BaseModel):
    chunk_summary: str; concepts: list[Concept]; claims: list[Claim]; items: list[Item]
class MemoryHit(BaseModel): kind: str; id: str; lecture_id: str; text: str; score: float
class Verdict(BaseModel):  verdict: Literal["correct","incorrect","imprecise","unverifiable"]; confidence: float; explanation: str; sources: list[str]
```

### 3.5 `agents/*.py` (four plain async functions + recap)
```python
async def extract(segments: list[Segment], ctx: Ctx) -> ExtractResult          # one Ollama call, prompts/extract_v0.md, pydantic validate + 1 retry
async def remember(res: ExtractResult, ctx: Ctx) -> ExtractResult             # batch embed; per concept/claim: search → already_said flag, contradiction → importance += 20
async def verify(claim_id: str, ctx: Ctx) -> Verdict                          # Gemini + grounding + tool get_course_context(topic); cache by normalized hash
def rank(lecture_id: str) -> list[ClaimRow]                                    # ORDER BY importance DESC; labels: material (≥85) / minor
async def recap(lecture_id: str, minutes: int = 5) -> str                      # last N chunk_summaries → one Ollama call → 3 lines
class Ctx(BaseModel): lecture_id: str; course_id: str; profile: Profile; store: Store; asr: ASR
```

### 3.6 `store/db.py` (schema = DESIGN_HE §data model)
```text
courses(id, name, folder, language, is_meetings)
lectures(id ULID, course_id, week, date, title, source mic|zoom|file|transcript, fact_check, continues_id, started_at, ended_at, audio_path, status recording|ended|digested)
segments(id, lecture_id, chunk_id, t0, t1, text, speaker, asr, chunk_summary)
items(id, lecture_id, segment_id, kind concept|question|action|highlight|note|decision, text, explanation, canonical_key, owner, due, first_seen_lecture_id, t0, embedding BLOB)
claims(id, lecture_id, segment_id, text, normalized, importance, status pending|verified|skipped|unchecked, verdict, confidence, sources_json, cache_key, embedding BLOB)
lecture_summaries(lecture_id, bullets_json, digest_md, embedding BLOB)
fact_cache(cache_key, verdict, sources_json, checked_at)
decisions(id, lecture_id, node extractor|memory|verifier|ranker|net, input_ref, output_json, ms, tokens_in, tokens_out, cost_usd, ts)
items_fts / claims_fts  = FTS5(text, canonical_key)   ·   vec_items / vec_claims = sqlite-vec (bge-m3, 1024 dims)
```
```python
class Store:
    def upsert_lecture(...); def add_segments(...); def add_items(...); def add_claims(...)
    def search(self, query: str, course_id: str, k: int = 5) -> list[MemoryHit]   # RRF(FTS5 bm25, vec cosine); numpy fallback when sqlite-vec missing
    def previous_lecture(self, course_id: str) -> LectureRow | None
    def log(self, node: str, **fields) -> None                                     # decisions
```

### 3.7 `output/`
```python
def digest(lecture_id: str) -> DigestDoc      # map-reduce: chunk_summaries + items + claims + previous bullets → gemma3:12b (prompts/digest_v0.md) → sections in fixed order
def course_page(course_id: str) -> CoursePage  # templates only, no LLM: glossary (canonical_key, first_seen), all ★, claims lecturer-said/actually, open questions, tasks
class Sink(Protocol):
    def write_lecture(self, doc: DigestDoc) -> None
    def write_course(self, page: CoursePage) -> None
class FolderSink:  # COURSES_ROOT/<course>/W05_2026-11-04_<slug>/{digest.md, digest.html, transcript.txt, claims.json, flashcards.tsv} + <course>/course.html + index.md
class NotionSink:  # M6 — see §3.8
```

### 3.8 `NotionSink` (M6) — Notion layout (DESIGN_HE §Notion)
```text
Workspace page "🎓 לימודים"
├── DB Courses     (name, language, folder, cover)
├── DB Lectures    (title, course→Courses, week, date, status, exec_summary(rich text ≤ 5 bullets), material_claims(number), page body = Digest via markdown endpoint)
├── DB Glossary    (term, explanation, course→Courses, first_seen→Lectures, canonical_key, ★)
├── DB Claims      (statement, verdict select, lecturer_said, actually, source url, course→Courses, lecture→Lectures, importance)
└── DB Tasks       (task, due date, done checkbox, owner, course→Courses, lecture→Lectures)
Views: Lectures gallery (cover + week + exec summary) · Glossary table grouped by course · Claims filtered verdict≠correct · Tasks board by done / calendar by due · ★ view = Glossary + Highlights where ★
Page body: POST /v1/pages with `markdown` (callout for exec summary, <details> for full summary, H2 per Digest section). Idempotent: `lecture_id` stored in a hidden text property; re-sync updates in place.
```

### 3.9 `web/app.py`
```text
GET  /                → index.html (state from /api/state)
GET  /api/state       → {ready, ollama, mw, courses, previous_bullets, current_lecture|null, queue_depth, cost}
POST /api/record      {course_id, title, fact_check}     → starts LiveSource + worker
POST /api/replay      {course_id, file, pace}
POST /api/stop        → ends lecture, runs digest + sinks in background, returns digest path
POST /api/mark        {t: "now"}                          → highlight of last 30 s (kind=highlight, source=user)
POST /api/note        {text}                              → kind=note with timestamp
GET  /api/recap       → recap(lecture_id, 5)
GET  /api/search?q=   → store.search
WS   /ws              → pushes rows on every store write (items, claims, level, queue_depth)
```

## 4. Milestones and gates

| M | Dates | Delivers | Gate checks (`scripts/gate.py --m N`) |
|---|---|---|---|
| **M0 skeleton + stage-0** | 23–25.9 | repo, env `copilot`, `.cursor/rules`, `scripts/setup_models.sh`, fixture, `prompts/extract_v0.md`, stage-0 measurements in `docs/notes/STAGE0_HE.html` | `mw version` ok · 3 Ollama models present · sqlite-vec loads (or numpy fallback flagged) · fixture exists · `mw` 5 runs on 45 s: p50 logged, hot/cold verdict written · 10 real chunks → extract_v0 → ≥ 9/10 valid JSON · mic-from-seat test recorded (readable yes/no) · secret scan empty |
| **M1 pipeline** | 4–5.10 | `FileSource`, `vad.split`, `MacWhisperASR`, `extract`, `Store` (schema, ULID, upserts), `cli replay --pace fast` | fixture → segments ≥ 10, items ≥ 5, claims ≥ 1 · rerun = identical row counts (upsert) · every chunk ≤ budget on the fixture (p95 logged) · `ASRError` on a corrupt wav marks the chunk failed and continues · tests with stubs green |
| **M2 digest** | 6–7.10 | `digest()`, `FolderSink` (md + html + transcript + claims.json), `TranscriptSource` (.vtt + mw JSON), `prompts/digest_v0.md` | digest.md has all 9 sections in order · 60-min lecture → digest < 120 s wall · replay of a .vtt yields the same sections · course folder created under `COURSES_ROOT` · html renders RTL (screenshot) |
| **M3 memory** | 8–10.10 | `embed`, FTS5 + sqlite-vec, `search` (RRF), `remember` (already_said / contradicts), `canonical_key`, `previous_lecture`, continuation chapter, "ממשיך את" bullets | 7/6 then 9/6 replay: ≥ 80% of shared concepts flagged already_said · injected contradiction raises importance by 20 · search("CAC") returns the 9/6 explanation top-1 · continuation chapter present in 9/6 digest · numpy fallback passes the same tests |
| **M4 verifier + ranker + eval** | 11–12.10 | `verify` (Gemini, grounding, `get_course_context` tool, cache), verifier worker, offline mode, `rank`, `eval.py`, `benchmark.json` (labels from Sukkot) | eval: verdict accuracy ≥ 80% on 40 claims · Precision@5 material ≥ 80% · wifi off mid-replay → claims `unchecked`, batch-verified at stop · cost per lecture < $0.20 from `decisions` · cache hit on repeated claim · CI never calls Gemini |
| **M5 live product** | 13–15.10 | `LiveSource` + meter, `cli copilot` launcher (starts Ollama, checks mw, opens tab), web page before/during/after, ★ mark, notes, recap, crash-recovery banner, `course_page()` + search box | 20-min live simulation (YouTube lecture through speakers, 3 m): rows arrive, no popups, digest < 120 s after stop · kill process mid-lecture → reopen shows banner, resume finishes digest with chunks so far · meter reads > −40 dB within 10 s · course.html lists glossary across both lectures · RTL screenshots |
| **M6 Notion** | 16.10 | `NotionSink`: 5 DBs created once (`cli notion-init`), lecture page via markdown endpoint, glossary/claims/tasks rows, idempotent re-sync | after M5 replay: Lectures page exists with 9 sections · Glossary rows == distinct canonical_keys · re-sync twice → no duplicates · NOTION_TOKEN absent → sink skipped with a logged reason, folder sink still writes |
| **M7 buffer + Zoom** | 17.10 | first-lecture checklist (`docs/notes/FIRST_LECTURE_HE.html`); if time: BlackHole + Aggregate device, `source=zoom` | `copilot` reaches "ready" from a cold boot < 60 s · checklist walked once end-to-end · (zoom) 10-min Zoom test: both channels present, drift < 200 ms |

Cut ladder (spec): Zoom/BlackHole → `graph.py` LangGraph wrapper → re-rank → `flashcards.tsv` → `mlx.py` → `digest.html`.

## 5. Per-milestone protocol
See `CLAUDE.md` — branch, plain paragraph, tests, gate, Hebrew report, terminal summary, Obsidian, log, PR, Dor approves.

## 6. Standing rules
See `CLAUDE.md` §Standing rules. Plus: every LLM call is (prompt file, model, pydantic schema, one retry); every provider
failure is a logged row, never an exception that stops the lecture.

## 7. Dor's manual steps (collected; never done by Claude)

- [ ] M0: download `7/6 Lecture` and `9/6 Lecture` (video) from Drive `AI DEVELOPERS 11 / 2. Python / REC` into `data/lectures/`; copy their `.vtt` next to them.
- [x] M0: Gemini API key from aistudio.google.com → `.env` `GEMINI_API_KEY`. (2026-09-23)
- [x] M0: MacWhisper → Settings → Advanced → Install CLI; confirm `mw version`. (2026-09-23, 14.7.1)
- [x] M0: sit in the usual seat (or 3–5 m from a speaker) and record 1 minute for the mic test. (2026-09-23, speaker at 3–5 m, 2 min)
- [ ] Sukkot: label `eval/benchmark.json` from the two `.vtt` files (~3 h): 30 real claims + verdicts, 10 injected errors (5 contradicting 7/6), 30 concepts, shared-concept pairs.
- [ ] M2: `COURSES_ROOT` inside the Google Drive desktop folder; create course "יזמות וחדשנות" (he) and any English course (en).
- [ ] M6: Notion internal integration → `NOTION_TOKEN`; share the "🎓 לימודים" page with it; run `cli notion-init` once.
- [ ] M7 (optional): install BlackHole 2ch; Audio MIDI Setup: Multi-Output (speakers + BlackHole), Aggregate (mic + BlackHole, drift correction on).
- [ ] Before 18.10: run the first-lecture checklist.
