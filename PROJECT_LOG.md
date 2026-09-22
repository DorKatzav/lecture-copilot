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
