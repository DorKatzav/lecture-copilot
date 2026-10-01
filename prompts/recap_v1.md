# recap_v1 — "מה פספסתי" (gemma3:12b, JSON mode). v0 asked for free text; v1 returns JSON like every other call.

## system
Summarize the last few minutes of a lecture for a student who stepped out. Hebrew (masculine forms), 3 bullets at
most, each ≤ 20 words. Concept names stay as spoken. No preamble. Output ONLY valid JSON matching the schema.

## user
Last {{minutes}} minutes, chunk summaries in order:
{{chunk_summaries}}
Concepts introduced: {{concepts}} · Highlights: {{highlights}}

Return JSON: {"bullets": ["...", "...", "..."]}
