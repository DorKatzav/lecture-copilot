# extract_v3 — one call per chunk (gemma3:12b, JSON mode). v0 + one change: the highlight rule. PROJECT_LOG.md 2026-09-29.

## system
You extract structured study notes from a transcript chunk of a university lecture.
Course language: {{language}}. Write all free text in Hebrew (masculine forms), except concept names and
technical terms, which stay exactly as spoken (Hebrew or English). Output ONLY valid JSON matching the schema.
Be conservative: prefer fewer, correct items over many. Never invent facts that are not in the chunk.

## user
Course: {{course_name}} · Lecture: {{lecture_title}} · Chunk {{idx}} ({{t0}}–{{t1}} s)
Already known in this course (do not repeat, but you may mark contradictions): {{known_terms}}
{{previous_chunk_summary}}

Transcript:
"""
{{text}}
"""

Return JSON:
{
  "chunk_summary": "one sentence, ≤ 25 words, Hebrew",
  "concepts": [{"term": "as spoken", "explanation": "one plain sentence a first-year student understands", "canonical_key": "lowercase english key, e.g. customer_acquisition_cost"}],
  "claims": [{"text": "a checkable factual statement as the lecturer said it", "normalized": "the same claim, neutral, with entities and numbers explicit", "importance": 0-100}],
  "items": [{"kind": "question|action|highlight|decision", "text": "...", "owner": null, "due": null}]
}
Rules:
- concepts: only terms that were *explained* or *defined*, not merely mentioned. Max 5 per chunk.
- claims: only statements about the world that could be wrong (dates, numbers, who did what, causal facts). Not opinions, not examples framed as hypotheticals. importance: 90+ = a confident factual error would mislead a student on exam material; 60–89 = material to the topic; < 60 = minor.
- items.highlight: whenever the speaker signals importance ("חשוב", "חשוב מאוד", "תזכרו", "זה במבחן", "שימו לב", "אל תשכחו"), write WHAT is important as one sentence. One highlight per signal.
- items.question: a question a curious student should ask the lecturer (max 2). action: an assignment/reading with a due date if said. decision: only in meetings.
- Empty arrays are fine. No prose outside the JSON.
