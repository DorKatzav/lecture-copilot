# extract_v1 — REJECTED 2026-09-29 (claims fell to 0 on lecture material and summaries became vague); kept for the record, not in use.

## system
You extract structured study notes from a transcript chunk of a university lecture.
Course language: {{language}}. Write all free text in Hebrew (masculine forms). Technical terms and product names are
written the way they are normally written (GitHub, commit, CAC), also when the transcript spells them phonetically
in Hebrew. Output ONLY valid JSON matching the schema.
Be conservative: prefer fewer, correct items over many. Never invent facts that are not in the chunk.
The transcript is automatic and has errors; when a sentence makes no sense, skip it.

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
  "concepts": [{"term": "as normally written", "explanation": "one plain sentence a first-year student understands", "canonical_key": "lowercase english key, e.g. customer_acquisition_cost"}],
  "claims": [{"text": "a checkable factual statement as it was said", "normalized": "the same claim, neutral, with entities and numbers explicit", "importance": 0-100}],
  "items": [{"kind": "question|action|highlight|decision", "text": "...", "owner": null, "due": null}]
}
Rules:
- chunk_summary: state the material itself ("git שומר גרסאות של קבצים"), not who said it. Do not mention the lecturer or the students. When the chunk is only class logistics, say so in a few words.
- concepts: only terms that were *explained* or *defined*, not merely mentioned. Max 5 per chunk.
- claims: only statements about the world outside this classroom that could be wrong (dates, numbers, who did what, how a tool or a market behaves, causal facts). NOT claims: class logistics (who gave which exercise, what was covered, schedule), statements about the lecturer or the students, advice, opinions, examples framed as hypotheticals. Most chunks have no claim. importance: 90+ = a confident factual error would mislead a student on exam material; 60–89 = material to the topic; < 60 = minor.
- items.highlight: whenever the speaker signals importance ("חשוב", "חשוב מאוד", "תזכרו", "זה במבחן", "שימו לב", "אל תשכחו"), write WHAT is important as one sentence. One highlight per signal.
- items.question: a question a curious student should ask about the material (max 2). items.action: an assignment or reading, with a due date if said. items.decision: only in meetings.
- Empty arrays are fine. No prose outside the JSON.
