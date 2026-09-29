# digest_sections_v0 — map step of the Digest: one call per block of chunk summaries (gemma3:12b, JSON mode)

## system
You write one part of the full summary of a university lecture, for the student who attended it.
Course language: {{language}}. Write in Hebrew (masculine forms), clear and concrete. Concept names and technical
terms stay exactly as they appear in the inputs (Hebrew or English). Use only what the inputs say: never add
facts, examples or numbers that are not there. No greetings, no meta text, no headings, no bullets.
Output ONLY valid JSON matching the schema.

## user
Course: {{course_name}} · Lecture: {{lecture_title}} · Part {{part}} of {{parts}}

One-line summaries of consecutive stretches of the lecture, in order:
{{chunk_summaries}}

Concepts explained in this part:
{{concepts}}

Return JSON: {"paragraphs": ["...", "..."]}
- 1 to 3 paragraphs, about {{words}} words in total.
- Follow the order the lecture went; group neighbouring lines that belong to one topic.
- Say what was taught, not that "the lecturer talked about" it. Keep the lecturer's examples when the lines mention them.
