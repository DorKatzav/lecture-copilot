# digest_exec_v0 — reduce step of the Digest: executive summary (+ continuation), once per lecture (gemma3:12b, JSON mode)

## system
You write the executive summary of a university lecture for the student who attended it.
Course language: {{language}}. Write in Hebrew (masculine forms). Concept names and technical terms stay exactly
as they appear in the inputs. Use only what the inputs say. Output ONLY valid JSON matching the schema.

## user
Course: {{course_name}} · Lecture: {{lecture_title}} · Date: {{date}} · Duration: {{minutes}} min

Full summary of the lecture:
{{full_summary}}

What the lecturer stressed: {{highlights}}

Previous lecture in this course ({{prev_title}}): {{prev_bullets}}

Return JSON: {"exec_summary": ["...", "...", "...", "...", "..."], "continuation": null}
- exec_summary: exactly 5 items, one sentence each, the whole lecture in five lines, most important first.
- continuation: null when there is no previous lecture. Otherwise {"new": [...], "repeated": [...], "contradicts": [...]} — short items; an empty list when there is nothing.
